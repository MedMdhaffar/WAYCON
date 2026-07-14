"""Layer 8 — real-time (RTSP / live camera) processing skeleton.

Design (see docs/gpu_pipeline_architecture_analysis.md §7):

    RTSP reader / NVDEC decoder thread
            ↓  LatestFrameQueue (bounded, drop-oldest, counters)
    GPU detector worker (person → face, CUDA tensors, batch=1)
            ↓
    tracker / identity worker / event output (downstream, out of scope here)

Hard rules implemented:
 - the queue NEVER grows unbounded: when full, the oldest frame is dropped and
   counted, so the consumer always sees the newest frames and latency stays
   bounded;
 - queue depth defaults to 2 — the RTX 2050 has 4 GB VRAM, so no deep GPU
   frame buffer is permissible (one 1080p RGB uint8 tensor ≈ 6 MB, but decoder
   surfaces and model activations dominate);
 - camera id and capture timestamps travel with every frame;
 - decode-loop crashes trigger bounded-backoff reconnection;
 - the clothing VLM is never called here — accepted body crops must be handed
   to a *separate* bounded VLM queue owned by an async worker.

STATUS: the queue and reconnect logic are unit-tested (tests/test_realtime_queue.py);
end-to-end RTSP behavior needs a live camera and is validated in a later sprint.
PyNvVideoCodec's demuxer accepts RTSP URLs; OpenCVDecoder is the fallback there
too (cv2 FFMPEG build supports rtsp://).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from forensics.person_creation.gpu.decoder import DecodedFrame, create_decoder


@dataclass
class QueueStats:
    put_count: int = 0
    dropped_count: int = 0
    get_count: int = 0


@dataclass
class TimedFrame:
    camera_id: str
    capture_monotonic: float
    capture_epoch: float
    frame: DecodedFrame


class LatestFrameQueue:
    """Bounded frame queue that discards the *oldest* entry when full.

    The consumer therefore always processes the newest available frame and
    end-to-end latency is bounded by (maxsize × frame interval).
    """

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._items: deque[TimedFrame] = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.stats = QueueStats()

    def put(self, item: TimedFrame) -> None:
        with self._not_empty:
            if len(self._items) >= self._maxsize:
                self._items.popleft()
                self.stats.dropped_count += 1
            self._items.append(item)
            self.stats.put_count += 1
            self._not_empty.notify()

    def get(self, timeout: float | None = None) -> TimedFrame | None:
        with self._not_empty:
            if not self._items and not self._not_empty.wait_for(
                lambda: bool(self._items), timeout=timeout
            ):
                return None
            self.stats.get_count += 1
            return self._items.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class DecodeWorker(threading.Thread):
    """Owns one decoder for one source; feeds a LatestFrameQueue and
    reconnects with bounded exponential backoff on stream failure."""

    def __init__(
        self,
        source: str,
        camera_id: str,
        queue: LatestFrameQueue,
        use_nvdec: bool | None = None,
        reconnect_initial_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        frame_hook: Callable[[TimedFrame], None] | None = None,
    ) -> None:
        super().__init__(name=f"decode-{camera_id}", daemon=True)
        self.source = source
        self.camera_id = camera_id
        self.queue = queue
        self.use_nvdec = use_nvdec
        self.reconnect_initial_seconds = reconnect_initial_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        self.frame_hook = frame_hook
        self.reconnect_count = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        backoff = self.reconnect_initial_seconds
        while not self._stop_event.is_set():
            decoder = None
            try:
                decoder = create_decoder(self.source, use_nvdec=self.use_nvdec)
                backoff = self.reconnect_initial_seconds
                while not self._stop_event.is_set():
                    frame = decoder.read()
                    if frame is None:
                        break  # end of stream / connection lost
                    item = TimedFrame(
                        camera_id=self.camera_id,
                        capture_monotonic=time.monotonic(),
                        capture_epoch=time.time(),
                        frame=frame,
                    )
                    if self.frame_hook is not None:
                        self.frame_hook(item)
                    self.queue.put(item)
            except Exception as exc:
                print(f"[realtime:{self.camera_id}] decode error: {exc!r}")
            finally:
                if decoder is not None:
                    decoder.close()
            if self._stop_event.is_set():
                return
            self.reconnect_count += 1
            print(
                f"[realtime:{self.camera_id}] stream ended — reconnect #{self.reconnect_count} "
                f"in {backoff:.1f}s (dropped so far: {self.queue.stats.dropped_count})"
            )
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, self.reconnect_max_seconds)
