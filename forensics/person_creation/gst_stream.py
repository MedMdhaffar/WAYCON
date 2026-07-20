"""GStreamer-based bounded live-camera capture, replacing OpenCV's LiveFrameBuffer.

Pipeline shape: rtspsrc -> rtp{h264,h265}depay -> parse -> decoder -> videoconvert -> appsink,
running inside one GLib MainLoop thread per buffer instance. The decoder element is
configurable so the same code runs with a software decoder (avdec_h264/avdec_h265, used in
dev/CI) or an NVIDIA hardware decoder (nvv4l2decoder, used on the deployment box) without any
other code change.

Connection handling is a small state machine (CONNECTED <-> RECONNECTING) driven by:
  - the GStreamer bus (ERROR / EOS messages), and
  - an appsink watchdog thread that declares a stall if no frame arrives within
    `stall_timeout_seconds`.
On either trigger, the pipeline is torn down and rebuilt with exponential backoff
(`reconnect_backoff_seconds`, capped at `reconnect_backoff_cap_seconds`). `on_disconnect` and
`on_reconnected` callbacks let a caller (e.g. the segment accumulator) force-close and tag an
open segment as `segment_incomplete` and mark presence as "unknown" for the duration of the
outage -- see forensics/person_creation/nodes/process_live_stream.py for the intended caller.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from forensics.person_creation.live_stream import (  # noqa: E402
    capture_source_from_uri,
    mask_camera_uri,
)

_GST_INITIALIZED = False
_GST_INIT_LOCK = threading.Lock()


def _ensure_gst_initialized() -> None:
    global _GST_INITIALIZED
    if _GST_INITIALIZED:
        return
    with _GST_INIT_LOCK:
        if not _GST_INITIALIZED:
            Gst.init(None)
            _GST_INITIALIZED = True


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConnectionState(str, Enum):
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


@dataclass
class BufferedFrame:
    frame_idx: int
    timestamp: str
    frame: Any  # numpy ndarray, BGR, HxWx3


# Software decoders by default (portable, works without a GPU). Override with e.g.
# "nvv4l2decoder ! nvvideoconvert" on a Jetson/L4T box, or "vaapih264dec" on an
# Intel/VAAPI box.
_DEFAULT_DECODERS = {
    "h264": "avdec_h264",
    "h265": "avdec_h265",
}


class GstFrameBuffer:
    """Continuously read an RTSP stream via GStreamer, retaining at most the latest N frames.

    Interface-compatible with `live_stream.LiveFrameBuffer` (start/get/stop/stats/empty),
    plus connection-state tracking and reconnect callbacks that OpenCV's VideoCapture-based
    buffer has no equivalent for.
    """

    def __init__(
        self,
        uri: str,
        max_size: int = 30,
        codec: str = "h264",
        decoder: str | None = None,
        latency_ms: int = 200,
        protocols: str = "tcp",
        stall_timeout_seconds: float = 5.0,
        reconnect_backoff_seconds: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 60),
        reconnect_backoff_cap_seconds: float = 60.0,
        on_disconnect: Callable[[str], None] | None = None,
        on_reconnected: Callable[[], None] | None = None,
        on_state_change: Callable[[ConnectionState], None] | None = None,
    ) -> None:
        _ensure_gst_initialized()

        self.uri = uri
        self.max_size = max(1, int(max_size))
        self.codec = codec.lower()
        if self.codec not in _DEFAULT_DECODERS:
            raise ValueError(f"Unsupported codec: {codec!r} (expected 'h264' or 'h265')")
        self.decoder = decoder or _DEFAULT_DECODERS[self.codec]
        self.latency_ms = max(0, int(latency_ms))
        self.protocols = protocols
        self.stall_timeout_seconds = max(0.5, float(stall_timeout_seconds))
        self.reconnect_backoff_seconds = tuple(reconnect_backoff_seconds) or (1.0,)
        self.reconnect_backoff_cap_seconds = float(reconnect_backoff_cap_seconds)
        self._on_disconnect = on_disconnect
        self._on_reconnected = on_reconnected
        self._on_state_change = on_state_change

        # Public, read-only-by-convention stats (mirrors LiveFrameBuffer's fields).
        self.frames_read = 0
        self.frames_dropped = 0
        self.first_frame_time: str | None = None
        self.last_frame_time: str | None = None
        self.stream_opened = False
        self.error: str | None = None
        self.ended = False

        # Reconnect bookkeeping.
        self.state: ConnectionState = ConnectionState.STOPPED
        self.reconnect_count = 0
        self.last_disconnect_reason: str | None = None

        self._queue: queue.Queue[BufferedFrame] = queue.Queue(maxsize=self.max_size)
        self._stop_event = threading.Event()
        self._reconnect_trigger = threading.Event()
        self._pending_reconnect_reason: str | None = None
        self._lock = threading.RLock()

        self._pipeline: Gst.Pipeline | None = None
        self._appsink = None
        self._bus_watch_id: int | None = None
        self._last_frame_monotonic = time.monotonic()

        self._main_loop: GLib.MainLoop | None = None
        self._main_loop_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._reconnect_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ build/teardown

    def _build_pipeline_string(self) -> str:
        source = capture_source_from_uri(self.uri)
        depay = "rtph264depay" if self.codec == "h264" else "rtph265depay"
        parse = "h264parse" if self.codec == "h264" else "h265parse"
        return (
            f'rtspsrc location="{source}" latency={self.latency_ms} '
            f'protocols={self.protocols} name=src ! '
            f"{depay} ! {parse} ! {self.decoder} ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
        )

    def _build_pipeline(self) -> None:
        pipeline_str = self._build_pipeline_string()
        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_new_sample)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus_watch_id = bus.connect("message", self._on_bus_message)

        self._pipeline = pipeline
        self._appsink = appsink
        self._bus_watch_id = bus_watch_id

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._teardown_pipeline()
            raise OSError(f"GStreamer pipeline failed to reach PLAYING for {mask_camera_uri(self.uri)}")

    def _teardown_pipeline(self) -> None:
        if self._pipeline is not None:
            bus = self._pipeline.get_bus()
            if self._bus_watch_id is not None:
                try:
                    bus.disconnect(self._bus_watch_id)
                except Exception:
                    pass
                self._bus_watch_id = None
            bus.remove_signal_watch()
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None

    # ------------------------------------------------------------------ appsink / bus callbacks

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        frame = _sample_to_bgr_ndarray(sample)
        self._last_frame_monotonic = time.monotonic()
        if frame is None:
            return Gst.FlowReturn.OK

        timestamp = utc_now_iso()
        with self._lock:
            frame_idx = self.frames_read
            self.frames_read += 1
            if self.first_frame_time is None:
                self.first_frame_time = timestamp
            self.last_frame_time = timestamp

        item = BufferedFrame(frame_idx=frame_idx, timestamp=timestamp, frame=frame)
        if self._queue.full():
            try:
                self._queue.get_nowait()
                with self._lock:
                    self.frames_dropped += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.frames_dropped += 1
        return Gst.FlowReturn.OK

    def _on_bus_message(self, _bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._request_reconnect(f"gst_error: {err} ({debug})")
        elif message.type == Gst.MessageType.EOS:
            self._request_reconnect("gst_eos")

    # ------------------------------------------------------------------ reconnect state machine

    def _request_reconnect(self, reason: str) -> None:
        with self._lock:
            if self.state == ConnectionState.RECONNECTING or self._stop_event.is_set():
                return
            self.last_disconnect_reason = reason
            self._pending_reconnect_reason = reason
        self._reconnect_trigger.set()

    def _set_state(self, state: ConnectionState) -> None:
        with self._lock:
            self.state = state
        if self._on_state_change is not None:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def _reconnect_loop(self) -> None:
        """Runs in a dedicated thread; owns pipeline teardown/rebuild during outages."""
        while not self._stop_event.is_set():
            triggered = self._reconnect_trigger.wait(timeout=0.5)
            if not triggered or self._stop_event.is_set():
                continue
            self._reconnect_trigger.clear()

            with self._lock:
                reason = self._pending_reconnect_reason or "unknown"
                self._pending_reconnect_reason = None

            self._set_state(ConnectionState.RECONNECTING)
            self.error = reason
            if self._on_disconnect is not None:
                try:
                    self._on_disconnect(reason)
                except Exception:
                    pass

            self._teardown_pipeline()

            backoff_idx = 0
            while not self._stop_event.is_set():
                delay = self.reconnect_backoff_seconds[
                    min(backoff_idx, len(self.reconnect_backoff_seconds) - 1)
                ]
                delay = min(delay, self.reconnect_backoff_cap_seconds)
                if self._stop_event.wait(timeout=delay):
                    break
                try:
                    self._build_pipeline()
                    self._last_frame_monotonic = time.monotonic()
                    with self._lock:
                        self.reconnect_count += 1
                        self.error = None
                        self.ended = False
                    self._set_state(ConnectionState.CONNECTED)
                    if self._on_reconnected is not None:
                        try:
                            self._on_reconnected()
                        except Exception:
                            pass
                    break
                except Exception as exc:  # noqa: BLE001
                    self.error = f"reconnect_failed: {exc}"
                    backoff_idx += 1

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.5)
            if self.state != ConnectionState.CONNECTED:
                continue
            if time.monotonic() - self._last_frame_monotonic > self.stall_timeout_seconds:
                self._request_reconnect(
                    f"appsink_stall: no frame for {self.stall_timeout_seconds:g}s"
                )

    # ------------------------------------------------------------------ public API

    def start(self) -> None:
        self._main_loop = GLib.MainLoop()
        self._main_loop_thread = threading.Thread(
            target=self._main_loop.run,
            name="gst-frame-buffer-mainloop",
            daemon=True,
        )
        self._main_loop_thread.start()

        try:
            self._build_pipeline()
        except OSError:
            self._stop_event.set()
            if self._main_loop is not None:
                self._main_loop.quit()
            raise

        self.stream_opened = True
        self._last_frame_monotonic = time.monotonic()
        self._set_state(ConnectionState.CONNECTED)

        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, name="gst-frame-buffer-reconnect", daemon=True
        )
        self._reconnect_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="gst-frame-buffer-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def get(self, timeout: float = 1.0) -> BufferedFrame | None:
        try:
            return self._queue.get(timeout=max(0.01, float(timeout)))
        except queue.Empty:
            return None

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    def stop(self) -> None:
        self._stop_event.set()
        self._reconnect_trigger.set()
        self._set_state(ConnectionState.STOPPED)
        self._teardown_pipeline()
        if self._main_loop is not None:
            self._main_loop.quit()
        for thread in (self._reconnect_thread, self._watchdog_thread, self._main_loop_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def stats(self, frames_processed: int, warnings: list[str] | None = None) -> dict:
        with self._lock:
            return {
                "stream_opened": self.stream_opened,
                "frames_read": self.frames_read,
                "frames_processed": int(frames_processed),
                "frames_dropped": self.frames_dropped,
                "buffer_max_size": self.max_size,
                "first_frame_time": self.first_frame_time,
                "last_frame_time": self.last_frame_time,
                "warnings": list(warnings or []),
                "connection_state": self.state.value,
                "reconnect_count": self.reconnect_count,
                "last_disconnect_reason": self.last_disconnect_reason,
            }


# ---------------------------------------------------------------------------------------
# GPU-resident decode (NOT ENABLED by default -- see the discussion this came out of).
#
# The pipeline above always lands frames in host (CPU) memory: `nvvideoconvert`/`videoconvert`
# converts out of NVMM into a system-memory BGR buffer that `_sample_to_bgr_ndarray` maps with
# a plain `buf.map()`. That's a deliberate simplification for now, not the final state --
# BufferedFrame.frame is consumed today as a host-memory np.ndarray by
# nodes/process_video.py / nodes/process_live_stream.py (cv2.imwrite crops to disk,
# CPU-side face/person detection calls), so switching to GPU-resident buffers here without
# also changing those consumers would just move the CPU<->GPU copy from inside GStreamer to
# inside the first node that touches `frame`, not eliminate it.
#
# The zero-copy version -- worth doing once §3 (collapse face_engine in-process) and the
# detection nodes are rewritten to accept GPU tensors natively -- looks like this:
#
# 1. Decoder choice:
#    - Jetson / L4T (embedded, unified memory):        nvv4l2decoder
#    - Discrete GPU / dGPU (e.g. this RTX 5080 box):    nvh264dec / nvh265dec (nvcodec plugin,
#      part of gstreamer1.0-plugins-bad; confirm with `gst-inspect-1.0 nvh264dec`)
#
# 2. Pipeline stays in NVMM the whole way instead of converting to system memory:
#
#     PIPELINE_NVMM = (
#         f'rtspsrc location="{source}" latency={self.latency_ms} protocols={self.protocols} name=src ! '
#         f'{depay} ! {parse} ! '
#         f'nvh264dec ! '                                   # or nvv4l2decoder on Jetson
#         f'nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! '  # stays in NVMM, no host copy
#         f'appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false '
#         f'caps=video/x-raw(memory:NVMM),format=RGBA'
#     )
#
# 3. Mapping an NVMM Gst.Buffer into a CUDA tensor without a host round-trip requires one of:
#      a) DeepStream's Python bindings (`pyds`) + `gst-nvdsbufferpool`, which give you a
#         `pyds.NvBufSurface` you can wrap as a CuPy/torch tensor via
#         `pyds.get_nvds_buf_surface()` -- the standard path on Jetson/DeepStream deployments.
#      b) On a discrete GPU without DeepStream: import the buffer's EGLImage/DMABUF via
#         `Gst.Buffer` -> `GstAllocator`'s NVMM memory -> `nvbuf_utils`/`cudaGraphicsGLRegisterImage`
#         (or `cuGraphicsEGLRegisterImage`), producing a CUDA device pointer you can wrap with
#         `torch.as_tensor(..., device="cuda")` (via `torch.utils.dlpack` or a raw CUDA IPC/array
#         interface) -- more manual, no single canonical Python binding covers this cleanly today.
#
#    Rough shape of (b), sketched (not runnable as-is -- needs nvbuf_utils/pycuda/dlpack glue
#    that isn't vendored anywhere in this repo yet):
#
#     def _nvmm_sample_to_cuda_tensor(sample: Gst.Sample):
#         buf = sample.get_buffer()
#         # NVMM memory exposes a dmabuf fd via Gst.Memory when the allocator supports it:
#         mem = buf.peek_memory(0)
#         ok, dmabuf_fd = mem.get_dmabuf_fd()  # illustrative; exact API depends on nvbuf_utils version
#         if not ok:
#             return None
#         # Import the dmabuf fd into a CUDA device pointer (needs pycuda or a small C extension;
#         # cuGraphicsEGLRegisterImage / cuImportExternalMemory are the relevant CUDA driver calls):
#         cuda_ptr, pitch = _import_dmabuf_as_cuda_pointer(dmabuf_fd)  # not implemented
#         import torch
#         tensor = torch.as_tensor(
#             _cuda_ptr_to_dlpack(cuda_ptr, shape=(height, width, 4), dtype="uint8"),
#             device="cuda",
#         )  # not implemented -- placeholder for the dlpack-from-raw-pointer glue
#         return tensor
#
# 4. This changes BufferedFrame.frame's contract from "np.ndarray, BGR, host memory" to
#    "cuda tensor, device memory" -- a breaking change for every current consumer. Do this as
#    part of §3, alongside rewriting nodes/process_video.py and process_live_stream.py's
#    detect_and_save_frame() to run YOLO/face-embed directly on the GPU tensor and only copy
#    to host memory for the handful of crops that actually get written to disk (JPEG staging),
#    instead of copying every full frame to host memory as today.
# ---------------------------------------------------------------------------------------


def _sample_to_bgr_ndarray(sample: Gst.Sample):
    import numpy as np

    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")

    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
        expected = width * height * 3
        if frame.size < expected:
            return None
        frame = frame[:expected].reshape((height, width, 3)).copy()
        return frame
    finally:
        buf.unmap(mapinfo)
