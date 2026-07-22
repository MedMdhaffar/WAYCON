"""Long-lived realtime process: one camera, continuous presence-gated capture.

    RTSP camera -> load_models() -> GStreamer pipeline (gst_stream.GstFrameBuffer)
    -> Tier 0 presence gate -> segment accumulator -> queue.Queue(maxsize=2)
    -> sync pipeline (graph.stream() once per closed segment)
    -> PostgreSQL (persons / segments / camera_events)
                 \\-> clothing_jobs -> async VLM worker (vlm_worker.py) -> appearances

This is the "one resident process continuously consuming one camera" the realtime
architecture calls for. service.py no longer launches a pipeline run per HTTP
request for camera input -- see its module docstring. Run this as its own
long-lived OS process, separate from:
  - service.py: a lightweight monitoring/query API (+ occasional offline
    video-file enrollment jobs), and
  - vlm_worker.py: started here by default (one worker per realtime process is
    the common case), but runnable standalone too if you'd rather split it out.

Usage:
    python -m forensics.person_creation.realtime_main \
        --camera-uri "rtsp://admin:Waycon2026@10.0.0.104:554/Streaming/Channels/101" \
        --camera-id cam1

    python -m forensics.person_creation.realtime_main \
        --camera-uri "rtsp://localhost:8554/test" \
        --camera-id cam1

            
"""

from __future__ import annotations

import argparse
import queue
import signal
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from forensics.person_creation.live_stream import mask_camera_uri
from forensics.person_creation.presence_segmentation import PresenceGatedIngestion, SegmentBatch


def _build_initial_state(
    segment: SegmentBatch,
    *,
    camera_id: str | None,
    camera_uri_masked: str,
    output_dir: Path,
    process_every_n: int,
    reid_result: dict,
    pipeline_version: str | None,
) -> dict:
    return {
        "person_name": f"camera_{camera_id or 'live'}",
        "video_paths": [],
        "input_type": "camera_uri",
        "source_type": "live_camera",
        "camera_id": camera_id,
        "source_uri_masked": camera_uri_masked,
        "output_dir": str(output_dir / segment.segment_id),
        "process_every_n": process_every_n,
        "identity_clustering_config": {},
        "body_crops": [],
        "face_crops": [],
        "segment_id": segment.segment_id,
        "segment_seq_num": segment.seq_num,
        "segment_start_ts": segment.segment_start_ts,
        "segment_end_ts": segment.segment_end_ts,
        "segment_incomplete": segment.segment_incomplete,
        "segment_frames": segment.frames,
        "segment_frame_timestamps": segment.frame_timestamps,
        "pipeline_version": pipeline_version,
        **reid_result,
    }


class RealtimeProcessor:
    """Owns ingestion, segment consumption, graph execution, and segment/
    camera-event bookkeeping for one camera. Split out from main() so it's
    testable without argparse/signal handling/an actual camera or GPU -- see
    `frame_source_factory`/`graph_factory`/`gm_factory` below, all test seams.
    """

    def __init__(
        self,
        camera_uri: str,
        *,
        camera_id: str | None = None,
        codec: str = "h264",
        decoder: str | None = None,
        output_dir: str | Path = "forensics/person_db/_realtime",
        process_every_n: int = 5,
        sample_interval_seconds: float = 0.35,
        open_debounce_count: int = 2,
        close_debounce_seconds: float = 2.5,
        max_segment_seconds: float = 10.0,
        buffer_max_size: int = 30,
        stall_timeout_seconds: float = 5.0,
        pipeline_version: str | None = None,
        frame_source_factory: Callable[..., Any] | None = None,
        graph_factory: Callable[[], Any] | None = None,
        gm_factory: Callable[[], Any] | None = None,
        detect_person_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        self.camera_uri = camera_uri
        self.camera_id = camera_id
        self.codec = codec
        self.decoder = decoder
        self.output_dir = Path(output_dir)
        self.process_every_n = max(1, int(process_every_n))
        self.sample_interval_seconds = sample_interval_seconds
        self.open_debounce_count = open_debounce_count
        self.close_debounce_seconds = close_debounce_seconds
        self.max_segment_seconds = max_segment_seconds
        self.buffer_max_size = buffer_max_size
        self.stall_timeout_seconds = stall_timeout_seconds
        self.pipeline_version = pipeline_version
        self._frame_source_factory = frame_source_factory or self._default_frame_source_factory
        self._graph_factory = graph_factory or self._default_graph_factory
        self._gm_factory = gm_factory or self._default_gm_factory
        self._detect_person_fn = detect_person_fn or self._default_detect_person_fn

        self._stop_event = threading.Event()
        self._buffer = None
        self._ingestion: PresenceGatedIngestion | None = None
        self._graph = None
        self._gm_instance = None
        # Guards gm close-vs-in-flight-use: stop() can be called from another
        # thread (e.g. an HTTP request via RealtimeManager) while run_forever's
        # thread is still inside process_segment() using self._gm_instance --
        # without this, stop() closing the connection pool mid-query raised
        # psycopg_pool.PoolClosed and crashed the processing thread.
        self._gm_close_lock = threading.Lock()
        self._reid_result: dict = {}

        self.segments_processed = 0
        self.segments_failed = 0

    # -- test seams / lazy singletons ---------------------------------------------

    def _default_frame_source_factory(self):
        from forensics.person_creation.gst_stream import GstFrameBuffer

        return GstFrameBuffer(
            self.camera_uri,
            max_size=self.buffer_max_size,
            codec=self.codec,
            decoder=self.decoder,
            stall_timeout_seconds=self.stall_timeout_seconds,
            on_disconnect=self._on_camera_disconnect,
            on_reconnected=self._on_camera_reconnected,
            on_state_change=self._on_connection_state_change,
        )

    def _default_graph_factory(self):
        from forensics.person_creation.graph import build_graph

        return build_graph()

    def _default_gm_factory(self):
        from forensics.global_memory import GlobalMemory

        return GlobalMemory()

    def _default_detect_person_fn(self, frame) -> bool:
        from forensics.person_creation.models.person_detector import get_person_detector

        return len(get_person_detector().detect(frame)) > 0

    def _get_gm(self):
        if self._gm_instance is None:
            self._gm_instance = self._gm_factory()
        return self._gm_instance

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._graph_factory()
        return self._graph

    # -- Postgres-facing callbacks --------------------------------------------------

    def _on_segment_state_change(self, segment: SegmentBatch, status: str) -> None:
        gm = self._get_gm()
        gm.upsert_segment(
            segment_id=segment.segment_id,
            seq_num=segment.seq_num,
            codec=segment.codec,
            pipeline_version=self.pipeline_version,
            segment_start_ts=segment.segment_start_ts,
            segment_end_ts=segment.segment_end_ts,
            status=status,
            segment_incomplete=segment.segment_incomplete,
        )

    def _on_camera_disconnect(self, reason: str) -> None:
        self._get_gm().log_camera_event("disconnected", reason=reason)

    def _on_camera_reconnected(self) -> None:
        self._get_gm().log_camera_event("reconnected", reason=None)

    def _on_connection_state_change(self, state) -> None:
        if self._ingestion is not None:
            self._ingestion.on_connection_state_change(state)

    # -- segment processing (the "sync pipeline" box) ------------------------------

    def process_segment(self, segment: SegmentBatch) -> None:
        """Run the graph once for one already-captured segment, with segment
        PROCESSING -> SUCCEEDED/FAILED_* bookkeeping around it. Public (not
        prefixed `_`) because tests call it directly to check the state-machine
        transitions without running the full ingestion loop.
        """
        from forensics.global_memory import config as gm_config

        # Held for the whole segment (not just the gm calls) so stop() -- which
        # acquires this same lock before closing the pool -- always waits for an
        # in-flight segment to finish its own bookkeeping first, instead of
        # racing to close the pool underneath it.
        with self._gm_close_lock:
            gm = self._get_gm()
            gm.set_segment_status(segment.segment_id, gm_config.SEGMENT_STATUS_PROCESSING)

            initial_state = _build_initial_state(
                segment,
                camera_id=self.camera_id,
                camera_uri_masked=mask_camera_uri(self.camera_uri),
                output_dir=self.output_dir,
                process_every_n=self.process_every_n,
                reid_result=self._reid_result,
                pipeline_version=self.pipeline_version,
            )

            try:
                graph = self._get_graph()
                for _event in graph.stream(initial_state, stream_mode="updates"):
                    pass
                gm.set_segment_status(segment.segment_id, gm_config.SEGMENT_STATUS_SUCCEEDED)
                self.segments_processed += 1
                print(f"[realtime_main] segment={segment.segment_id} -> SUCCEEDED")
            except Exception:
                error = traceback.format_exc()
                current = gm.get_segment(segment.segment_id)
                retry_count = int(current["retry_count"]) if current else 0
                next_status = (
                    gm_config.SEGMENT_STATUS_FAILED_FINAL
                    if retry_count + 1 >= gm_config.SEGMENT_MAX_RETRIES
                    else gm_config.SEGMENT_STATUS_FAILED_RETRYABLE
                )
                gm.set_segment_status(segment.segment_id, next_status, error=error[-2000:], increment_retry=True)
                self.segments_failed += 1
                print(f"[realtime_main] segment={segment.segment_id} -> {next_status}\n{error}")

    # -- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        from forensics.person_creation.load_models import load_models

        self._reid_result = load_models()

        self._buffer = self._frame_source_factory()   #this is an object from the class  GstFrameBuffer
        self._ingestion = PresenceGatedIngestion(
            frame_source=self._buffer,
            detect_person_fn=self._detect_person_fn,
            sample_interval_seconds=self.sample_interval_seconds,
            open_debounce_count=self.open_debounce_count,
            close_debounce_seconds=self.close_debounce_seconds,
            max_segment_seconds=self.max_segment_seconds,
            codec=self.codec,
            on_segment_state_change=self._on_segment_state_change,
        )
        self._buffer.start()
        self._ingestion.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ingestion is not None:
            self._ingestion.stop()
        if self._buffer is not None:
            self._buffer.stop()
        # Wait for any in-flight process_segment() to finish its own gm calls
        # (see the lock in process_segment) before closing the pool underneath it.
        with self._gm_close_lock:
            if self._gm_instance is not None:
                self._gm_instance.close()
                self._gm_instance = None

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                try:
                    segment = self._ingestion.output_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self.process_segment(segment)
        finally:
            self.stop()

    # -- debugging / monitoring ---------------------------------------------------

    def stats(self) -> dict:
        """Snapshot for /api/realtime/status -- safe to call from another thread
        (only reads plain attributes / thread-safe sub-objects, no locking needed
        since every field here is either append-only counters or an Enum/str set
        atomically by its owning thread).
        """
        buffer_stats = self._buffer.stats(0) if self._buffer is not None else {}
        presence_state = self._ingestion.gate.state.value if self._ingestion is not None else "unknown"
        queue_depth = self._ingestion.output_queue.qsize() if self._ingestion is not None else 0
        frames_seen = self._ingestion.frames_seen if self._ingestion is not None else 0
        samples_taken = self._ingestion.samples_taken if self._ingestion is not None else 0
        return {
            "camera_id": self.camera_id,
            "camera_uri_masked": mask_camera_uri(self.camera_uri),
            "running": not self._stop_event.is_set(),
            "connection_state": buffer_stats.get("connection_state"),
            "reconnect_count": buffer_stats.get("reconnect_count"),
            "last_disconnect_reason": buffer_stats.get("last_disconnect_reason"),
            "presence_state": presence_state,
            "segment_queue_depth": queue_depth,
            "frames_seen": frames_seen,
            "samples_taken": samples_taken,
            "segments_processed": self.segments_processed,
            "segments_failed": self.segments_failed,
        }


class RealtimeManager:
    """In-process registry of running RealtimeProcessors, keyed by camera_id.

    Lets service.py start/stop/inspect camera realtime capture as background
    threads inside the same long-lived process instead of requiring a separate
    `python -m realtime_main` OS process per camera -- one process, one set of
    resident models (load_models() is idempotent/cached, see load_models.py),
    shared by both offline video-file jobs and any number of realtime cameras.
    A dedicated OS process (this module's __main__/main()) remains available for
    anyone who wants a camera fully isolated from the monitoring API's process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}

    def start(self, camera_uri: str, *, camera_id: str, start_vlm_worker: bool = True, **kwargs) -> dict:
        with self._lock:
            existing = self._entries.get(camera_id)
            if existing is not None and existing["thread"].is_alive():
                raise ValueError(f"camera_id '{camera_id}' is already running")

            processor = RealtimeProcessor(camera_uri, camera_id=camera_id, **kwargs)
            vlm_worker = None
            if start_vlm_worker:
                from forensics.person_creation.vlm_worker import VLMWorker

                vlm_worker = VLMWorker()
                vlm_worker.start()

            thread = threading.Thread(
                target=processor.run_forever,
                name=f"realtime-{camera_id}",
                daemon=True,
            )
            self._entries[camera_id] = {
                "processor": processor,
                "thread": thread,
                "vlm_worker": vlm_worker,
            }
            thread.start()
            return processor.stats()

    def stop(self, camera_id: str, *, timeout: float = 10.0) -> dict:
        with self._lock:
            entry = self._entries.get(camera_id)
        if entry is None:
            raise KeyError(f"camera_id '{camera_id}' is not running")
        stats = entry["processor"].stats()
        entry["processor"].stop()
        if entry["vlm_worker"] is not None:
            entry["vlm_worker"].stop()
        entry["thread"].join(timeout=timeout)
        with self._lock:
            self._entries.pop(camera_id, None)
        return stats

    def status(self, camera_id: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(camera_id)
        if entry is None:
            return None
        return entry["processor"].stats()

    def list(self) -> list[dict]:
        with self._lock:
            entries = list(self._entries.items())
        return [entry["processor"].stats() for _camera_id, entry in entries]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-uri", required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--decoder", default=None, help="e.g. 'nvv4l2decoder ! nvvideoconvert' on Jetson")
    parser.add_argument("--output-dir", default="forensics/person_db/_realtime")
    parser.add_argument("--process-every-n", type=int, default=5)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.35)
    parser.add_argument("--open-debounce-count", type=int, default=2)
    parser.add_argument("--close-debounce-seconds", type=float, default=2.5)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--no-vlm-worker", action="store_true", help="don't start the async VLM worker in this process")
    args = parser.parse_args()

    processor = RealtimeProcessor(
        args.camera_uri,
        camera_id=args.camera_id,
        codec=args.codec,
        decoder=args.decoder,
        output_dir=args.output_dir,
        process_every_n=args.process_every_n,
        sample_interval_seconds=args.sample_interval_seconds,
        open_debounce_count=args.open_debounce_count,
        close_debounce_seconds=args.close_debounce_seconds,
        max_segment_seconds=args.segment_seconds,
    )

    vlm_worker = None
    if not args.no_vlm_worker:
        from forensics.person_creation.vlm_worker import VLMWorker

        vlm_worker = VLMWorker()
        vlm_worker.start()
        print("[realtime_main] async VLM worker started")

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    processor_thread = threading.Thread(target=processor.run_forever, name="realtime-processor", daemon=True)
    processor_thread.start()
    print(f"[realtime_main] running for {mask_camera_uri(args.camera_uri)}. Ctrl+C to stop.")

    stop_event.wait()
    print("[realtime_main] stopping...")
    processor.stop()
    if vlm_worker is not None:
        vlm_worker.stop()
    processor_thread.join(timeout=10.0)
    print(
        f"[realtime_main] stopped. segments_processed={processor.segments_processed} "
        f"segments_failed={processor.segments_failed}"
    )


if __name__ == "__main__":
    main()
