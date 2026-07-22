"""Observational instrumentation harness for realtime_main.RealtimeProcessor.

Everything here is additive: it subclasses RealtimeProcessor and overrides only the
methods needed to attach a forensics.person_creation.timeline_recorder.TimelineRecorder
to the segment lifecycle (open/close/enqueue/dequeue/drop) and the graph execution
(overall + per-node). No production file (realtime_main.py, presence_segmentation.py,
graph.py) is modified. Detection/clustering/ReID/Global-Memory-matching/VLM behavior is
untouched -- every overridden method calls the exact same underlying calls the base
class makes (gm.upsert_segment/set_segment_status/get_segment, graph.stream(), the same
node functions), just with recorder.record(...) calls layered around them.

Used by:
  - tools/test_realtime_timeline.py (deterministic, fakes only)
  - tools/profile_realtime_timeline.py (opt-in, real camera/models/Postgres)
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

from forensics.person_creation.presence_segmentation import PresenceGatedIngestion, SegmentBatch
from forensics.person_creation.realtime_main import RealtimeProcessor, _build_initial_state
from forensics.person_creation.timeline_recorder import TimelineRecorder, timed_node


def build_instrumented_graph(recorder: TimelineRecorder):
    """Mirrors forensics/person_creation/graph.py's node list and edges exactly, with
    every node wrapped in timed_node() for start/finish/failed timing. Kept as a
    separate builder (not a monkeypatch of graph.py) so production graph construction
    is completely untouched -- if graph.py's DAG shape changes, this needs a matching
    update, called out explicitly here rather than silently drifting.
    """
    from langgraph.graph import StateGraph, START
    from forensics.person_creation.state import PersonCreationState
    from forensics.person_creation.nodes.process_video import process_video
    from forensics.person_creation.nodes.process_live_stream import process_live_stream
    from forensics.person_creation.nodes.filter_quality import filter_quality
    from forensics.person_creation.nodes.embed_all_faces import embed_all_faces
    from forensics.person_creation.nodes.cluster_identities import cluster_identities
    from forensics.person_creation.nodes.assign_bodies_to_clusters import assign_bodies_to_clusters
    from forensics.person_creation.nodes.promote_crops import promote_crops
    from forensics.person_creation.nodes.select_best import select_best
    from forensics.person_creation.nodes.compute_reid import compute_reid
    from forensics.person_creation.nodes.build_profile import build_profile
    from forensics.person_creation.nodes.finalize import finalize

    def route_ingestion(state: PersonCreationState) -> str:
        return "process_live_stream" if state.get("input_type") == "camera_uri" else "process_video"

    raw_nodes = {
        "process_video": process_video,
        "process_live_stream": process_live_stream,
        "filter_quality": filter_quality,
        "embed_all_faces": embed_all_faces,
        "cluster_identities": cluster_identities,
        "assign_bodies_to_clusters": assign_bodies_to_clusters,
        "promote_crops": promote_crops,
        "select_best": select_best,
        "compute_reid": compute_reid,
        "build_profile": build_profile,
        "finalize": finalize,
    }

    builder = StateGraph(PersonCreationState)
    for name, fn in raw_nodes.items():
        builder.add_node(name, timed_node(name, fn, recorder))

    builder.add_conditional_edges(
        START,
        route_ingestion,
        {"process_video": "process_video", "process_live_stream": "process_live_stream"},
    )
    builder.add_edge("process_video", "filter_quality")
    builder.add_edge("process_live_stream", "filter_quality")
    builder.add_edge("filter_quality", "embed_all_faces")
    builder.add_edge("embed_all_faces", "cluster_identities")
    builder.add_edge("cluster_identities", "assign_bodies_to_clusters")
    builder.add_edge("assign_bodies_to_clusters", "promote_crops")
    builder.add_edge("promote_crops", "select_best")
    builder.add_edge("select_best", "compute_reid")
    builder.add_edge("compute_reid", "build_profile")
    builder.add_edge("build_profile", "finalize")
    from langgraph.graph import END

    builder.add_edge("finalize", END)
    return builder.compile()


class InstrumentedRealtimeProcessor(RealtimeProcessor):
    """RealtimeProcessor with a TimelineRecorder wired into segment lifecycle and
    graph execution. Every gm/graph call the base class makes is still made, in the
    same order -- this class only adds recorder.record(...) around them.
    """

    def __init__(
        self, *args: Any, recorder: TimelineRecorder, skip_load_models: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._recorder = recorder
        # Test-only seam: the deterministic test (tools/test_realtime_timeline.py) has
        # no GPU/models and injects detect_person_fn/graph_factory directly, so it has
        # no need for the real load_models() call. Off by default -- the real profiler
        # (tools/profile_realtime_timeline.py) never sets this.
        self._skip_load_models = skip_load_models

    # -- segment lifecycle (replaces the callbacks start() wires up) ---------------

    def _instrumented_on_segment_state_change(self, segment: SegmentBatch, status: str) -> None:
        if status == "CAPTURING":
            self._recorder.record(
                "segment_opened", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
            )
        elif status == "READY":
            self._recorder.record(
                "segment_closed",
                segment_id=segment.segment_id,
                segment_seq_num=segment.seq_num,
                details={
                    "close_reason": segment.close_reason,
                    "segment_incomplete": segment.segment_incomplete,
                    "frame_count": len(segment.frames),
                },
            )
            qdepth_before_push = self._ingestion.output_queue.qsize() if self._ingestion else 0
            maxsize = self._ingestion.output_queue.maxsize if self._ingestion else 0
            if maxsize and qdepth_before_push >= maxsize:
                self._recorder.record(
                    "queue_full", segment_id=segment.segment_id, segment_seq_num=segment.seq_num,
                    queue_depth=qdepth_before_push,
                )
        # Preserve the exact production behavior this callback normally performs
        # (Global Memory segments-table bookkeeping) -- unchanged, just also timed.
        self._on_segment_state_change(segment, status)
        if status == "READY":
            self._recorder.record(
                "segment_enqueued",
                segment_id=segment.segment_id,
                segment_seq_num=segment.seq_num,
                queue_depth=self._ingestion.output_queue.qsize() if self._ingestion else None,
            )

    def _instrumented_on_segment_dropped(self, segment: SegmentBatch) -> None:
        self._recorder.record(
            "segment_dropped", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
        )

    def start(self) -> None:
        if not self._skip_load_models:
            from forensics.person_creation.load_models import load_models

            self._reid_result = load_models()

        self._buffer = self._frame_source_factory()
        self._ingestion = PresenceGatedIngestion(
            frame_source=self._buffer,
            detect_person_fn=self._detect_person_fn,
            sample_interval_seconds=self.sample_interval_seconds,
            open_debounce_count=self.open_debounce_count,
            close_debounce_seconds=self.close_debounce_seconds,
            max_segment_seconds=self.max_segment_seconds,
            codec=self.codec,
            on_segment_state_change=self._instrumented_on_segment_state_change,
            on_segment_dropped=self._instrumented_on_segment_dropped,
        )
        self._wrap_add_frame_for_first_frame_timing()
        self._buffer.start()
        self._ingestion.start()

    def _wrap_add_frame_for_first_frame_timing(self) -> None:
        """Instance-only monkeypatch (not a class/module edit) so `first_frame_added`
        can be recorded without changing SegmentAccumulator's production code or
        behavior -- the original bound method is still called for every frame,
        unconditionally, before the recorder is touched.
        """
        accumulator = self._ingestion.accumulator
        original_add_frame = accumulator.add_frame

        def add_frame_with_timing(frame: Any, timestamp: str) -> None:
            current = accumulator._current
            is_first_frame = current is not None and len(current.frames) == 0
            segment_id = current.segment_id if current is not None else None
            segment_seq_num = current.seq_num if current is not None else None
            original_add_frame(frame, timestamp)
            if is_first_frame and segment_id is not None:
                self._recorder.record(
                    "first_frame_added", segment_id=segment_id, segment_seq_num=segment_seq_num
                )

        accumulator.add_frame = add_frame_with_timing

    # -- segment processing (mirrors RealtimeProcessor.process_segment, timed) -----

    def process_segment(self, segment: SegmentBatch) -> None:
        from forensics.global_memory import config as gm_config

        self._recorder.record(
            "segment_dequeued",
            segment_id=segment.segment_id,
            segment_seq_num=segment.seq_num,
            queue_depth=self._ingestion.output_queue.qsize() if self._ingestion else None,
        )

        with self._gm_close_lock:
            gm = self._get_gm()
            self._recorder.record(
                "segment_processing_started", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
            )
            gm.set_segment_status(segment.segment_id, gm_config.SEGMENT_STATUS_PROCESSING)

            self._recorder.record(
                "initial_state_build_started", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
            )
            initial_state = _build_initial_state(
                segment,
                camera_id=self.camera_id,
                camera_uri_masked=self._masked_uri(),
                output_dir=self.output_dir,
                process_every_n=self.process_every_n,
                reid_result=self._reid_result,
                pipeline_version=self.pipeline_version,
            )
            self._recorder.record(
                "initial_state_build_finished", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
            )

            try:
                graph = self._get_graph()
                self._recorder.record(
                    "graph_started", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
                )
                for _event in graph.stream(initial_state, stream_mode="updates"):
                    pass
                self._recorder.record(
                    "graph_finished", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
                )
                gm.set_segment_status(segment.segment_id, gm_config.SEGMENT_STATUS_SUCCEEDED)
                self.segments_processed += 1
                self._recorder.record(
                    "segment_processing_succeeded", segment_id=segment.segment_id, segment_seq_num=segment.seq_num
                )
            except Exception as exc:
                self._recorder.record(
                    "graph_failed",
                    segment_id=segment.segment_id,
                    segment_seq_num=segment.seq_num,
                    details={"error_type": type(exc).__name__},
                )
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
                self._recorder.record(
                    "segment_processing_failed",
                    segment_id=segment.segment_id,
                    segment_seq_num=segment.seq_num,
                    details={"error_type": type(exc).__name__},
                )

    def _masked_uri(self) -> str:
        from forensics.person_creation.live_stream import mask_camera_uri

        return mask_camera_uri(self.camera_uri)
