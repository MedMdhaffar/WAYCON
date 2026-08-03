from __future__ import annotations

import threading

from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk
from forensics.person_creation.live_core import LiveCoreInferenceSession
from forensics.person_creation.live_stream import BufferedFrame
from forensics.person_creation.nodes.process_live_stream import capture_live_chunk


class _Ledger:
    def __init__(self) -> None:
        self.committed = []

    def commit_preprocessed(self, chunk, result, **_kwargs) -> None:
        self.committed.append((chunk.chunk_index, result.chunk_index))


def _preprocessed(chunk):
    return PreprocessedLiveChunk(
        chunk_index=chunk.chunk_index,
        capture_summary=chunk.report_metrics(),
        quality_body_crops=[],
        quality_face_crops=[],
        face_embeddings=[],
        failed_face_embeddings=[],
        warnings=[],
        processing_elapsed_seconds=0.0,
    )


def _frame(index: int) -> BufferedFrame:
    return BufferedFrame(
        frame_idx=index,
        timestamp=f"2026-01-01T00:00:{index:02d}+00:00",
        frame=object(),
        captured_monotonic=float(index),
    )


def test_latest_frame_queue_is_capacity_two_and_drops_oldest():
    ledger = _Ledger()
    first_started = threading.Event()
    release_first = threading.Event()
    consumed = []

    def detect(_frame, *, frame_idx, **_kwargs):
        consumed.append(frame_idx)
        if frame_idx == 0:
            first_started.set()
            assert release_first.wait(2.0)
        return [], []

    session = LiveCoreInferenceSession(
        base_state={},
        preprocessing_session=ledger,
        person_detector=object(),
        face_detector=object(),
        body_dir="body",
        face_dir="face",
        source_metadata={},
        queue_capacity=2,
        detect=detect,
        preprocess=lambda *, chunk, **_kwargs: _preprocessed(chunk),
    )
    session.start()
    assert session.offer(_frame(0), chunk_index=0, source_stem="live")
    assert first_started.wait(1.0)
    assert session.offer(_frame(1), chunk_index=0, source_stem="live")
    assert session.offer(_frame(2), chunk_index=0, source_stem="live")
    assert session.offer(_frame(3), chunk_index=0, source_stem="live")
    before = session.public_snapshot()
    assert before["core_queue_capacity"] == 2
    assert before["core_queue_depth"] == 2
    assert before["core_queue_peak"] == 2
    assert before["frames_dropped_core_queue_oldest"] == 1

    release_first.set()
    final = session.finish(2.0)
    assert final["core_drain_timed_out"] is False
    assert consumed == [0, 2, 3]
    assert len(ledger.committed) == 3
    assert final["frames_consumed_by_person_detector"] == 3
    assert final["frames_sent_to_face_detector"] == 3
    assert final["core_queue_unfinished_tasks"] == 0


def test_slow_analysis_outside_core_does_not_starve_recent_frame_consumption():
    ledger = _Ledger()
    analysis_blocked = threading.Event()
    analysis_release = threading.Event()
    consumed = []

    def blocked_analysis():
        analysis_blocked.set()
        analysis_release.wait(2.0)

    analysis_thread = threading.Thread(target=blocked_analysis)
    analysis_thread.start()
    assert analysis_blocked.wait(1.0)

    session = LiveCoreInferenceSession(
        base_state={},
        preprocessing_session=ledger,
        person_detector=object(),
        face_detector=object(),
        body_dir="body",
        face_dir="face",
        source_metadata={},
        detect=lambda _frame, *, frame_idx, **_kwargs: (
            consumed.append(frame_idx) or ([], [])
        ),
        preprocess=lambda *, chunk, **_kwargs: _preprocessed(chunk),
    )
    session.start()
    for index in range(12):
        session.offer(_frame(index), chunk_index=0, source_stem="live")
    final = session.finish(2.0)
    analysis_release.set()
    analysis_thread.join(1.0)

    assert final["core_drain_timed_out"] is False
    assert consumed
    assert consumed[-1] == 11
    assert final["core_queue_peak"] <= 2
    assert final["frames_dropped_core_queue_oldest"] > 0


def test_core_drain_is_bounded_and_abandons_no_queue_accounting():
    ledger = _Ledger()
    started = threading.Event()
    release = threading.Event()

    def detect(_frame, **_kwargs):
        started.set()
        release.wait(2.0)
        return [], []

    session = LiveCoreInferenceSession(
        base_state={},
        preprocessing_session=ledger,
        person_detector=object(),
        face_detector=object(),
        body_dir="body",
        face_dir="face",
        source_metadata={},
        detect=detect,
        preprocess=lambda *, chunk, **_kwargs: _preprocessed(chunk),
    )
    session.start()
    session.offer(_frame(0), chunk_index=0, source_stem="live")
    assert started.wait(1.0)
    session.offer(_frame(1), chunk_index=0, source_stem="live")
    session.offer(_frame(2), chunk_index=0, source_stem="live")

    final = session.finish(0.01)
    assert final["core_drain_timed_out"] is True
    assert final["core_queue_depth"] == 0
    assert "deadline" in final["core_worker_error"]
    release.set()


def test_regular_chunk_clock_is_not_extended_by_slow_downstream_analysis():
    class Clock:
        value = 0.0

        def monotonic(self):
            return self.value

    class Buffer:
        def __init__(self, clock):
            self.clock = clock
            self.frames_read = 0
            self.frames_dropped = 0
            self.error = None
            self.ended = False

        @property
        def empty(self):
            return False

        def get(self, **_kwargs):
            self.clock.value += 1.0
            index = self.frames_read
            self.frames_read += 1
            return _frame(index)

        def stats(self, frames_processed, warnings=None):
            return {
                "frames_read": self.frames_read,
                "frames_processed": frames_processed,
                "frames_dropped": self.frames_dropped,
                "warnings": list(warnings or []),
            }

    class Core:
        def __init__(self):
            self.offered = []

        def offer(self, frame, **_kwargs):
            self.offered.append(frame.frame_idx)
            return True

        def records_snapshot(self):
            return {"chunk_body_crops": {}, "chunk_face_crops": {}}

    clock = Clock()
    buffer = Buffer(clock)
    core = Core()
    result = capture_live_chunk(
        buffer=buffer,
        duration_seconds=10.0,
        every_n=3,
        source_stem="live",
        source_metadata={},
        body_dir="body",
        face_dir="face",
        person_detector=object(),
        face_detector=object(),
        monotonic=clock.monotonic,
        core_session=core,
    )

    assert result.elapsed_seconds == 10.0
    assert result.frames_processed == 3
    assert core.offered == [0, 3, 6]


def test_core_worker_exception_is_exposed_and_requests_stop():
    stop_requested = threading.Event()
    session = LiveCoreInferenceSession(
        base_state={},
        preprocessing_session=_Ledger(),
        person_detector=object(),
        face_detector=object(),
        body_dir="body",
        face_dir="face",
        source_metadata={},
        detect=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("detector exploded")
        ),
        preprocess=lambda *, chunk, **_kwargs: _preprocessed(chunk),
        request_stop=stop_requested.set,
    )
    session.start()
    session.offer(_frame(0), chunk_index=0, source_stem="live")
    assert stop_requested.wait(1.0)
    snapshot = session.finish(1.0)

    assert snapshot["core_worker_alive"] is False
    assert "RuntimeError" in snapshot["core_worker_error"]
    assert snapshot["core_queue_depth"] == 0
