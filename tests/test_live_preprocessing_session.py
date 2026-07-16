from __future__ import annotations

from copy import deepcopy
import json
import threading
import time

import pytest

from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk
from forensics.person_creation.live_session import (
    LivePreprocessingSession,
    LivePreprocessingSessionError,
)
from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


def _crop(index: int, kind: str) -> dict:
    return {
        "path": f"{kind}-{index}.jpg",
        "frame_idx": index,
        "video": "camera-source",
        "bbox": [0, 0, 120, 180] if kind == "body" else [0, 0, 80, 80],
        "sharpness": 100.0,
    }


def _chunk(index: int, *, empty: bool = False) -> LiveChunkResult:
    return LiveChunkResult(
        chunk_index=index,
        started_at=float(index),
        elapsed_seconds=10.0,
        stop_requested=False,
        frames_read=10,
        frames_processed=2,
        frames_skipped=8,
        frames_dropped=0,
        body_crops=[] if empty else [_crop(index, "body")],
        face_crops=[] if empty else [_crop(index, "face")],
        body_detection_count=0 if empty else 1,
        face_detection_count=0 if empty else 1,
        warnings=[],
    )


def _result(chunk: LiveChunkResult, *, warning: str | None = None) -> PreprocessedLiveChunk:
    face_embeddings = []
    if chunk.face_crops:
        face_embeddings = [{
            "crop_path": chunk.face_crops[0]["path"],
            "embedding": [0.25, 0.75],
        }]
    return PreprocessedLiveChunk(
        chunk_index=chunk.chunk_index,
        capture_summary=chunk.report_metrics(),
        quality_body_crops=deepcopy(chunk.body_crops),
        quality_face_crops=deepcopy(chunk.face_crops),
        face_embeddings=face_embeddings,
        failed_face_embeddings=[],
        warnings=[warning] if warning else [],
        processing_elapsed_seconds=0.01,
    )


def test_queue_capacity_is_two_and_exactly_one_worker_exists():
    entered = threading.Event()
    release = threading.Event()

    def preprocess(*, chunk, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        return _result(chunk)

    session = LivePreprocessingSession(base_state={}, preprocess=preprocess)
    assert session.queue_capacity == 2
    assert session._queue.maxsize == 2

    session.start()
    session.submit_chunk(_chunk(0))
    assert entered.wait(1.0)
    workers = [
        thread
        for thread in threading.enumerate()
        if thread.name == "person-creation-live-preprocessing"
    ]
    assert len(workers) == 1
    release.set()
    session.finish()
    assert not session.worker_alive


def test_backpressure_loses_no_chunks_and_preserves_order():
    entered = threading.Event()
    release = threading.Event()
    processed: list[int] = []

    def preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            entered.set()
            assert release.wait(2.0)
        processed.append(chunk.chunk_index)
        return _result(chunk)

    session = LivePreprocessingSession(
        base_state={}, preprocess=preprocess, queue_retry_seconds=0.01
    )
    session.start()
    session.submit_chunk(_chunk(0))
    assert entered.wait(1.0)
    session.submit_chunk(_chunk(1))
    session.submit_chunk(_chunk(2))

    producer_errors: list[Exception] = []

    def submit_final() -> None:
        try:
            session.submit_chunk(_chunk(3))
        except Exception as exc:  # pragma: no cover - assertion captures details
            producer_errors.append(exc)

    producer = threading.Thread(target=submit_final)
    producer.start()
    time.sleep(0.05)
    assert producer.is_alive()

    release.set()
    producer.join(2.0)
    session.finish()

    assert not producer_errors
    assert processed == [0, 1, 2, 3]
    records = session.records_snapshot()
    assert records["capture_completed_chunk_indices"] == [0, 1, 2, 3]
    assert records["preprocessing_completed_chunk_indices"] == [0, 1, 2, 3]
    assert [item["path"] for item in records["raw_body_crops"]] == [
        f"body-{index}.jpg" for index in range(4)
    ]


def test_stop_with_full_queue_drains_final_chunk_before_sentinel():
    entered = threading.Event()
    release = threading.Event()
    processed: list[int] = []

    def preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            entered.set()
            assert release.wait(2.0)
        processed.append(chunk.chunk_index)
        return _result(chunk)

    session = LivePreprocessingSession(
        base_state={}, preprocess=preprocess, queue_retry_seconds=0.01
    )
    session.start()
    session.submit_chunk(_chunk(0))
    assert entered.wait(1.0)
    session.submit_chunk(_chunk(1))
    session.submit_chunk(_chunk(2))

    finish_errors: list[Exception] = []

    def finish() -> None:
        try:
            session.finish()
        except Exception as exc:  # pragma: no cover - assertion captures details
            finish_errors.append(exc)

    finisher = threading.Thread(target=finish)
    finisher.start()
    time.sleep(0.05)
    assert finisher.is_alive()
    release.set()
    finisher.join(2.0)

    assert not finish_errors
    assert processed == [0, 1, 2]
    assert not session.worker_alive


def test_duplicate_chunk_index_is_rejected_without_duplicate_raw_records():
    session = LivePreprocessingSession(
        base_state={}, preprocess=lambda *, chunk, **_kwargs: _result(chunk)
    )
    session.start()
    session.submit_chunk(_chunk(5))
    with pytest.raises(ValueError, match="Duplicate live chunk index: 5"):
        session.submit_chunk(_chunk(5))
    session.finish()

    records = session.records_snapshot()
    assert records["capture_completed_chunk_indices"] == [5]
    assert len(records["raw_body_crops"]) == 1


def test_empty_chunk_is_processed_once():
    processed: list[int] = []

    def preprocess(*, chunk, **_kwargs):
        processed.append(chunk.chunk_index)
        return _result(chunk)

    session = LivePreprocessingSession(base_state={}, preprocess=preprocess)
    session.start()
    session.submit_chunk(_chunk(0, empty=True))
    session.finish()

    assert processed == [0]
    assert session.public_snapshot()["preprocessing_completed_chunks"] == 1


def test_phase_one_only_mode_creates_no_rolling_frozen_evidence():
    session = LivePreprocessingSession(
        base_state={}, preprocess=lambda *, chunk, **_kwargs: _result(chunk)
    )
    session.start()
    session.submit_chunk(_chunk(0))
    session.finish()

    assert session.rolling_analysis_enabled is False
    assert session._frozen_identity_config is None
    assert session._analysis_quality_body_chunks == ()
    assert session._analysis_quality_face_chunks == ()
    assert session._analysis_embedding_chunks == ()
    assert session._analysis_membership_chunks == ()
    assert session._analysis_membership_paths == set()
    with pytest.raises(LivePreprocessingSessionError, match="evidence is disabled"):
        session.analysis_snapshot()


def test_worker_failure_unblocks_full_queue_and_requests_stop():
    entered = threading.Event()
    fail_now = threading.Event()
    stop_requested = threading.Event()
    producer_errors: list[Exception] = []

    def preprocess(*, chunk, **_kwargs):
        entered.set()
        assert fail_now.wait(2.0)
        raise RuntimeError(
            "embedding failed for rtsp://user:secret@camera.local/private"
        )

    session = LivePreprocessingSession(
        base_state={},
        preprocess=preprocess,
        request_stop=stop_requested.set,
        queue_retry_seconds=0.01,
    )
    session.start()
    session.submit_chunk(_chunk(0))
    assert entered.wait(1.0)
    session.submit_chunk(_chunk(1))
    session.submit_chunk(_chunk(2))

    def blocked_submit() -> None:
        try:
            session.submit_chunk(_chunk(3))
        except Exception as exc:
            producer_errors.append(exc)

    producer = threading.Thread(target=blocked_submit)
    producer.start()
    time.sleep(0.05)
    assert producer.is_alive()
    fail_now.set()
    producer.join(2.0)

    assert stop_requested.is_set()
    assert len(producer_errors) == 1
    assert isinstance(producer_errors[0], LivePreprocessingSessionError)
    with pytest.raises(LivePreprocessingSessionError, match="chunk 0"):
        session.finish()
    serialized = json.dumps(session.public_snapshot())
    assert "rtsp://" not in serialized
    assert "user" not in serialized
    assert "secret" not in serialized


def test_public_snapshots_are_json_safe_and_embedding_vector_free():
    uri = "rtsp://user:secret@camera.local/private"
    session = LivePreprocessingSession(
        base_state={},
        preprocess=lambda *, chunk, **_kwargs: _result(
            chunk, warning=f"safe this warning: {uri}"
        ),
    )
    session.start()
    session.submit_chunk(_chunk(0))
    session.finish()

    public = session.public_snapshot()
    records = session.records_snapshot()
    serialized = json.dumps({"public": public, "records": records})
    assert public["embedded_faces"] == 1
    assert "face_embeddings" not in records
    assert "[0.25, 0.75]" not in serialized
    assert "rtsp://" not in serialized
    assert "secret" not in serialized


def test_join_timeout_cannot_report_success_while_worker_is_alive():
    worker_entered = threading.Event()
    release_worker = threading.Event()

    def preprocess(*, chunk, **_kwargs):
        worker_entered.set()
        release_worker.wait()
        return _result(chunk)

    session = LivePreprocessingSession(
        base_state={},
        preprocess=preprocess,
        join_timeout_seconds=0.1,
        queue_retry_seconds=0.01,
    )
    session.start()
    session.submit_chunk(_chunk(0))
    assert worker_entered.wait(1.0)

    try:
        with pytest.raises(LivePreprocessingSessionError, match="did not stop"):
            session.finish()
        assert session.public_snapshot()["worker_failed"] is True
    finally:
        release_worker.set()
        session._thread.join(1.0)

    assert not session.worker_alive


def test_full_queue_bounds_sentinel_insertion_and_reports_failure():
    worker_entered = threading.Event()
    release_worker = threading.Event()
    finish_returned = threading.Event()
    finish_errors: list[Exception] = []

    def preprocess(*, chunk, **_kwargs):
        if chunk.chunk_index == 0:
            worker_entered.set()
            release_worker.wait()
        return _result(chunk)

    session = LivePreprocessingSession(
        base_state={},
        preprocess=preprocess,
        queue_capacity=1,
        join_timeout_seconds=0.1,
        queue_retry_seconds=0.01,
    )
    session.start()
    session.submit_chunk(_chunk(0))
    assert worker_entered.wait(1.0)
    session.submit_chunk(_chunk(1))

    def finish() -> None:
        try:
            session.finish()
        except Exception as exc:
            finish_errors.append(exc)
        finally:
            finish_returned.set()

    finisher = threading.Thread(target=finish)
    finisher.start()
    try:
        assert finish_returned.wait(1.0)
        assert len(finish_errors) == 1
        assert isinstance(finish_errors[0], LivePreprocessingSessionError)
        assert "sentinel was not accepted within" in str(finish_errors[0]).lower()
        assert session.public_snapshot()["worker_failed"] is True
    finally:
        release_worker.set()
        finisher.join(1.0)
        session._thread.join(1.0)

    assert not finisher.is_alive()
    assert not session.worker_alive
