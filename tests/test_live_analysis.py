from __future__ import annotations

import json
import sqlite3
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from forensics.global_memory import GlobalMemory
from forensics.person_creation.live_analysis import (
    LiveAnalysisLifecycleError,
    LiveRollingAnalysisSession,
)
from forensics.person_creation.live_chunk_processing import PreprocessedLiveChunk
from forensics.person_creation.live_session import (
    FrozenAnalysisSnapshot,
    FrozenRecord,
    LivePreprocessingSession,
)
from forensics.person_creation.nodes.identity_config import load_identity_config
from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


def _record(value: dict) -> FrozenRecord:
    return FrozenRecord.from_mapping(value)


def test_dbscan_defaults_remain_unchanged(monkeypatch):
    for name in ("EPS", "MIN_SAMPLES", "MIN_CLUSTER_FACE_COUNT"):
        monkeypatch.delenv(f"PERSON_CREATION_IDENTITY_{name}", raising=False)

    assert load_identity_config({}) == {
        "eps": 0.4,
        "min_samples": 3,
        "min_cluster_face_count": 3,
    }


def _snapshot(version: int, paths: tuple[str, ...], *, chunk: int | None = None):
    embeddings = tuple(
        _record({
            "crop_path": path,
            "embedding": [1.0, float(index)],
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0 + index,
        })
        for index, path in enumerate(paths)
    )
    faces = tuple(
        _record({
            "path": path,
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0 + index,
        })
        for index, path in enumerate(paths)
    )
    return FrozenAnalysisSnapshot(
        version=version,
        last_completed_preprocessing_chunk=chunk,
        person_name="Test",
        video_paths=("camera-source",),
        identity_clustering_config=_record({"eps": 0.4, "min_samples": 2}),
        quality_body_crops=(),
        quality_face_crops=faces,
        face_embeddings=embeddings,
        face_chunk_membership=tuple((path, chunk or 0) for path in paths),
    )


def _cluster_all(state: dict) -> dict:
    records = list(state["all_face_embeddings"])
    if not records:
        return {"identity_clusters": [], "unresolved_faces": []}
    return {
        "identity_clusters": [{
            "cluster_id": 0,
            "face_records": records,
            "representative_embedding": [1.0, 0.0],
            "face_count": len(records),
            "confidence": 1.0,
            "low_confidence": False,
        }],
        "unresolved_faces": [],
    }


def _association(_state):
    return SimpleNamespace(cluster_assignments={0: []})


def _wait_for(session: LiveRollingAnalysisSession, version: int) -> dict:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        snapshot = session.public_snapshot()
        if snapshot["analysis_version"] >= version or (
            snapshot["requested_version"] >= version
            and snapshot["analysis_state"] == "warning"
            and not snapshot["analysis_in_progress"]
        ):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"analysis version {version} was not completed")


def _database_contents(path: Path) -> dict[str, list[tuple]]:
    connection = sqlite3.connect(str(path))
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: connection.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid'
            ).fetchall()
            for table in tables
        }
    finally:
        connection.close()


def test_successful_analysis_publishes_compact_embedding_free_status(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=3)]
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)
    session.finish(1)

    assert status["analysis_state"] == "ready"
    assert status["evidence_version"] == 1
    assert status["publication_sequence"] >= 0
    assert status["generated_at"]
    assert status["analyzed_embedding_count"] == 1
    assert status["last_completed_preprocessing_chunk"] == 3
    assert status["live_identities"][0]["session_person_id"] == "live_0001"
    serialized = json.dumps(status)
    assert "all_face_embeddings" not in serialized
    assert "[1.0, 0.0]" not in serialized
    assert "database" not in serialized


def test_committed_live_state_contains_production_final_stages(tmp_path):
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",), chunk=3),
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    _wait_for(session, 1)
    session.finish(1)

    canonical = session.canonical_state()
    assert canonical["identity_clusters"][0]["face_count"] == 1
    assert canonical["per_cluster_best_body_crops"] == {0: []}
    assert canonical["reid_reasons"][0] == "no_reid_model_configured"
    assert canonical["per_cluster_profiles"][0]["cluster_id"] == 0


def test_rolling_dbscan_exception_is_fail_open_and_final_pass_can_exit(tmp_path):
    stop_requested = threading.Event()

    def fail(_state):
        raise RuntimeError("dbscan exploded with secret details")

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=fail,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)

    session.finish(1)

    assert not stop_requested.is_set()
    assert not session.worker_alive
    assert status["analysis_state"] == "warning"
    assert "RuntimeError" in status["analysis_warning"]
    assert "secret details" not in status["analysis_warning"]


def test_global_memory_error_is_recoverable_and_connection_is_closed(tmp_path):
    database = tmp_path / "memory.db"
    database.touch()
    stop_requested = threading.Event()

    class FailingMemory:
        closed = False

        def query_by_face(self, *_args, **_kwargs):
            raise OSError("database is locked")

        def close(self):
            self.closed = True

    memory = FailingMemory()
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
        memory_factory=lambda _path: memory,
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)
    session.finish(1)

    assert status["analysis_state"] == "warning"
    assert not stop_requested.is_set()
    assert memory.closed is True


def test_rolling_memory_lookup_preserves_every_table_value(tmp_path):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(str(database))
    writer.register({
        "face_embedding": [1.0, 0.0],
        "face_crops": ["known-face.jpg"],
        "appearance": {"date": "2026-07-15"},
        "best_body_crops": [],
        "video_sources": [],
        "cameras": [],
    })
    writer.close()
    before = _database_contents(database)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)
    session.finish(1)

    assert status["live_identities"][0]["memory_match"]["person_id"] == "person_001"
    assert _database_contents(database) == before


def test_hung_analysis_worker_blocks_safe_return(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def hang(state):
        entered.set()
        release.wait()
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=hang,
        associate=_association,
        database_path=tmp_path / "missing.db",
        join_timeout_seconds=0.1,
    )
    session.start()
    session.request_version(1)
    assert entered.wait(1.0)
    try:
        with pytest.raises(LiveAnalysisLifecycleError, match="staging must be preserved"):
            session.finish(1)
        assert session.worker_alive
    finally:
        release.set()
        session._thread.join(1.0)

    assert not session.worker_alive


def test_unexpected_dead_worker_does_not_forge_user_stop(tmp_path):
    stop_requested = threading.Event()
    current = [_snapshot(1, ("face-a.jpg",))]
    calls = 0

    def die_after_valid_result(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit("unexpected worker death")
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=die_after_valid_result,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    valid = _wait_for(session, 1)
    current[0] = _snapshot(2, ("face-a.jpg", "face-b.jpg"))
    session.request_version(2)
    deadline = time.monotonic() + 2.0
    while session.worker_alive and time.monotonic() < deadline:
        time.sleep(0.01)

    session.finish(2)
    failed = session.public_snapshot()

    assert not stop_requested.is_set()
    assert not session.worker_alive
    assert failed["analysis_state"] == "worker_failed"
    assert failed["analysis_version"] == 1
    assert failed["live_identities"] == valid["live_identities"]


def test_no_new_embeddings_advance_progress_without_rerun_or_duplicate_events(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    calls = []

    def cluster(state):
        calls.append(len(state["all_face_embeddings"]))
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=cluster,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    first = _wait_for(session, 1)
    current[0] = _snapshot(2, ("face-a.jpg",), chunk=1)
    session.request_version(2)
    second = _wait_for(session, 2)
    session.finish(2)

    assert calls == [1]
    assert second["last_completed_preprocessing_chunk"] == 1
    assert second["live_recognition_events"] == first["live_recognition_events"]


def test_body_only_update_reuses_unchanged_dbscan_but_refreshes_association(
    tmp_path,
):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    cluster_calls = 0
    association_calls = 0

    def cluster(state):
        nonlocal cluster_calls
        cluster_calls += 1
        return _cluster_all(state)

    def associate(_state):
        nonlocal association_calls
        association_calls += 1
        return SimpleNamespace(cluster_assignments={0: []})

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=cluster,
        associate=associate,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    _wait_for(session, 1)
    current[0] = replace(
        _snapshot(2, ("face-a.jpg",), chunk=1),
        quality_body_crops=(_record({
            "path": str(tmp_path / "body.jpg"),
            "frame_idx": 2,
            "video": "camera-source",
            "bbox": [0, 0, 80, 160],
            "sharpness": 100.0,
        }),),
    )
    session.request_version(2)
    _wait_for(session, 2)
    session.finish(2)

    assert cluster_calls == 1
    assert association_calls == 2


def test_semantic_events_are_not_repeated_at_newer_analysis_version(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    first = _wait_for(session, 1)
    current[0] = _snapshot(2, ("face-a.jpg", "face-b.jpg"), chunk=1)
    session.request_version(2)
    second = _wait_for(session, 2)
    session.finish(2)

    assert [event["type"] for event in first["live_recognition_events"]] == [
        "identity_created",
        "memory_no_match",
    ]
    assert second["live_recognition_events"] == first["live_recognition_events"]


def test_memory_transition_events_cover_found_changed_and_lost(tmp_path):
    database = tmp_path / "memory.db"
    database.touch()
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]

    class ChangingMemory:
        def __init__(self):
            self.calls = 0

        def query_by_face(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [{"person_id": "person_001", "name": "One", "similarity": 0.9}]
            if self.calls == 2:
                return [{"person_id": "person_002", "name": "Two", "similarity": 0.8}]
            return []

        def close(self):
            return None

    memory = ChangingMemory()
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
        memory_factory=lambda _path: memory,
    )
    session.start()
    for version in (1, 2, 3):
        current[0] = _snapshot(
            version,
            tuple(f"face-{index}.jpg" for index in range(version)),
            chunk=version - 1,
        )
        session.request_version(version)
        status = _wait_for(session, version)
    session.finish(3)

    event_types = [event["type"] for event in status["live_recognition_events"]]
    assert "memory_match_found" in event_types
    assert "memory_match_changed" in event_types
    assert "memory_match_lost" in event_types


def test_internal_and_public_event_histories_are_bounded_to_100(tmp_path):
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(0, ()),
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    events, keys, next_number = session._prepare_events(
        raw_events=tuple(
            {
                "type": "identity_created",
                "session_person_id": f"live_{index:04d}",
                "analysis_version": index,
            }
            for index in range(125)
        ),
        previous_events=(),
        previous_keys=(),
        next_event_number=1,
    )
    with session._condition:
        session._events = events
        session._event_keys = keys
        session._next_event_number = next_number

    status = session.public_snapshot()
    assert len(session._event_keys) == 100
    assert len(status["live_recognition_events"]) == 100


def test_latest_only_request_does_not_publish_stale_result(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def cluster(state):
        calls.append(tuple(item["crop_path"] for item in state["all_face_embeddings"]))
        if len(calls) == 1:
            entered.set()
            assert release.wait(2.0)
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=cluster,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    assert entered.wait(1.0)
    current[0] = _snapshot(2, ("face-a.jpg", "face-b.jpg"), chunk=1)
    session.request_version(2)
    release.set()
    status = _wait_for(session, 2)
    session.finish(2)

    assert len(calls) == 2
    assert status["analysis_version"] == 2
    assert status["live_identities"][0]["face_count"] == 2


def test_missing_memory_does_not_create_parent_or_database(tmp_path):
    database = tmp_path / "new-parent" / "memory.db"
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
    )
    session.start()
    session.request_version(1)
    _wait_for(session, 1)
    session.finish(1)

    assert not database.parent.exists()
    assert not database.exists()


def test_temporary_node_state_cannot_mutate_accumulator_or_previous_snapshot():
    chunk = LiveChunkResult(
        chunk_index=0,
        started_at=0.0,
        elapsed_seconds=1.0,
        stop_requested=False,
        frames_read=1,
        frames_processed=1,
        frames_skipped=0,
        frames_dropped=0,
        body_crops=[],
        face_crops=[],
        body_detection_count=0,
        face_detection_count=0,
        warnings=[],
    )

    def preprocess(**_kwargs):
        return PreprocessedLiveChunk(
            chunk_index=0,
            capture_summary={},
            quality_body_crops=[],
            quality_face_crops=[{
                "path": "face-a.jpg",
                "frame_idx": 0,
                "video": "camera-source",
                "bbox": [0, 0, 80, 80],
                "sharpness": 100.0,
            }],
            face_embeddings=[{
                "crop_path": "face-a.jpg",
                "embedding": [1.0, 0.0],
                "frame_idx": 0,
                "video": "camera-source",
                "bbox": [0, 0, 80, 80],
                "sharpness": 100.0,
            }],
            failed_face_embeddings=[],
            warnings=[],
            processing_elapsed_seconds=0.01,
        )

    accumulator = LivePreprocessingSession(
        base_state={}, preprocess=preprocess, rolling_analysis=True
    )
    accumulator.start()
    accumulator.submit_chunk(chunk)
    accumulator.finish()
    previous = accumulator.analysis_snapshot()
    before = deepcopy(previous.mutable_node_state())

    temporary = previous.mutable_node_state()
    temporary["all_face_embeddings"][0]["embedding"][0] = 999.0
    temporary["quality_face_crops"][0]["bbox"].append(999)
    temporary["all_face_embeddings"].clear()

    current = accumulator.analysis_snapshot()
    assert previous.mutable_node_state() == before
    assert current.mutable_node_state() == before


def test_status_callback_failure_is_recoverable(tmp_path):
    callback_calls = []

    def fail_callback(_snapshot):
        callback_calls.append(True)
        raise RuntimeError("callback details")

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
        notify=fail_callback,
    )
    session.start()
    session.request_version(1)
    _wait_for(session, 1)
    session.finish(1)

    assert callback_calls
    assert not session.worker_alive
    assert "status publication" in session.public_snapshot()["analysis_warning"]


def test_event_failure_is_transactional_and_worker_retries(tmp_path, monkeypatch):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=_cluster_all,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    original_prepare = session._prepare_events
    fail_once = [True]

    def injected_failure(**kwargs):
        if fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("event insertion failed")
        return original_prepare(**kwargs)

    monkeypatch.setattr(session, "_prepare_events", injected_failure)
    before = {
        "memberships": dict(session._active_memberships),
        "retired": session._retired_live_ids,
        "next_id": session._next_live_number,
        "matches": deepcopy(session._memory_matches),
        "events": deepcopy(session._events),
        "identities": deepcopy(session._live_identities),
    }
    session.start()
    session.request_version(1)
    failed = _wait_for(session, 1)

    assert failed["analysis_version"] == 0
    assert dict(session._active_memberships) == before["memberships"]
    assert session._retired_live_ids == before["retired"]
    assert session._next_live_number == before["next_id"]
    assert session._memory_matches == before["matches"]
    assert session._events == before["events"]
    assert session._live_identities == before["identities"]
    assert session.worker_alive

    current[0] = _snapshot(2, ("face-a.jpg",), chunk=1)
    session.request_version(2)
    recovered = _wait_for(session, 2)
    session.finish(2)

    assert recovered["analysis_state"] == "ready"
    assert recovered["analysis_version"] == 2
    assert recovered["analysis_warning"] is None
    assert recovered["live_identities"][0]["session_person_id"] == "live_0001"


def test_same_embedding_count_retries_dbscan_after_failure(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    calls = []

    def transient_cluster(state):
        calls.append(len(state["all_face_embeddings"]))
        if len(calls) == 1:
            raise RuntimeError("transient DBSCAN failure")
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=transient_cluster,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    failed = _wait_for(session, 1)
    current[0] = _snapshot(2, ("face-a.jpg",), chunk=1)
    session.request_version(2)
    recovered = _wait_for(session, 2)
    session.finish(2)

    assert failed["analysis_version"] == 0
    assert calls == [1, 1]
    assert recovered["analysis_state"] == "ready"
    assert recovered["analysis_warning"] is None
    assert recovered["analyzed_embedding_count"] == 1


def test_same_embedding_count_retries_global_memory_after_failure(tmp_path):
    database = tmp_path / "memory.db"
    database.touch()
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]

    class TransientMemory:
        def __init__(self):
            self.calls = 0

        def query_by_face(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("database locked")
            return []

        def close(self):
            return None

    memory = TransientMemory()
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
        memory_factory=lambda _path: memory,
    )
    session.start()
    session.request_version(1)
    failed = _wait_for(session, 1)
    current[0] = _snapshot(2, ("face-a.jpg",), chunk=1)
    session.request_version(2)
    recovered = _wait_for(session, 2)
    session.finish(2)

    assert failed["analysis_version"] == 0
    assert memory.calls == 2
    assert recovered["analysis_state"] == "ready"
    assert recovered["analysis_warning"] is None


def test_failed_new_evidence_retains_previous_valid_identities(tmp_path):
    current = [_snapshot(1, ("face-a.jpg",), chunk=0)]
    fail_new = [False]

    def cluster(state):
        if fail_new[0]:
            raise RuntimeError("new evidence failed")
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: current[0],
        cluster=cluster,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    first = _wait_for(session, 1)
    fail_new[0] = True
    current[0] = _snapshot(2, ("face-a.jpg", "face-b.jpg"), chunk=1)
    session.request_version(2)
    failed = _wait_for(session, 2)
    session.finish(2)

    assert failed["analysis_version"] == 1
    assert failed["live_identities"] == first["live_identities"]


def test_clustering_starts_after_accumulator_lock_is_released(tmp_path):
    chunk = LiveChunkResult(
        chunk_index=0,
        started_at=0.0,
        elapsed_seconds=1.0,
        stop_requested=False,
        frames_read=1,
        frames_processed=1,
        frames_skipped=0,
        frames_dropped=0,
        body_crops=[],
        face_crops=[],
        body_detection_count=0,
        face_detection_count=0,
        warnings=[],
    )

    def preprocess(**_kwargs):
        return PreprocessedLiveChunk(
            chunk_index=0,
            capture_summary={},
            quality_body_crops=[],
            quality_face_crops=[],
            face_embeddings=[{"crop_path": "face-a.jpg", "embedding": [1.0, 0.0]}],
            failed_face_embeddings=[],
            warnings=[],
            processing_elapsed_seconds=0.01,
        )

    accumulator = LivePreprocessingSession(
        base_state={}, preprocess=preprocess, rolling_analysis=True
    )
    accumulator.start()
    accumulator.submit_chunk(chunk)
    accumulator.finish()

    def cluster(state):
        assert accumulator._lock.acquire(blocking=False)
        accumulator._lock.release()
        return _cluster_all(state)

    session = LiveRollingAnalysisSession(
        snapshot_provider=accumulator.analysis_snapshot,
        cluster=cluster,
        associate=_association,
        database_path=tmp_path / "missing.db",
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)
    session.finish(1)

    assert status["analysis_state"] == "ready"


def test_inaccessible_memory_path_is_recoverable_warning(tmp_path, monkeypatch):
    database = tmp_path / "memory.db"
    original_stat = Path.stat

    def inaccessible(path, *args, **kwargs):
        if path == database:
            raise PermissionError("not allowed")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", inaccessible)
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
    )
    session.start()
    session.request_version(1)
    status = _wait_for(session, 1)
    session.finish(1)

    assert status["analysis_state"] == "warning"
    assert "OSError" in status["analysis_warning"]
    assert "not allowed" not in status["analysis_warning"]


def test_connection_cleanup_failure_wakes_shutdown_waiters(tmp_path):
    database = tmp_path / "memory.db"
    database.touch()

    class CleanupFailureMemory:
        def query_by_face(self, *_args, **_kwargs):
            return []

        def close(self):
            raise RuntimeError("close details")

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: _snapshot(1, ("face-a.jpg",)),
        cluster=_cluster_all,
        associate=_association,
        database_path=database,
        memory_factory=lambda _path: CleanupFailureMemory(),
    )
    session.start()
    session.request_version(1)
    _wait_for(session, 1)
    session.finish(1)
    status = session.public_snapshot()

    assert not session.worker_alive
    assert status["analysis_state"] == "shutdown_warning"
    assert "Global Memory cleanup" in status["analysis_warning"]
    assert "close details" not in status["analysis_warning"]


def test_intermediate_versions_are_conflated_to_newest_pending_version(tmp_path):
    current = {"version": 0}
    first_run_started = threading.Event()
    release_first_run = threading.Event()
    cluster_counts = []

    def snapshot_provider():
        version = current["version"]
        embeddings = tuple(
            FrozenRecord.from_mapping({
                "crop_path": f"face-{index}.jpg",
                "embedding": [1.0, 0.0],
                "sharpness": 100.0,
            })
            for index in range(version)
        )
        return FrozenAnalysisSnapshot(
            version=version,
            last_completed_preprocessing_chunk=version,
            person_name="",
            video_paths=(),
            identity_clustering_config=FrozenRecord.from_mapping({
                "eps": 0.4,
                "min_samples": 3,
                "min_cluster_face_count": 3,
            }),
            quality_body_crops=(),
            quality_face_crops=(),
            face_embeddings=embeddings,
            face_chunk_membership=tuple(
                (f"face-{index}.jpg", version) for index in range(version)
            ),
        )

    def cluster(state):
        cluster_counts.append(len(state["all_face_embeddings"]))
        if len(cluster_counts) == 1:
            first_run_started.set()
            assert release_first_run.wait(2.0)
        return {"identity_clusters": [], "unresolved_faces": []}

    session = LiveRollingAnalysisSession(
        snapshot_provider=snapshot_provider,
        database_path=tmp_path / "missing.db",
        cluster=cluster,
        associate=lambda _state: SimpleNamespace(
            associations=[],
            cluster_assignments={},
            unattached_bodies=[],
            frame_groups=[],
            rejected_pairs=[],
        ),
        debounce_seconds=0.5,
        minimum_new_versions=3,
    )
    session.start()
    for version in (1, 2, 3):
        current["version"] = version
        session.request_version(version)
    assert first_run_started.wait(1.0)
    for version in (4, 5, 6, 7):
        current["version"] = version
        session.request_version(version)
    release_first_run.set()
    session.finish(7)

    snapshot = session.public_snapshot()
    assert cluster_counts == [3, 7]
    assert snapshot["analysis_runs_started"] == 2
    assert snapshot["analysis_versions_conflated"] >= 5
    assert snapshot["analysis_version"] == 7
