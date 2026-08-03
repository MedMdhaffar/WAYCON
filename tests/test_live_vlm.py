from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from forensics.global_memory import GlobalMemory
from forensics.person_creation import service
from forensics.person_creation.live_vlm import LiveIdentityVLMCoordinator


def _write(root: Path, relative: str, contents: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return relative


def _snapshot(live_id: str, version: int, selected: str, *, sequence=None) -> dict:
    sequence = version if sequence is None else sequence
    return {
        "enabled": True,
        "publication_sequence": sequence,
        "requested_version": version,
        "analysis_version": version,
        "last_completed_preprocessing_chunk": version,
        "live_identities": [{
            "live_identity_id": live_id,
            "session_person_id": live_id,
            "version": version,
            "cluster_label": 0,
            "clustering_state": "resolved",
            "canonical_person_id": "person_001",
            "representative_face_path": "_staging/face_crops/face.jpg",
            "best_face_path": "_staging/face_crops/face.jpg",
            "best_body_path": selected,
        }],
    }


def _receipt(live_id: str, *body_paths: str) -> dict:
    return {
        "job_id": "job-vlm",
        "live_identity_id": live_id,
        "canonical_person_id": "person_001",
        "canonical_face_paths": [],
        "canonical_body_paths": list(body_paths),
    }


def _success(top: str = "black jacket") -> dict:
    return {
        "per_cluster_clothing": {
            0: {
                "status": "ok",
                "attempts": 1,
                "top": top,
                "bottom": "blue jeans",
                "shoes": "white shoes",
                "full": f"{top}, blue jeans, white shoes",
                "failure_reason": None,
            }
        },
        "clothing_diagnostics": [{
            "cluster_id": 0,
            "attempts": 1,
            "status": "ok",
            "failure_reason": None,
        }],
    }


def _failure(reason: str) -> dict:
    return {
        "per_cluster_clothing": {
            0: {
                "status": "failed",
                "attempts": 2,
                "failure_reason": reason,
            }
        },
        "clothing_diagnostics": [{
            "cluster_id": 0,
            "attempts": 2,
            "status": "failed",
            "failure_reason": reason,
        }],
    }


def _identity(snapshot: dict, live_id: str = "live_0001") -> dict:
    return next(
        item for item in snapshot["live_identities"]
        if item["live_identity_id"] == live_id
    )


def _wait_for(coordinator, predicate, *, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = coordinator.public_snapshot()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.005)
    raise AssertionError("timed out waiting for live VLM state")


def test_vlm_defers_while_core_inference_has_pending_work(tmp_path):
    root = tmp_path / "person_db"
    core_busy = threading.Event()
    core_busy.set()
    described = threading.Event()
    source = _write(root, "_staging/live/body_crops/body.jpg", b"body")
    canonical = _write(root, "person_001/body_crops/body.jpg", b"body")

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm",
        media_root=root,
        queue_capacity=2,
        describe=lambda _job: (described.set() or _success()),
        persist=lambda *_args: None,
        core_work_pending=core_busy.is_set,
    )
    try:
        coordinator.observe(
            _snapshot("live_0001", 1, source),
            [_receipt("live_0001", canonical)],
        )
        assert not described.wait(0.1)
        deferred = coordinator.public_snapshot()
        assert deferred["vlm_jobs_deferred_for_core_work"] > 0
        assert deferred["vlm_queue_depth"] == 1
        assert deferred["vlm_queue_peak"] == 1

        core_busy.clear()
        assert described.wait(1.0)
        completed = _wait_for(
            coordinator,
            lambda item: item.get("vlm_completed") == 1,
        )
        assert completed["vlm_active_jobs"] == 0
    finally:
        core_busy.clear()
        coordinator.close(1.0)


def test_continuous_core_activity_has_bounded_vlm_fairness_and_metrics(tmp_path):
    root = tmp_path / "person_db"
    source = _write(root, "_staging/live/body_crops/body.jpg", b"body")
    canonical = _write(root, "person_001/body_crops/body.jpg", b"body")
    started = threading.Event()
    release = threading.Event()

    def describe(_job):
        started.set()
        assert release.wait(1.0)
        return _success("green coat")

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm",
        media_root=root,
        describe=describe,
        persist=lambda *_args: None,
        core_work_pending=lambda: True,
        core_queue_depth=lambda: 1,
        maximum_core_deferral_seconds=0.05,
    )
    try:
        pending = coordinator.observe(
            _snapshot("live_0001", 1, source),
            [_receipt("live_0001", canonical)],
        )
        assert _identity(pending)["vlm_state"] == "pending"
        assert started.wait(0.5)
        running = coordinator.public_snapshot()
        assert _identity(running)["vlm_state"] == "running"
        assert running["vlm_jobs_deferred_for_core_work"] > 0
        assert running["vlm_jobs_started"] == 1

        release.set()
        completed = _wait_for(
            coordinator,
            lambda snap: _identity(snap)["vlm_state"] == "completed",
        )
        identity = _identity(completed)
        assert identity["live_identity_id"] == "live_0001"
        assert identity["clothing_description"].startswith("green coat")
        assert completed["per_cluster_clothing"][0]["top"] == "green coat"
        assert completed["vlm_observations_received"] == 1
        assert completed["vlm_jobs_eligible"] == 1
        assert completed["vlm_jobs_submitted"] == 1
        assert completed["vlm_jobs_completed"] == 1
        assert completed["vlm_results_merged"] == 1
        assert completed["vlm_max_deferral_ms"] >= 40
        assert completed["vlm_last_error"] is None
    finally:
        release.set()
        coordinator.close(1.0)


def test_vlm_never_blocks_observe_and_queue_stays_bounded(tmp_path):
    root = tmp_path / "person_db"
    release = threading.Event()
    started = threading.Event()

    def describe(_job):
        started.set()
        release.wait(2)
        return _success()

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm",
        media_root=root,
        queue_capacity=2,
        describe=describe,
        persist=lambda *_args: None,
    )
    try:
        first_source = _write(root, "_staging/live/body_crops/first.jpg", b"first")
        first_body = _write(root, "person_001/body_crops/first.jpg", b"first")
        coordinator.observe(
            _snapshot("live_0001", 1, first_source),
            [_receipt("live_0001", first_body)],
        )
        assert started.wait(1)

        started_at = time.monotonic()
        identities = _snapshot("live_0001", 1, first_source)["live_identities"]
        receipts = [_receipt("live_0001", first_body)]
        for number in range(2, 7):
            live_id = f"live_{number:04d}"
            source = _write(
                root, f"_staging/live/body_crops/{number}.jpg", str(number).encode()
            )
            body = _write(
                root, f"person_001/body_crops/{number}.jpg", str(number).encode()
            )
            identities.extend(_snapshot(live_id, 1, source)["live_identities"])
            receipts.append(_receipt(live_id, body))
            coordinator.observe({
                "enabled": True,
                "publication_sequence": number,
                "requested_version": number,
                "analysis_version": number,
                "last_completed_preprocessing_chunk": number,
                "live_identities": identities,
            }, receipts)
            assert coordinator.public_snapshot()["vlm_queue_depth"] <= 2
        assert time.monotonic() - started_at < 0.25
        snapshot = coordinator.public_snapshot()
        assert snapshot["vlm_queue_capacity"] == 2
        assert snapshot["vlm_dropped"] >= 3
        assert snapshot["vlm_failed"] == 0
    finally:
        release.set()
        coordinator.close(1)


def test_latest_crop_replaces_queued_work_for_same_identity(tmp_path):
    root = tmp_path / "person_db"
    release = threading.Event()
    active = threading.Event()
    calls = []

    def describe(job):
        calls.append((job.live_identity_id, job.live_identity_version, job.body_crop_path))
        if job.live_identity_id == "live_0001":
            active.set()
            release.wait(2)
        return _success()

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=describe,
        persist=lambda *_args: None,
    )
    try:
        blocker_source = _write(root, "_staging/body/block.jpg", b"block")
        blocker = _write(root, "person_001/body_crops/block.jpg", b"block")
        coordinator.observe(
            _snapshot("live_0001", 1, blocker_source),
            [_receipt("live_0001", blocker)],
        )
        assert active.wait(1)

        old_source = _write(root, "_staging/body/old.jpg", b"old")
        old_body = _write(root, "person_001/body_crops/old.jpg", b"old")
        coordinator.observe(
            _snapshot("live_0002", 1, old_source),
            [_receipt("live_0002", old_body)],
        )
        new_source = _write(root, "_staging/body/new.jpg", b"new")
        new_body = _write(root, "person_001/body_crops/new.jpg", b"new")
        coordinator.observe(
            _snapshot("live_0002", 2, new_source),
            [_receipt("live_0002", old_body, new_body)],
        )
        assert coordinator.public_snapshot()["vlm_queue_depth"] == 1
        release.set()
        _wait_for(
            coordinator,
            lambda snap: _identity(snap, "live_0002")["vlm_status"] == "completed",
        )
        assert calls == [
            ("live_0001", 1, blocker),
            ("live_0002", 2, new_body),
        ]
    finally:
        release.set()
        coordinator.close(1)


def test_duplicate_submissions_are_deduplicated(tmp_path):
    root = tmp_path / "person_db"
    release = threading.Event()
    started = threading.Event()
    calls = []

    def describe(job):
        calls.append(job)
        started.set()
        release.wait(2)
        return _success()

    source = _write(root, "_staging/body/body.jpg", b"body")
    body = _write(root, "person_001/body_crops/body.jpg", b"body")
    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=describe,
        persist=lambda *_args: None,
    )
    try:
        for sequence in range(10):
            coordinator.observe(
                _snapshot("live_0001", 1, source, sequence=sequence),
                [_receipt("live_0001", body)],
            )
        assert started.wait(1)
        assert coordinator.public_snapshot()["vlm_queue_depth"] == 0
        release.set()
        _wait_for(
            coordinator,
            lambda snap: _identity(snap)["vlm_status"] == "completed",
        )
        assert len(calls) == 1
    finally:
        release.set()
        coordinator.close(1)


def test_stale_identity_version_result_is_discarded(tmp_path):
    root = tmp_path / "person_db"
    release = threading.Event()
    started = threading.Event()
    persisted = []

    def describe(job):
        if job.live_identity_version == 1:
            started.set()
            release.wait(2)
        return _success(f"version-{job.live_identity_version}")

    old_source = _write(root, "_staging/body/old.jpg", b"old")
    old_body = _write(root, "person_001/body_crops/old.jpg", b"old")
    new_source = _write(root, "_staging/body/new.jpg", b"new")
    new_body = _write(root, "person_001/body_crops/new.jpg", b"new")
    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=describe,
        persist=lambda job, _clothing: persisted.append(job.live_identity_version),
    )
    try:
        coordinator.observe(
            _snapshot("live_0001", 1, old_source),
            [_receipt("live_0001", old_body)],
        )
        assert started.wait(1)
        coordinator.observe(
            _snapshot("live_0001", 2, new_source),
            [_receipt("live_0001", old_body, new_body)],
        )
        release.set()
        snapshot = _wait_for(
            coordinator,
            lambda snap: (
                _identity(snap)["vlm_status"] == "completed"
                and _identity(snap)["vlm_version"] == 2
            ),
        )
        assert persisted == [2]
        assert _identity(snapshot)["clothing_description"].startswith("version-2")
    finally:
        release.set()
        coordinator.close(1)


def test_content_mapping_survives_relocation_and_ignores_duplicate_basename(tmp_path):
    root = tmp_path / "person_db"
    selected = _write(root, "_staging/body/same.jpg", b"selected bytes")
    wrong = _write(root, "person_001/body_crops/same.jpg", b"different bytes")
    relocated = _write(
        root, "person_001/body_crops/same_1.jpg", b"selected bytes"
    )
    jobs = []
    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root,
        describe=lambda job: jobs.append(job) or _success(),
        persist=lambda *_args: None,
    )
    try:
        snapshot = coordinator.observe(
            _snapshot("live_0001", 1, selected),
            [_receipt("live_0001", wrong, relocated)],
        )
        assert _identity(snapshot)["selected_body_crop"] == relocated
        _wait_for(coordinator, lambda snap: _identity(snap)["vlm_status"] == "completed")
        assert jobs[0].body_crop_path == relocated
    finally:
        coordinator.close(1)


def test_only_existing_canonical_body_media_can_enter_queue(tmp_path):
    root = tmp_path / "person_db"
    source = _write(root, "_staging/body/body.jpg", b"body")
    staging = _write(root, "_staging/body/queued.jpg", b"body")
    clustered = _write(root, "person_001/cluster_2/body.jpg", b"body")
    session = _write(root, "person_001/session/body.jpg", b"body")
    canonical_absolute = str(
        (root / _write(root, "person_001/body_crops/absolute.jpg", b"body")).resolve()
    )
    calls = []
    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root,
        describe=lambda job: calls.append(job) or _success(),
        persist=lambda *_args: None,
    )
    try:
        snapshot = coordinator.observe(
            _snapshot("live_0001", 1, source),
            [_receipt(
                "live_0001",
                staging,
                clustered,
                session,
                "person_001/body_crops/missing.jpg",
                canonical_absolute,
                "../outside.jpg",
            )],
        )
        identity = _identity(snapshot)
        assert identity["vlm_status"] == "not_started"
        assert identity["selected_body_crop"] is None
        assert snapshot["vlm_queue_depth"] == 0
        time.sleep(0.02)
        assert calls == []
    finally:
        coordinator.close(1)


def test_success_updates_canonical_appearance(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    database = tmp_path / "memory.db"
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    source = _write(root, "_staging/body/body.jpg", b"body")
    body = _write(root, "person_001/body_crops/body.jpg", b"body")
    with GlobalMemory(str(database), media_root=root) as memory:
        memory._conn.execute(
            """INSERT INTO persons(
                person_id, name, embedding, embedding_count,
                enrolled_at, updated_at, cameras
            ) VALUES ('person_001', 'Person 001', ?, 2,
                      '2026-07-22', '2026-07-22', '[]')""",
            (b"embedding",),
        )

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=lambda _job: _success(),
    )
    try:
        coordinator.observe(
            _snapshot("live_0001", 1, source),
            [_receipt("live_0001", body)],
        )
        snapshot = _wait_for(
            coordinator, lambda snap: _identity(snap)["vlm_status"] == "completed"
        )
        assert _identity(snapshot)["selected_body_crop"] == body
    finally:
        coordinator.close(1)

    with GlobalMemory(str(database), media_root=root) as memory:
        appearance = memory._conn.execute(
            "SELECT * FROM appearances WHERE person_id='person_001'"
        ).fetchone()
        assert appearance["top"] == "black jacket"
        assert appearance["full_description"].startswith("black jacket")
        assert json.loads(appearance["best_body_crops"]) == [body]
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM person_gallery WHERE person_id='person_001'"
        ).fetchone()[0] == 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM recognition_log"
        ).fetchone()[0] == 0


def test_failure_timeout_and_existing_retry_diagnostics_are_published(tmp_path):
    root = tmp_path / "person_db"
    source = _write(root, "_staging/body/body.jpg", b"body")
    body = _write(root, "person_001/body_crops/body.jpg", b"body")
    for reason, expected in (("invalid_output", "failed"), ("timeout", "timed_out")):
        coordinator = LiveIdentityVLMCoordinator(
            job_id=f"job-{reason}", media_root=root,
            describe=lambda _job, value=reason: _failure(value),
            persist=lambda *_args: None,
        )
        try:
            coordinator.observe(
                _snapshot("live_0001", 1, source),
                [_receipt("live_0001", body)],
            )
            snapshot = _wait_for(
                coordinator,
                lambda snap, value=expected: _identity(snap)["vlm_status"] == value,
            )
            identity = _identity(snapshot)
            assert identity["vlm_error"] == reason
            assert identity["clothing_diagnostics"][0]["attempts"] == 2
        finally:
            coordinator.close(1)


def test_stop_drain_has_hard_bound_and_marks_unfinished_timeout(tmp_path):
    root = tmp_path / "person_db"
    source = _write(root, "_staging/body/body.jpg", b"body")
    body = _write(root, "person_001/body_crops/body.jpg", b"body")
    release = threading.Event()
    started = threading.Event()
    persisted = []

    def describe(_job):
        started.set()
        release.wait(2)
        return _success()

    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=describe,
        persist=lambda *_args: persisted.append(True),
    )
    coordinator.observe(
        _snapshot("live_0001", 1, source), [_receipt("live_0001", body)]
    )
    assert started.wait(1)
    started_at = time.monotonic()
    snapshot = coordinator.close(0.05)
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.3
    assert _identity(snapshot)["vlm_status"] == "timed_out"
    assert snapshot["vlm_timed_out"] == 1
    release.set()
    time.sleep(0.05)
    assert persisted == []


def test_completed_version_is_not_persisted_twice(tmp_path):
    root = tmp_path / "person_db"
    source = _write(root, "_staging/body/body.jpg", b"body")
    body = _write(root, "person_001/body_crops/body.jpg", b"body")
    writes = []
    coordinator = LiveIdentityVLMCoordinator(
        job_id="job-vlm", media_root=root, describe=lambda _job: _success(),
        persist=lambda job, _clothing: writes.append(
            (job.live_identity_id, job.live_identity_version)
        ),
    )
    try:
        snapshot = _snapshot("live_0001", 1, source)
        receipt = _receipt("live_0001", body)
        coordinator.observe(snapshot, [receipt])
        _wait_for(coordinator, lambda snap: _identity(snap)["vlm_status"] == "completed")
        for sequence in range(2, 12):
            coordinator.observe(
                _snapshot("live_0001", 1, source, sequence=sequence), [receipt]
            )
        time.sleep(0.05)
        assert writes == [("live_0001", 1)]
    finally:
        coordinator.close(1)


def test_status_api_exposes_vlm_diagnostics_without_path_or_error_leaks(
    tmp_path, monkeypatch
):
    root = tmp_path / "person_db"
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    canonical = _write(root, "person_001/body_crops/body.jpg", b"body")
    staging = _write(root, "_staging/body/secret.jpg", b"secret")
    absolute = str((root / canonical).resolve())
    with service._jobs_lock:
        service._jobs["job-vlm-status"] = service.JobState(
            "job-vlm-status",
            input_type="camera_uri",
            snapshot={
                "rolling_analysis": {
                    "live_identities": [{
                        "live_identity_id": "live_0001",
                        "canonical_person_id": "person_001",
                        "best_body_path": staging,
                        "selected_body_crop": absolute,
                        "vlm_status": "failed",
                        "vlm_version": 1,
                        "vlm_error": (
                            "token=secret rtsp://user:password@camera.local/live "
                            + absolute
                        ),
                    }],
                    "vlm_queue_depth": 1,
                    "vlm_queue_capacity": 2,
                    "vlm_active_identity": "live_0001",
                    "vlm_completed": 0,
                    "vlm_failed": 1,
                    "vlm_dropped": 0,
                    "vlm_timed_out": 0,
                }
            },
        )
    try:
        payload = service.app.test_client().get(
            "/api/person/status/job-vlm-status"
        ).get_json()
    finally:
        with service._jobs_lock:
            service._jobs.pop("job-vlm-status", None)

    rolling = payload["snapshot"]["rolling_analysis"]
    identity = rolling["live_identities"][0]
    assert identity["selected_body_crop"] == canonical
    assert identity["best_body_path"] is None
    assert identity["vlm_error"] == "inference_error"
    assert identity["vlm_version"] == 1
    assert rolling["vlm_queue_capacity"] == 2
    serialized = json.dumps(payload)
    assert "password" not in serialized
    assert "camera.local" not in serialized
    assert str(root) not in serialized
