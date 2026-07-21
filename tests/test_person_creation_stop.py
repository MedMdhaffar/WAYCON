from __future__ import annotations

import json
import threading

import pytest

from forensics.person_creation import service


@pytest.fixture(autouse=True)
def isolated_jobs(monkeypatch):
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()
    monkeypatch.setattr(service, "_start_pipeline_thread", lambda *_args: None)
    yield
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()


@pytest.fixture
def client():
    return service.app.test_client()


def _start_camera(client, uri: str = "rtsp://supervisor:secret@camera.local/live") -> str:
    response = client.post("/api/person/start", json={
        "name": "Camera Test",
        "input_type": "camera_uri",
        "camera_uri": uri,
        "duration_seconds": 10,
    })
    assert response.status_code == 200
    return response.get_json()["job_id"]


def test_starting_camera_job_creates_unique_stop_event(client):
    first_id = _start_camera(client)
    second_id = _start_camera(client, "rtsp://camera.local/second")

    with service._jobs_lock:
        first = service._job_runtimes[first_id].stop_event
        second = service._job_runtimes[second_id].stop_event

    assert first is not second
    assert not first.is_set()
    assert not second.is_set()


def test_stop_sets_only_targeted_job_event(client):
    first_id = _start_camera(client)
    second_id = _start_camera(client, "rtsp://camera.local/second")
    with service._jobs_lock:
        first = service._job_runtimes[first_id].stop_event
        second = service._job_runtimes[second_id].stop_event

    response = client.post(f"/api/person/stop/{first_id}")

    assert response.get_json() == {"job_id": first_id, "status": "stop_requested"}
    assert first.is_set()
    assert not second.is_set()


def test_stop_unknown_job_returns_not_found(client):
    response = client.post("/api/person/stop/missing-job")

    assert response.status_code == 404
    assert response.get_json() == {"error": "job not found"}


def test_repeated_stop_requests_are_idempotent(client):
    job_id = _start_camera(client)

    first = client.post(f"/api/person/stop/{job_id}")
    second = client.post(f"/api/person/stop/{job_id}")

    assert first.status_code == second.status_code == 200
    assert first.get_json() == second.get_json() == {
        "job_id": job_id,
        "status": "stop_requested",
    }


def test_completed_job_returns_already_finished(client):
    job_id = _start_camera(client)
    with service._jobs_lock:
        service._jobs[job_id].status = "done"
        service._job_runtimes.pop(job_id)

    response = client.post(f"/api/person/stop/{job_id}")

    assert response.status_code == 200
    assert response.get_json() == {"job_id": job_id, "status": "already_finished"}


def test_status_exposes_stop_requested_without_runtime_objects(client):
    job_id = _start_camera(client)
    client.post(f"/api/person/stop/{job_id}")

    payload = client.get(f"/api/person/status/{job_id}").get_json()
    serialized = json.dumps(payload)

    assert payload["status"] == "stop_requested"
    assert "_stop_event" not in serialized
    assert "Event" not in serialized


def test_status_exposes_safe_continuous_capture_progress(client):
    job_id = _start_camera(client)
    with service._jobs_lock:
        service._jobs[job_id].snapshot.update({
            "chunk_index": 4,
            "completed_chunks": 5,
            "last_chunk": {"frames_read": 320, "body_detections": 2},
            "session_totals": {"frames_read": 1580, "body_detections": 8},
            "stop_requested": False,
            "continuous": True,
            "duration_seconds_per_chunk": 10,
        })

    snapshot = client.get(f"/api/person/status/{job_id}").get_json()["snapshot"]

    assert snapshot["chunk_index"] == 4
    assert snapshot["completed_chunks"] == 5
    assert snapshot["last_chunk"]["body_detections"] == 2
    assert snapshot["session_totals"]["frames_read"] == 1580
    assert snapshot["continuous"] is True
    assert snapshot["duration_seconds_per_chunk"] == 10


def test_reconnect_status_is_public_and_preserves_rolling_identities(client):
    job_id = _start_camera(client)
    rolling = _rolling_publication(4, 3, 3, 2)
    rolling["live_identities"] = [{
        "session_person_id": "live_0001",
        "status": "provisional",
        "face_count": 5,
        "associated_body_count": 3,
        "memory_match": None,
    }]
    reconnect_stats = {
        "stream_state": "reconnecting",
        "stream_reconnect_count": 2,
        "stream_warning": "Temporary camera interruption; reconnecting.",
        "last_frame_age_seconds": 3.2,
    }
    with service._jobs_lock:
        job = service._jobs[job_id]
        service._merge_job_snapshot(job, {"rolling_analysis": rolling})
        service._merge_job_snapshot(job, {"stream_stats": reconnect_stats})

    snapshot = client.get(f"/api/person/status/{job_id}").get_json()["snapshot"]
    serialized = json.dumps(snapshot)

    assert snapshot["stream_stats"] == reconnect_stats
    assert snapshot["rolling_analysis"]["live_identities"] == rolling["live_identities"]
    assert "supervisor" not in serialized
    assert "secret" not in serialized


def test_status_exposes_json_safe_embedding_free_preprocessing_progress(client):
    job_id = _start_camera(client)
    preview = {
        "enabled": True,
        "queue_capacity": 2,
        "capture_completed_chunks": 3,
        "preprocessing_completed_chunks": 2,
        "preprocessing_pending_chunks": 1,
        "preprocessing_active_chunk": 2,
        "quality_body_crops": 8,
        "quality_face_crops": 4,
        "embedded_faces": 4,
        "failed_face_embeddings": 0,
    }
    with service._jobs_lock:
        service._jobs[job_id].snapshot.update({
            "live_preprocessing": preview,
            "stream_stats": {
                "frames_read": 100,
                "live_preprocessing": preview,
            },
        })

    snapshot = client.get(f"/api/person/status/{job_id}").get_json()["snapshot"]
    serialized = json.dumps(snapshot)

    assert snapshot["stream_stats"]["frames_read"] == 100
    assert snapshot["stream_stats"]["live_preprocessing"] == preview
    assert snapshot["live_preprocessing"] == preview
    assert "face_embeddings" not in snapshot["live_preprocessing"]
    assert "[0.25, 0.75]" not in serialized
    assert snapshot["source_uri_masked"] == "rtsp://****@camera.local/live"
    assert "supervisor" not in serialized
    assert "secret" not in serialized


def test_status_exposes_only_compact_rolling_analysis(client):
    job_id = _start_camera(client)
    rolling = {
        "enabled": True,
        "requested_version": 4,
        "analysis_version": 3,
        "analysis_state": "ready",
        "analysis_in_progress": False,
        "analyzed_embedding_count": 7,
        "last_completed_preprocessing_chunk": 2,
        "analysis_warning": None,
        "live_identities": [{
            "session_person_id": "live_0001",
            "cluster_label": 0,
            "status": "provisional",
            "face_count": 7,
            "associated_body_count": 3,
            "representative_face_path": "face.jpg",
            "memory_match": None,
        }],
        "live_recognition_events": [],
    }
    with service._jobs_lock:
        service._jobs[job_id].snapshot.update({
            "rolling_analysis": rolling,
            "all_face_embeddings": [[0.5, 0.5]],
            "analysis_thread": "must-not-escape",
            "database_path": "/private/memory.db",
        })

    snapshot = client.get(f"/api/person/status/{job_id}").get_json()["snapshot"]
    serialized = json.dumps(snapshot)

    expected = json.loads(json.dumps(rolling))
    expected["live_identities"][0]["representative_face_path"] = None
    assert snapshot["rolling_analysis"] == expected
    assert "all_face_embeddings" not in serialized
    assert "analysis_thread" not in serialized
    assert "database_path" not in serialized


def _rolling_publication(sequence: int, requested: int, analyzed: int, chunk: int):
    return {
        "enabled": True,
        "publication_sequence": sequence,
        "requested_version": requested,
        "analysis_version": analyzed,
        "analysis_state": "ready",
        "analysis_in_progress": False,
        "analyzed_embedding_count": analyzed,
        "last_completed_preprocessing_chunk": chunk,
        "analysis_warning": None,
        "live_identities": [],
        "live_recognition_events": [],
    }


def test_older_rolling_publication_cannot_overwrite_newer_status():
    job = service.JobState("publication-order")
    with service._jobs_lock:
        service._merge_job_snapshot(job, {
            "rolling_analysis": _rolling_publication(2, 3, 2, 1),
        })
        service._merge_job_snapshot(job, {
            "rolling_analysis": _rolling_publication(1, 1, 1, 0),
        })

    rolling = job.snapshot["rolling_analysis"]
    assert rolling["publication_sequence"] == 2
    assert rolling["requested_version"] == 3
    assert rolling["analysis_version"] == 2


def test_concurrent_callback_completion_cannot_regress_status():
    job = service.JobState("concurrent-publication")
    old_started = threading.Event()
    allow_old = threading.Event()

    def apply_old():
        old_started.set()
        assert allow_old.wait(2.0)
        with service._jobs_lock:
            service._merge_job_snapshot(job, {
                "rolling_analysis": _rolling_publication(1, 1, 1, 0),
            })

    thread = threading.Thread(target=apply_old)
    thread.start()
    assert old_started.wait(1.0)
    with service._jobs_lock:
        service._merge_job_snapshot(job, {
            "rolling_analysis": _rolling_publication(2, 4, 3, 2),
        })
    allow_old.set()
    thread.join(2.0)

    assert not thread.is_alive()
    rolling = job.snapshot["rolling_analysis"]
    assert rolling["publication_sequence"] == 2
    assert rolling["requested_version"] == 4
    assert rolling["analysis_version"] == 3
    assert rolling["last_completed_preprocessing_chunk"] == 2


def test_video_job_remains_finite_and_has_no_stop_runtime(client, monkeypatch):
    monkeypatch.setattr(service, "build_initial_state", lambda _body: {
        "person_name": "Video Test",
        "input_type": "video_file",
        "source_type": "video_file",
        "video_paths": ["video.mp4"],
        "output_dir": "output",
        "process_every_n": 15,
        "identity_clustering_config": {},
        "reid_config": {},
        "body_crops": [],
        "face_crops": [],
    })

    start_response = client.post("/api/person/start", json={"name": "Video Test"})
    job_id = start_response.get_json()["job_id"]
    stop_response = client.post(f"/api/person/stop/{job_id}")

    assert start_response.status_code == 200
    with service._jobs_lock:
        assert job_id not in service._job_runtimes
    assert stop_response.status_code == 409
    assert stop_response.get_json()["status"] == "not_live_camera"


def test_runtime_cleanup_keeps_public_job_and_passes_event_to_graph(monkeypatch):
    job_id = "cleanup-job"
    runtime = service.JobRuntime()
    captured = {}

    class FakeGraph:
        def stream(self, state, stream_mode):
            captured.update(state)
            assert stream_mode == "updates"
            return iter(())

    with service._jobs_lock:
        service._jobs[job_id] = service.JobState(job_id, input_type="camera_uri")
        service._job_runtimes[job_id] = runtime
    monkeypatch.setattr(service, "_get_graph", lambda: FakeGraph())

    service._run_pipeline(job_id, {"input_type": "camera_uri"})

    assert captured["_stop_event"] is runtime.stop_event
    with service._jobs_lock:
        assert job_id in service._jobs
        assert service._jobs[job_id].status == "done"
        assert job_id not in service._job_runtimes


def test_failed_pipeline_also_cleans_runtime(monkeypatch):
    job_id = "failed-job"

    class FailingGraph:
        def stream(self, _state, stream_mode):
            assert stream_mode == "updates"
            raise RuntimeError("mock pipeline failure")

    with service._jobs_lock:
        service._jobs[job_id] = service.JobState(job_id, input_type="camera_uri")
        service._job_runtimes[job_id] = service.JobRuntime()
    monkeypatch.setattr(service, "_get_graph", lambda: FailingGraph())

    service._run_pipeline(job_id, {"input_type": "camera_uri"})

    with service._jobs_lock:
        assert service._jobs[job_id].status == "error"
        assert job_id not in service._job_runtimes


def test_stop_response_and_log_do_not_expose_camera_uri(client, capsys):
    camera_uri = "rtsp://supervisor:very-secret@camera.local/private/live"
    job_id = _start_camera(client, camera_uri)
    capsys.readouterr()

    response = client.post(f"/api/person/stop/{job_id}")
    output = capsys.readouterr().out
    serialized = json.dumps(response.get_json())

    assert camera_uri not in output
    assert "very-secret" not in output
    assert camera_uri not in serialized
    assert "very-secret" not in serialized
    assert job_id in output
