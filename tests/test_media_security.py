from __future__ import annotations

from pathlib import Path

import pytest

from forensics.person_creation import service


@pytest.fixture(autouse=True)
def isolated_service(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    root.mkdir()
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setattr(service, "_start_pipeline_thread", lambda *_args: None)
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()
    yield root
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()


@pytest.fixture
def client():
    return service.app.test_client()


def test_valid_encoded_and_nested_image_request(client, isolated_service):
    image = isolated_service / "person_001" / "face_crops" / "face one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")

    response = client.get("/api/images?path=person_001%2Fface_crops%2Fface%20one.jpg")

    assert response.status_code == 200
    assert response.data == b"jpeg"

    absolute = client.get("/api/images", query_string={"path": str(image)})
    assert absolute.status_code == 403


@pytest.mark.parametrize("path", ["/etc/passwd", "../../outside.jpg", "/mnt/c/Users/user/secret.txt"])
def test_image_endpoint_rejects_outside_paths(client, path):
    response = client.get("/api/images", query_string={"path": path})

    assert response.status_code == 403
    assert path not in response.get_json()["error"]


def test_image_endpoint_rejects_symlink_escape(client, isolated_service, tmp_path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (isolated_service / "escape.jpg").symlink_to(outside)

    response = client.get("/api/images", query_string={"path": "escape.jpg"})

    assert response.status_code == 403
    assert response.data != b"secret"


def test_missing_and_unsupported_media_are_safe(client, isolated_service):
    text_file = isolated_service / "notes.txt"
    text_file.write_text("private")

    missing = client.get("/api/images", query_string={"path": "missing.jpg"})
    unsupported = client.get("/api/images", query_string={"path": "notes.txt"})

    assert missing.status_code == 404
    assert unsupported.status_code == 400
    assert unsupported.data != b"private"


def _install_job(root: Path, *, target: Path) -> str:
    job_id = "job-1"
    with service._jobs_lock:
        service._jobs[job_id] = service.JobState(
            job_id,
            input_type="camera_uri",
            output_dir=str(root),
            snapshot={"quality_body_crops": [{"path": str(target)}]},
        )
    return job_id


def test_valid_job_crop_deletion(client, isolated_service):
    output = isolated_service / "session"
    crop = output / "_staging" / "body_crops" / "body.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"crop")
    job_id = _install_job(output, target=crop)

    response = client.delete(
        f"/api/person/crop/{job_id}",
        json={"path": "session/_staging/body_crops/body.jpg", "crop_type": "body"},
    )

    assert response.status_code == 200
    assert not crop.exists()


def test_crop_delete_cannot_remove_outside_file(client, isolated_service, tmp_path):
    output = isolated_service / "session"
    output.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"keep")
    job_id = _install_job(output, target=outside)

    response = client.delete(
        f"/api/person/crop/{job_id}",
        json={"path": str(outside), "crop_type": "body"},
    )

    assert response.status_code == 403
    assert outside.read_bytes() == b"keep"


def test_start_rejects_output_directory_outside_media_root(client, tmp_path):
    response = client.post("/api/person/start", json={
        "name": "Unsafe",
        "input_type": "camera_uri",
        "camera_uri": "rtsp://camera.local/live",
        "output_dir": str(tmp_path / "outside"),
    })

    assert response.status_code == 400
    assert response.get_json()["error"] == "output_dir must be inside the configured media root"


def test_status_exposes_relative_media_ids_not_internal_paths(client, isolated_service):
    output = isolated_service / "session"
    crop = output / "_staging" / "face_crops" / "face.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"face")
    with service._jobs_lock:
        service._jobs["job-status"] = service.JobState(
            "job-status",
            input_type="camera_uri",
            output_dir=str(output),
            snapshot={"quality_face_crops": [{"path": str(crop)}]},
        )

    payload = client.get("/api/person/status/job-status").get_json()

    public_path = payload["snapshot"]["quality_face_crops"][0]["path"]
    assert public_path == "session/_staging/face_crops/face.jpg"
    assert str(isolated_service) not in str(payload)
