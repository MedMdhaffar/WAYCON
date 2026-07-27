from __future__ import annotations

import json
import re
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


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../../outside.jpg",
        "/mnt/c/Users/user/secret.txt",
        r"C:\Users\user\secret.jpg",
        "file:///etc/passwd.jpg",
        "https://camera.local/private.jpg",
        "rtsp://camera.local/private.jpg",
    ],
)
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


def test_status_sanitizes_selected_body_crop_diagnostics(client, isolated_service):
    # A valid canonical relative crop that actually exists under the media root.
    relative_id = "person_001/body_crops/body.jpg"
    real_crop = isolated_service / "person_001" / "body_crops" / "body.jpg"
    real_crop.parent.mkdir(parents=True)
    real_crop.write_bytes(b"body")

    # The exact class of value the audit demonstrated leaking through the status
    # API: an absolute Windows path (drive letter + backslashes) carrying an
    # internal cluster_N staging segment. It does not resolve under the
    # temporary media root, so it must be dropped rather than exposed.
    absolute_cluster_path = (
        r"C:\Users\aziza\Documents\GitHub\WAYCON\forensics\person_db"
        r"\john\cluster_0\body_crops\body_000123.jpg"
    )

    with service._jobs_lock:
        service._jobs["job-clothing"] = service.JobState(
            "job-clothing",
            input_type="camera_uri",
            output_dir=str(isolated_service / "session"),
            snapshot={
                "clothing_diagnostics": [
                    {
                        "cluster_id": 0,
                        "selected_body_crop": absolute_cluster_path,
                        "status": "failed",
                    },
                    {
                        "cluster_id": 1,
                        "selected_body_crop": relative_id,
                        "status": "ok",
                    },
                ],
            },
        )

    payload = client.get("/api/person/status/job-clothing").get_json()
    diagnostics = payload["snapshot"]["clothing_diagnostics"]

    # The absolute cluster path is never returned unchanged; it is sanitized away.
    assert diagnostics[0]["selected_body_crop"] != absolute_cluster_path
    assert diagnostics[0]["selected_body_crop"] is None

    # A valid canonical relative reference stays relative and unchanged.
    assert diagnostics[1]["selected_body_crop"] == relative_id

    # No absolute path and no cluster_N segment leaks anywhere in the payload.
    serialized = json.dumps(payload)
    assert absolute_cluster_path not in serialized
    assert absolute_cluster_path.replace("\\", "/") not in serialized
    # A cluster_N path segment (cluster_0, cluster_12, ...) must not appear;
    # the plain "cluster_id" diagnostic key is metadata, not a path.
    assert re.search(r"cluster_\d", serialized) is None
    # No Windows/POSIX absolute path root (drive letter or leading marker).
    assert re.search(r"[A-Za-z]:[\\/]", serialized) is None
    assert str(isolated_service) not in serialized


def test_status_allows_safe_provisional_staging_face(client, isolated_service):
    face = (
        isolated_service
        / "session"
        / "_staging"
        / "face_crops"
        / "first-face.jpg"
    )
    face.parent.mkdir(parents=True)
    face.write_bytes(b"face")
    with service._jobs_lock:
        service._jobs["job-provisional-face"] = service.JobState(
            "job-provisional-face",
            input_type="camera_uri",
            output_dir=str(isolated_service / "session"),
            snapshot={
                "rolling_analysis": {
                    "enabled": True,
                    "live_identities": [{
                        "live_identity_id": "live_0001",
                        "canonical_person_id": None,
                        "provisional": True,
                        "clustering_state": "unresolved",
                        "best_face_path": str(face),
                    }],
                },
            },
        )

    payload = client.get("/api/person/status/job-provisional-face").get_json()
    identity = payload["snapshot"]["rolling_analysis"]["live_identities"][0]

    assert identity["best_face_path"] == (
        "session/_staging/face_crops/first-face.jpg"
    )
    assert str(isolated_service) not in json.dumps(payload)
