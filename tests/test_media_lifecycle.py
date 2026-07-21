from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from forensics.media_paths import normalize_media_path
from forensics.person_creation import media_lifecycle, service
from forensics.person_creation.nodes import finalize as finalize_node
from forensics.person_creation.nodes.promote_crops import (
    cleanup_promoted_crops,
    promote_crops,
)


def _embedding() -> list[float]:
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tolist()


def _path_values(value, parent_key="") -> list[str]:
    value_keys = {
        "path",
        "crop_path",
        "face_path",
        "body_path",
        "face_crop_path",
        "body_crop_path",
        "representative_face_path",
        "profile_image",
        "best_face_crop",
        "selected_body_crop",
    }
    list_keys = {"face_crops", "body_crops", "best_body_crops"}
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in value_keys and isinstance(item, str) and item:
                found.append(item)
            elif key in list_keys and isinstance(item, list):
                found.extend(raw for raw in item if isinstance(raw, str) and raw)
            else:
                found.extend(_path_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_path_values(item, parent_key))
    return found


@pytest.fixture
def lifecycle_environment(tmp_path, monkeypatch):
    media_root = tmp_path / "person_db"
    database = tmp_path / "global_memory.db"
    media_root.mkdir()
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(media_root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    monkeypatch.setenv("FORENSICS_PROFILE_ROOT", str(media_root))
    service.app.config.update(TESTING=True, GLOBAL_MEMORY_READ_ONLY=False)
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()
    yield media_root, database, service.app.test_client()
    with service._jobs_lock:
        service._jobs.clear()
        service._job_runtimes.clear()


def test_capture_promotion_finalization_cleanup_and_replay_are_canonical(
    lifecycle_environment,
):
    media_root, database, client = lifecycle_environment
    output = media_root / "malek"
    staging_face = output / "_staging" / "face_crops" / "live_face.jpg"
    staging_body = output / "_staging" / "body_crops" / "live_body.jpg"
    staging_face.parent.mkdir(parents=True)
    staging_body.parent.mkdir(parents=True)
    staging_face.write_bytes(b"retained-face")
    staging_body.write_bytes(b"retained-body")

    state = {
        "output_dir": str(output),
        "quality_face_crops": [{"path": str(staging_face), "sharpness": 120.0}],
        "quality_body_crops": [{"path": str(staging_body), "sharpness": 90.0}],
        "frame_groups": [{
            "faces": [{"path": str(staging_face)}],
            "bodies": [{"path": str(staging_body)}],
        }],
        "associations": [{
            "cluster_id": 0,
            "face_path": str(staging_face),
            "body_path": str(staging_body),
        }],
        "identity_clusters": [{
            "cluster_id": 0,
            "face_records": [{"crop_path": str(staging_face)}],
        }],
        "cluster_assignments": {str(staging_face): 0},
        "unresolved_faces": [],
        "unattached_bodies": [
            {"path": str(output / "_staging" / "body_crops" / f"unattached-{i:03d}.jpg")}
            for i in range(128)
        ],
        "media_lifecycle_version": 0,
    }
    job = service.JobState(
        "lifecycle-job",
        input_type="camera_uri",
        status="processing_live_frames",
        output_dir=str(output),
        snapshot=dict(state),
    )
    with service._jobs_lock:
        service._jobs[job.job_id] = job

    staging_payload = client.get(f"/api/person/status/{job.job_id}").get_json()
    staging_path = staging_payload["snapshot"]["quality_face_crops"][0]["path"]
    assert staging_path == "malek/_staging/face_crops/live_face.jpg"
    assert client.get("/api/images", query_string={"path": staging_path}).status_code == 200

    promoted = promote_crops(state)
    assert staging_face.is_file() and staging_body.is_file()
    promoted_face = Path(promoted["quality_face_crops"][0]["path"])
    promoted_body = Path(promoted["quality_body_crops"][0]["path"])
    assert promoted_face.is_file() and promoted_body.is_file()
    service._merge_job_snapshot(job, promoted)
    promoted_payload = client.get(f"/api/person/status/{job.job_id}").get_json()
    promoted_json = json.dumps(promoted_payload)
    assert "_staging" not in promoted_json
    assert staging_path not in promoted_json
    promoted_public = promoted_payload["snapshot"]["quality_face_crops"][0]["path"]
    assert promoted_public == "malek/cluster_0/face_crops/live_face.jpg"
    assert client.get("/api/images", query_string={"path": promoted_public}).status_code == 200

    promoted_state = {**state, **promoted}
    cleanup_promoted_crops(promoted_state)
    assert not staging_face.exists() and not staging_body.exists()
    assert promoted_face.is_file() and promoted_body.is_file()
    (output / "pairing_feedback.json").write_text(json.dumps({
        "confirmed_pairs": [{
            "face_path": str(staging_face),
            "body_path": str(staging_body),
        }],
    }), encoding="utf-8")
    (output / "cluster_0" / "profile.json").write_text("{}", encoding="utf-8")
    (output / "stream_report.json").write_text("{}", encoding="utf-8")

    profile = {
        "face_embedding": _embedding(),
        "cluster_face_count": 3,
        "low_confidence": False,
        "face_crops": [str(promoted_face)],
        "face_crop_sharpness": {str(promoted_face): 120.0},
        "body_crops": [str(promoted_body)],
        "best_body_crops": [str(promoted_body)],
        "body_crop_sharpness": {str(promoted_body): 90.0},
        "appearance": {
            "date": "2026-07-21",
            "clothing_status": "ok",
            "top": "black jacket",
        },
        "appearance_signals": {
            "color": {"samples": [{"path": str(promoted_body)}]},
        },
        "video_sources": ["synthetic-smoke"],
    }
    final_state = {
        **promoted_state,
        "source_type": "live_camera",
        "stream_report_path": str(output / "stream_report.json"),
        "per_cluster_profiles": {0: profile},
        "best_body_crops": [str(promoted_body)],
        "per_cluster_best_body_crops": {0: [str(promoted_body)]},
        "clothing_diagnostics": [{"selected_body_crop": str(promoted_body)}],
        "rolling_analysis": {
            "live_identities": [{"representative_face_path": str(promoted_face)}],
        },
    }
    finalized = finalize_node.finalize(final_state)
    service._merge_job_snapshot(job, finalized)
    job.status = "done"
    job.node = "cleanup_finalized_media"

    final_profile = finalized["per_cluster_profiles"][0]
    assert final_profile["id"] == "person_001"
    assert final_profile["face_crops"] == ["person_001/face_crops/live_face.jpg"]
    assert final_profile["body_crops"] == ["person_001/body_crops/live_body.jpg"]
    assert final_profile["best_body_crops"] == ["person_001/body_crops/live_body.jpg"]
    for path in _path_values(final_profile):
        assert not Path(path).is_absolute()
        assert "\\" not in path
        assert (media_root / path).is_file()

    cleanup_result = finalize_node.cleanup_finalized_media({**final_state, **finalized})
    service._merge_job_snapshot(job, cleanup_result)
    assert cleanup_result["media_cleanup_warning"] == ""
    assert not (output / "_staging").exists()
    assert not (output / "cluster_0").exists()
    assert (media_root / final_profile["face_crops"][0]).is_file()
    assert (media_root / final_profile["body_crops"][0]).is_file()

    final_status = client.get(f"/api/person/status/{job.job_id}")
    assert final_status.status_code == 200
    final_status_payload = final_status.get_json()
    final_status_json = json.dumps(final_status_payload)
    assert "_staging" not in final_status_json
    assert "malek/cluster_" not in final_status_json
    assert str(media_root) not in final_status_json
    assert "C:\\" not in final_status_json
    status_paths = _path_values(final_status_payload)
    assert status_paths
    for path in set(status_paths):
        assert client.get("/api/images", query_string={"path": path}).status_code == 200

    disk_profile = json.loads(
        (media_root / "person_001" / "profile.json").read_text(encoding="utf-8")
    )
    assert _path_values(disk_profile) == _path_values(final_profile)
    assert "malek/" not in json.dumps(disk_profile)
    pairing_report = json.loads((output / "pairing_feedback.json").read_text(encoding="utf-8"))
    assert pairing_report["confirmed_pairs"] == [{
        "face_path": "person_001/face_crops/live_face.jpg",
        "body_path": "person_001/body_crops/live_body.jpg",
    }]
    session_report = json.loads(
        (media_root / "person_001" / "session_report.json").read_text(encoding="utf-8")
    )
    assert session_report["stream_report_path"] == ""

    connection = sqlite3.connect(database)
    try:
        persisted = {
            "persons": connection.execute("SELECT profile_image FROM persons").fetchall(),
            "gallery": connection.execute("SELECT path FROM person_gallery ORDER BY id").fetchall(),
            "appearances": connection.execute("SELECT best_body_crops FROM appearances").fetchall(),
            "log": connection.execute("SELECT best_face_crop FROM recognition_log").fetchall(),
        }
        person_count = connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        gallery_count = connection.execute("SELECT COUNT(*) FROM person_gallery").fetchone()[0]
        log_count = connection.execute("SELECT COUNT(*) FROM recognition_log").fetchone()[0]
    finally:
        connection.close()
    persisted_json = json.dumps(persisted)
    assert "person_001/" in persisted_json
    assert "malek/" not in persisted_json
    assert "_staging" not in persisted_json
    assert gallery_count == 2

    memory_payload = client.get("/api/memory/persons/person_001").get_json()
    memory_list = client.get("/api/memory/persons").get_json()
    gallery_payload = client.get("/api/memory/persons/person_001/gallery").get_json()
    profile_payload = client.get("/api/profiles/person_001").get_json()
    for payload in (memory_payload, memory_list, gallery_payload, profile_payload):
        serialized = json.dumps(payload)
        assert "malek/" not in serialized
        assert str(media_root) not in serialized
        for path in set(_path_values(payload)):
            assert client.get("/api/images", query_string={"path": path}).status_code == 200

    print("MEDIA_LIFECYCLE_SMOKE " + json.dumps({
        "staging": {"path": staging_path, "image_status": 200},
        "promoted": {"path": promoted_public, "image_status": 200},
        "final_paths": [
            {"path": path, "image_status": 200}
            for path in sorted(set(status_paths))
        ],
        "api_statuses": {
            "status": final_status.status_code,
            "memory_detail": 200,
            "memory_list": 200,
            "gallery": 200,
            "profile": 200,
        },
        "forbidden_backend_paths": 0,
        "missing_backend_paths": 0,
    }, sort_keys=True))

    replay = finalize_node.finalize({**final_state, **finalized})
    finalize_node.cleanup_finalized_media({**final_state, **finalized, **replay})
    assert replay["per_cluster_profiles"][0]["face_crops"] == final_profile["face_crops"]
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == person_count
        assert connection.execute("SELECT COUNT(*) FROM person_gallery").fetchone()[0] == gallery_count
        assert connection.execute("SELECT COUNT(*) FROM recognition_log").fetchone()[0] == log_count
    finally:
        connection.close()


def test_relocation_failures_preserve_source_and_validate_replay(
    lifecycle_environment,
    monkeypatch,
):
    media_root, _database, _client = lifecycle_environment
    source = media_root / "session" / "_staging" / "face_crops" / "face.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination = media_root / "session" / "cluster_0" / "face_crops" / "face.jpg"

    original_replace = media_lifecycle.os.replace

    def fail_replace(_source, _destination):
        raise OSError("injected move failure")

    monkeypatch.setattr(media_lifecycle.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected move failure"):
        media_lifecycle.copy_media_for_handoff(source, destination, media_root=media_root)
    assert source.read_bytes() == b"source"
    assert not destination.exists()

    monkeypatch.setattr(media_lifecycle.os, "replace", original_replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"different")
    with pytest.raises(media_lifecycle.MediaRelocationError, match="collision"):
        media_lifecycle.copy_media_for_handoff(source, destination, media_root=media_root)
    assert source.read_bytes() == b"source"
    destination.unlink()
    relative, cleanup = media_lifecycle.copy_media_for_handoff(
        source,
        destination,
        media_root=media_root,
    )
    assert relative == "session/cluster_0/face_crops/face.jpg"
    assert cleanup is not None
    replay_relative, replay_cleanup = media_lifecycle.copy_media_for_handoff(
        source,
        destination,
        media_root=media_root,
    )
    assert replay_relative == relative
    assert replay_cleanup == cleanup
    source.unlink()
    replay_relative, replay_cleanup = media_lifecycle.copy_media_for_handoff(
        source,
        destination,
        media_root=media_root,
    )
    assert replay_relative == relative
    assert replay_cleanup is None

    destination.unlink()
    with pytest.raises(media_lifecycle.MediaRelocationError, match="both missing"):
        media_lifecycle.copy_media_for_handoff(source, destination, media_root=media_root)


def test_cleanup_failure_keeps_canonical_reference_valid(
    lifecycle_environment,
    monkeypatch,
):
    media_root, _database, _client = lifecycle_environment
    canonical = media_root / "person_001" / "face_crops" / "face.jpg"
    obsolete = media_root / "session" / "cluster_0" / "face_crops" / "face.jpg"
    canonical.parent.mkdir(parents=True)
    obsolete.parent.mkdir(parents=True)
    canonical.write_bytes(b"same")
    obsolete.write_bytes(b"same")

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(finalize_node, "cleanup_relocated_sources", fail_cleanup)
    result = finalize_node.cleanup_finalized_media({
        "output_dir": str(media_root / "session"),
        "_media_cleanup_pairs": [
            ("session/cluster_0/face_crops/face.jpg", "person_001/face_crops/face.jpg"),
        ],
    })
    assert "cleanup failed" in result["media_cleanup_warning"]
    assert canonical.read_bytes() == b"same"
    assert obsolete.read_bytes() == b"same"


def test_profile_export_failure_keeps_source_and_canonical_database_references(
    lifecycle_environment,
    monkeypatch,
):
    media_root, database, _client = lifecycle_environment
    output = media_root / "session"
    face = output / "cluster_0" / "face_crops" / "face.jpg"
    body = output / "cluster_0" / "body_crops" / "body.jpg"
    face.parent.mkdir(parents=True)
    body.parent.mkdir(parents=True)
    face.write_bytes(b"face")
    body.write_bytes(b"body")
    state = {
        "output_dir": str(output),
        "per_cluster_profiles": {0: {
            "face_embedding": _embedding(),
            "cluster_face_count": 3,
            "face_crops": [str(face)],
            "body_crops": [str(body)],
            "best_body_crops": [str(body)],
            "appearance": {"date": "2026-07-21", "clothing_status": "ok"},
        }},
        "identity_clusters": [{"cluster_id": 0}],
        "unresolved_faces": [],
        "unattached_bodies": [],
    }
    original_write = finalize_node._write_json

    def fail_profile(path, data):
        if path.name == "profile.json":
            raise OSError("injected profile export failure")
        return original_write(path, data)

    monkeypatch.setattr(finalize_node, "_write_json", fail_profile)
    with pytest.raises(OSError, match="injected profile export failure"):
        finalize_node.finalize(state)

    canonical_face = media_root / "person_001" / "face_crops" / "face.jpg"
    canonical_body = media_root / "person_001" / "body_crops" / "body.jpg"
    assert face.is_file() and body.is_file()
    assert canonical_face.is_file() and canonical_body.is_file()
    assert not (media_root / "person_001" / "profile.json").exists()

    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT profile_image FROM persons").fetchone()[0] == (
            "person_001/face_crops/face.jpg"
        )
        assert {
            row[0] for row in connection.execute("SELECT path FROM person_gallery")
        } == {
            "person_001/face_crops/face.jpg",
            "person_001/body_crops/body.jpg",
        }
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
    finally:
        connection.close()


def test_actual_safe_image_media_lifecycle_probe():
    repository = Path(__file__).resolve().parents[1]
    frontend = repository / "forensics/person_creation/frontend"
    node = shutil.which("node")
    if node is None:
        windows_node = Path("/mnt/c/Program Files/nodejs/node.exe")
        node = str(windows_node) if windows_node.is_file() else None
    assert node is not None, "the existing frontend Node runtime is unavailable"
    result = subprocess.run(
        [node, "scripts/validate-media-lifecycle-component.mjs"],
        cwd=frontend,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "actual media-lifecycle probe passed" in result.stdout


@pytest.mark.parametrize(
    "legacy",
    [
        r"C:\old\WAYCON\forensics\person_db\person_001\face_crops\face.jpg",
        "/mnt/c/old/WAYCON/forensics/person_db/person_001/face_crops/face.jpg",
    ],
)
def test_cross_platform_internal_inputs_emit_posix_relative_paths(
    tmp_path,
    legacy,
):
    root = tmp_path / "person_db"
    image = root / "person_001" / "face_crops" / "face.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"face")
    assert normalize_media_path(
        legacy,
        media_root=root,
        allow_legacy_absolute=True,
        require_exists=True,
    ) == "person_001/face_crops/face.jpg"
