from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
from forensics.global_memory.repair_media_paths import repair_media_paths
from forensics.media_paths import (
    MediaPathError,
    normalize_media_path,
    resolve_media_path,
)
from forensics.person_creation.nodes import finalize as finalize_node


def _embedding() -> list[float]:
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tolist()


def test_relative_write_resolution_and_separator_normalization(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    image = root / "person_001" / "face_crops" / "face one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))

    assert normalize_media_path(image) == "person_001/face_crops/face one.jpg"
    assert normalize_media_path(r"person_001\face_crops\face one.jpg") == (
        "person_001/face_crops/face one.jpg"
    )
    assert resolve_media_path("person_001/face_crops/face one.jpg") == image.resolve()


def test_old_wsl_absolute_path_is_portable_only_for_internal_migration(tmp_path):
    root = tmp_path / "person_db"
    image = root / "person_003" / "face_crops" / "old.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    legacy = "/mnt/d/old/WAYCON/forensics/person_db/person_003/face_crops/old.jpg"

    assert normalize_media_path(legacy, media_root=root) == "person_003/face_crops/old.jpg"
    with pytest.raises(MediaPathError):
        resolve_media_path(legacy, media_root=root, allow_legacy_absolute=False)


@pytest.mark.parametrize("unsafe", ["../outside.jpg", "folder/../../outside.jpg", "/etc/passwd", "C:/secret.txt"])
def test_unsafe_media_paths_are_rejected(tmp_path, unsafe):
    with pytest.raises(MediaPathError):
        resolve_media_path(unsafe, media_root=tmp_path)


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "person_db"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (root / "escape.jpg").symlink_to(outside)

    with pytest.raises(MediaPathError):
        resolve_media_path("escape.jpg", media_root=root)


def _repair_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "person_db"
    face = root / "person_001" / "face_crops" / "good face.jpg"
    body = root / "person_001" / "body_crops" / "body.jpg"
    face.parent.mkdir(parents=True)
    body.parent.mkdir(parents=True)
    face.write_bytes(b"face")
    body.write_bytes(b"body")
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=root)
    memory.register({
        "face_embedding": _embedding(),
        "face_crops": [str(face)],
        "appearance": {
            "date": "2026-07-16",
            "clothing_status": "ok",
            "top": "black jacket",
        },
        "best_body_crops": [str(body)],
        "video_sources": ["video.mp4"],
    })
    memory.close()

    legacy_face = "/mnt/d/old/WAYCON/forensics/person_db/person_001/face_crops/good face.jpg"
    legacy_body = r"C:\old\WAYCON\forensics\person_db\person_001\body_crops\body.jpg"
    connection = sqlite3.connect(database)
    connection.execute("UPDATE persons SET profile_image='/missing/profile.jpg'")
    connection.execute("UPDATE person_gallery SET path=? WHERE crop_type='face'", (legacy_face,))
    connection.execute("UPDATE person_gallery SET path=? WHERE crop_type='body'", (legacy_body,))
    connection.execute(
        "INSERT INTO person_gallery "
        "(person_id, crop_type, path, sharpness, session_date) VALUES (?, ?, ?, ?, ?)",
        ("person_001", "face", "missing/stale.jpg", 0.0, "2026-07-16"),
    )
    connection.execute(
        "UPDATE appearances SET best_body_crops=?",
        (json.dumps([legacy_body, "missing/body.jpg"]),),
    )
    connection.execute("UPDATE recognition_log SET best_face_crop=?", (legacy_face,))
    connection.commit()
    connection.close()
    return database, root


def _path_rows(database: Path) -> tuple:
    connection = sqlite3.connect(database)
    try:
        return (
            connection.execute("SELECT profile_image FROM persons").fetchall(),
            connection.execute("SELECT path FROM person_gallery ORDER BY id").fetchall(),
            connection.execute("SELECT best_body_crops FROM appearances").fetchall(),
            connection.execute("SELECT best_face_crop FROM recognition_log").fetchall(),
        )
    finally:
        connection.close()


def test_repair_dry_run_apply_backup_and_idempotency(tmp_path):
    database, root = _repair_fixture(tmp_path)
    before = _path_rows(database)

    dry = repair_media_paths(database, media_root=root)
    assert dry["dry_run"] is True
    assert dry["backup_path"] is None
    assert dry["rewritten"] >= 3
    assert dry["stale"] >= 1
    assert _path_rows(database) == before

    applied = repair_media_paths(database, media_root=root, dry_run=False)
    assert applied["dry_run"] is False
    assert applied["backup_path"]
    assert applied["gallery_removed"] == 1
    assert (database.parent / applied["backup_path"]).is_file()
    rows_after = _path_rows(database)
    assert rows_after[0] == [("person_001/face_crops/good face.jpg",)]
    assert json.loads(rows_after[2][0][0]) == ["person_001/body_crops/body.jpg"]

    second = repair_media_paths(database, media_root=root, dry_run=False)
    assert second["rewritten"] == 0
    assert _path_rows(database) == rows_after


def test_ambiguous_duplicate_filename_is_reported_not_guessed(tmp_path):
    root = tmp_path / "person_db"
    first = root / "person_001" / "body_crops" / "a" / "same.jpg"
    second = root / "person_001" / "body_crops" / "b" / "same.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=root)
    memory.register({
        "face_embedding": _embedding(),
        "face_crops": [],
        "appearance": {"date": "2026-07-16", "clothing_status": "failed"},
        "best_body_crops": [],
    })
    memory.close()
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE appearances SET best_body_crops=?",
        (json.dumps(["legacy/session/body_crops/same.jpg"]),),
    )
    connection.commit()
    connection.close()

    report = repair_media_paths(database, media_root=root)

    assert report["ambiguous"] == 1


def _finalize_state(root: Path, filename: str = "new.jpg") -> tuple[dict, Path, Path]:
    output = root / "session"
    face = output / "cluster_0" / "face_crops" / filename
    body = output / "cluster_0" / "body_crops" / "body.jpg"
    face.parent.mkdir(parents=True)
    body.parent.mkdir(parents=True)
    face.write_bytes(b"new-face")
    body.write_bytes(b"new-body")
    profile = {
        "face_embedding": _embedding(),
        "cluster_face_count": 3,
        "low_confidence": False,
        "face_crops": [str(face)],
        "face_crop_sharpness": {str(face): 100.0},
        "body_crops": [str(body)],
        "best_body_crops": [str(body)],
        "body_crop_sharpness": {str(body): 100.0},
        "appearance": {
            "date": "2026-07-16",
            "clothing_status": "failed",
        },
        "appearance_signals": {"color": {"samples": []}},
        "video_sources": ["video.mp4"],
    }
    state = {
        "output_dir": str(output),
        "per_cluster_profiles": {0: profile},
        "identity_clusters": [{"cluster_id": 0}],
        "unresolved_faces": [],
        "unattached_bodies": [],
        "quality_face_crops": [{"path": str(face)}],
        "quality_body_crops": [{"path": str(body)}],
    }
    return state, face, body


def test_finalize_copy_first_failure_keeps_original_database_reference_valid(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "person_db"
    database = tmp_path / "memory.db"
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    state, source_face, _source_body = _finalize_state(root)

    def fail_update(self, _person_id, _profile):
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(GlobalMemory, "update_crop_paths", fail_update)
    with pytest.raises(RuntimeError, match="injected update failure"):
        finalize_node.finalize(state)

    assert source_face.is_file()
    assert (root / "person_001" / "face_crops" / source_face.name).is_file()
    memory = GlobalMemory(database, media_root=root)
    try:
        stored = memory._conn.execute("SELECT profile_image FROM persons").fetchone()[0]
        assert stored == "session/cluster_0/face_crops/new.jpg"
        assert resolve_media_path(stored, media_root=root).is_file()
    finally:
        memory.close()


def test_reenrollment_preserves_old_referenced_media_and_prunes_unreferenced(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "person_db"
    database = tmp_path / "memory.db"
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    old_face = root / "person_001" / "face_crops" / "old.jpg"
    orphan = root / "person_001" / "face_crops" / "orphan.jpg"
    old_face.parent.mkdir(parents=True)
    old_face.write_bytes(b"old")
    orphan.write_bytes(b"orphan")
    memory = GlobalMemory(database, media_root=root)
    memory.register({
        "face_embedding": _embedding(),
        "face_crops": [str(old_face)],
        "face_crop_sharpness": {str(old_face): 200.0},
        "appearance": {"date": "2026-07-15", "clothing_status": "failed"},
        "best_body_crops": [],
    })
    memory.close()
    state, _face, _body = _finalize_state(root)

    finalize_node.finalize(state)

    assert old_face.is_file()
    assert not orphan.exists()


def test_database_reference_failure_skips_orphan_pruning(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(tmp_path / "memory.db"))
    state, _face, _body = _finalize_state(root)
    orphan = root / "person_001" / "face_crops" / "uncertain.jpg"

    original = GlobalMemory.referenced_media_paths

    def fail_lookup(self, person_id):
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"uncertain")
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(GlobalMemory, "referenced_media_paths", fail_lookup)
    try:
        finalize_node.finalize(state)
    finally:
        monkeypatch.setattr(GlobalMemory, "referenced_media_paths", original)

    assert orphan.is_file()
