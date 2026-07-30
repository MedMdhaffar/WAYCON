from __future__ import annotations

import io
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from werkzeug.datastructures import FileStorage

from forensics.global_memory import GlobalMemory
import forensics.global_memory.store as global_memory_store
from forensics.person_creation import profile_management as pm
from forensics.person_creation.profile_management import (
    ProfileImportManager,
    commit_fingerprint,
    content_set_fingerprint,
    filename_to_profile_name,
    resolve_profile_image,
)


# --- helpers -----------------------------------------------------------------

def unit(index: int, dimensions: int = 512) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    vector[index] = 1.0
    return vector


def mixed(theta: float) -> np.ndarray:
    """A unit vector at angle ``theta`` from ``unit(0)`` in the e0/e1 plane."""
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = math.cos(theta)
    vector[1] = math.sin(theta)
    return vector


class TaggedFaceEngine:
    """Deterministic stand-in keyed by a tag pixel baked into each image."""

    def __init__(self, embeddings=None, default=None, bbox=None, delay=0.0):
        self.embeddings = embeddings or {}
        self.default = default if default is not None else unit(0)
        self.bbox = bbox
        self.delay = delay

    def detect(self, image):
        if self.delay:
            time.sleep(self.delay)
        width = image.shape[1]
        if width == 80:
            return []
        if width == 90:
            return [
                {"bbox": [5, 5, 70, 70], "confidence": 0.99},
                {"bbox": [20, 20, 80, 80], "confidence": 0.98},
            ]
        return [{
            "bbox": self.bbox or [8, 8, width - 8, image.shape[0] - 8],
            "confidence": 0.99,
        }]

    def embed(self, crop):
        return self.embeddings.get(int(crop[0, 0, 0]), self.default)


def image_file(name: str, width: int = 120, tag: int | None = None) -> FileStorage:
    """A deterministic PNG whose bytes depend on ``name``.

    Distinct filenames therefore produce distinct content hashes (two different
    photos), while the same filename always produces identical bytes (a genuine
    re-upload).  ``tag`` selects which embedding the fake engine returns and is
    written where the detector's crop origin lands.
    """
    rows, columns = np.indices((120, width))
    checker = ((rows // 4 + columns // 4) % 2 * 255).astype(np.uint8)
    image = cv2.merge([checker, 255 - checker, checker])
    salt = zlib.crc32(name.encode("utf-8"))
    image[1, 1] = (salt & 255, (salt >> 8) & 255, (salt >> 16) & 255)
    if tag is not None:
        image[0, 0] = (tag, tag, tag)
        image[8, 8] = (tag, tag, tag)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return FileStorage(
        stream=io.BytesIO(encoded.tobytes()),
        filename=name,
        content_type="image/png",
    )


def same_image_bytes(file: FileStorage, filename: str) -> FileStorage:
    file.stream.seek(0)
    payload = file.stream.read()
    return FileStorage(
        stream=io.BytesIO(payload),
        filename=filename,
        content_type=file.content_type,
    )


REAL_RUNTIME_DATABASE = (
    Path(__file__).resolve().parent.parent / "forensics" / "global_memory.db"
)
PROTECTED_RUNTIME_FILES = tuple(
    Path(str(REAL_RUNTIME_DATABASE) + suffix) for suffix in ("", "-wal", "-shm")
)


def protected_file_state(path: Path) -> tuple[bool, int | None, int | None, str | None]:
    if not path.exists():
        return (False, None, None, None)
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (True, stat.st_size, stat.st_mtime_ns, digest)


def protected_store_state(paths=PROTECTED_RUNTIME_FILES) -> dict[str, tuple]:
    return {str(path): protected_file_state(path) for path in paths}


def assert_protected_store_unchanged(before: dict[str, tuple], paths=PROTECTED_RUNTIME_FILES):
    after = protected_store_state(paths)
    assert after == before, (
        "the protected runtime Global Memory DB/WAL/SHM state changed: "
        f"before={before!r}, after={after!r}"
    )


@pytest.fixture(autouse=True)
def protect_runtime_database():
    """Protect the runtime DB and both SQLite sidecars without opening SQLite."""
    before = protected_store_state()
    try:
        yield
    finally:
        assert_protected_store_unchanged(before)


@pytest.fixture
def profile_environment(tmp_path):
    """Redirect Global Memory to a temporary database.

    The redirection deliberately does NOT use ``monkeypatch``: a test that calls
    ``monkeypatch.undo()`` would otherwise revert it mid-test and start writing
    to the real runtime database.
    """
    database = tmp_path / "memory.db"
    media = tmp_path / "media"
    original_db_path = global_memory_store.config.DB_PATH
    original_environment = {
        key: os.environ.get(key)
        for key in (
            "FORENSICS_MEMORY_DB",
            "PERSON_CREATION_MEDIA_ROOT",
            "PROFILE_IMPORT_WORKERS",
            "PROFILE_IMPORT_PROTECTED_RUNTIME_DB",
        )
    }
    managers: list[ProfileImportManager] = []

    def build(engine=None):
        manager = ProfileImportManager(
            media_root=media,
            face_engine=engine or TaggedFaceEngine(),
        )
        managers.append(manager)
        return manager

    try:
        global_memory_store.config.DB_PATH = str(database)
        os.environ["FORENSICS_MEMORY_DB"] = str(database)
        os.environ["PERSON_CREATION_MEDIA_ROOT"] = str(media)
        os.environ["PROFILE_IMPORT_WORKERS"] = "2"
        os.environ["PROFILE_IMPORT_PROTECTED_RUNTIME_DB"] = str(REAL_RUNTIME_DATABASE)
        assert database.resolve() != REAL_RUNTIME_DATABASE.resolve()
        GlobalMemory(db_path=str(database), media_root=media).close()
        yield build, database, media
    finally:
        for manager in managers:
            manager.close()
        global_memory_store.config.DB_PATH = original_db_path
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def commit_one(manager, owner, filename, name, *, tag=None, action="create_new"):
    batch = manager.create_batch([image_file(filename, tag=tag)], owner)
    ready = manager.wait_ready(batch["batch_id"], owner)
    identity = ready["identities"][0]
    result = manager.commit(batch["batch_id"], owner, [{
        "identity_id": identity["identity_id"],
        "name": name,
        "action": action,
        "primary_source_id": identity["primary_source_id"],
    }])
    return result["results"][0]


def census(database):
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = sorted(
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
        return {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for table in tables
        }
    finally:
        connection.close()


# --- naming ------------------------------------------------------------------

def test_filename_to_name_conversion():
    assert filename_to_profile_name("Hadil_Karous.jpg") == "Hadil Karous"
    assert filename_to_profile_name("mohamed-trabelsi.png") == "Mohamed Trabelsi"
    assert filename_to_profile_name("../../SAFIA profile.WEBP") == "Safia Profile"


# --- image validation --------------------------------------------------------

def test_exactly_one_face_and_per_image_failure_isolation(profile_environment):
    build, _database, _media = profile_environment
    manager = build()
    batch = manager.create_batch(
        [
            image_file("valid_person.png"),
            image_file("no-face.png", width=80),
            image_file("many-faces.png", width=90),
            FileStorage(io.BytesIO(b"not an image"), filename="broken.png", content_type="image/png"),
        ],
        "supervisor",
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    states = {item["source_filename"]: item["state"] for item in ready["items"]}
    assert states == {
        "valid_person.png": "valid",
        "no-face.png": "no_face",
        "many-faces.png": "multiple_faces",
        "broken.png": "failed",
    }
    assert ready["progress"]["valid"] == 1
    assert ready["progress"]["failed"] == 3
    assert ready["can_commit"] is True


def test_non_image_payloads_are_rejected_by_decoded_content(profile_environment):
    build, _database, _media = profile_environment
    manager = build()
    batch = manager.create_batch(
        [
            FileStorage(io.BytesIO(b"#!/bin/sh\nrm -rf /\n"), filename="evil.jpg", content_type="image/jpeg"),
            FileStorage(io.BytesIO(b"\x7fELF\x02\x01\x01" + b"\x00" * 200), filename="mal.png", content_type="image/png"),
            FileStorage(io.BytesIO(b"%PDF-1.4"), filename="doc.pdf", content_type="application/pdf"),
            image_file("real.png"),
        ],
        "supervisor",
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    states = {item["source_filename"]: item["state"] for item in ready["items"]}
    assert states["evil.jpg"] == "failed"
    assert states["mal.png"] == "failed"
    assert states["doc.pdf"] == "failed"
    assert states["real.png"] == "valid"


def test_crop_bounds_and_embedding_dimensions_are_enforced(profile_environment):
    build, _database, _media = profile_environment
    outside = build(TaggedFaceEngine(bbox=[-500, -500, 99999, 99999]))
    ready = outside.wait_ready(
        outside.create_batch([image_file("oob.png")], "supervisor")["batch_id"], "supervisor"
    )
    x1, y1, x2, y2 = ready["items"][0]["bbox"]
    assert ready["items"][0]["state"] == "valid"
    assert 0 <= x1 < x2 <= 120 and 0 <= y1 < y2 <= 120

    inverted = build(TaggedFaceEngine(bbox=[10, 10, 5, 5]))
    ready = inverted.wait_ready(
        inverted.create_batch([image_file("inv.png")], "supervisor")["batch_id"], "supervisor"
    )
    assert ready["items"][0]["state"] == "failed"

    for bad in (np.zeros(128, np.float32), np.zeros(512, np.float32), np.full(512, np.nan, np.float32)):
        manager = build(TaggedFaceEngine(default=bad))
        ready = manager.wait_ready(
            manager.create_batch([image_file("bad.png")], "supervisor")["batch_id"], "supervisor"
        )
        assert ready["items"][0]["state"] == "failed"


def test_path_traversal_filename_is_never_used_as_a_storage_path(profile_environment):
    build, _database, media = profile_environment
    manager = build()
    batch = manager.create_batch([image_file("../../escape.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    item = ready["items"][0]
    assert ".." not in item["original_photo"]
    assert item["original_photo"].startswith(f"_profile_imports/{batch['batch_id']}/uploads/")
    assert not (media.parent / "escape.png").exists()


# --- grouping ----------------------------------------------------------------

def test_non_transitive_grouping_splits_inconsistent_photos(profile_environment):
    """A~B and B~C pass but A~C fails: the three must NOT become one identity."""
    build, _database, _media = profile_environment
    theta = math.acos(0.45)
    a, b, c = mixed(0.0), mixed(theta / 2), mixed(theta)
    assert float(a @ b) >= 0.78 and float(b @ c) >= 0.78 and float(a @ c) < 0.78
    manager = build(TaggedFaceEngine({10: a, 20: b, 30: c}))
    batch = manager.create_batch(
        [image_file("a.png", tag=10), image_file("b.png", tag=20), image_file("c.png", tag=30)],
        "supervisor",
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    groups = [sorted(photo["source_filename"] for photo in row["photos"]) for row in ready["identities"]]
    assert ["a.png", "b.png"] in groups
    assert ["c.png"] in groups
    assert len(ready["identities"]) == 2
    for identity in ready["identities"]:
        embeddings = {"a.png": a, "b.png": b, "c.png": c}
        members = [embeddings[photo["source_filename"]] for photo in identity["photos"]]
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                assert float(members[left] @ members[right]) >= manager.batch_duplicate_threshold


def test_grouping_is_deterministic_regardless_of_completion_order(profile_environment):
    build, _database, _media = profile_environment
    theta = math.acos(0.45)
    embeddings = {10: mixed(0.0), 20: mixed(theta / 2), 30: mixed(theta)}
    signatures = []
    for delay in (0.0, 0.04):
        manager = build(TaggedFaceEngine(embeddings, delay=delay))
        batch = manager.create_batch(
            [
                image_file("a.png", tag=10),
                image_file("b.png", tag=20),
                image_file("c.png", tag=30),
            ],
            "supervisor",
        )
        ready = manager.wait_ready(batch["batch_id"], "supervisor")
        signatures.append([
            sorted(photo["source_filename"] for photo in row["photos"])
            for row in ready["identities"]
        ])
    assert signatures[0] == signatures[1]


def test_identical_and_genuine_duplicates_group_into_one_identity(profile_environment):
    build, _database, _media = profile_environment
    close = mixed(math.acos(0.93))
    manager = build(TaggedFaceEngine({10: unit(0), 20: unit(0), 30: close}))
    batch = manager.create_batch(
        [
            image_file("hadil_1.png", tag=10),
            image_file("hadil_2.png", tag=20),
            image_file("hadil_3.png", tag=30),
        ],
        "supervisor",
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    assert len(ready["identities"]) == 1
    assert ready["identities"][0]["grouped_photo_count"] == 3


def test_two_distinct_people_stay_separate(profile_environment):
    build, _database, _media = profile_environment
    manager = build(TaggedFaceEngine({10: unit(0), 20: unit(7)}))
    batch = manager.create_batch(
        [image_file("aziz.png", tag=10), image_file("melek.png", tag=20)], "supervisor"
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    assert len(ready["identities"]) == 2
    assert all(row["grouped_photo_count"] == 1 for row in ready["identities"])


def test_duplicates_plus_invalid_image_keep_valid_group_intact(profile_environment):
    build, _database, _media = profile_environment
    manager = build()
    batch = manager.create_batch(
        [
            image_file("dup_a.png"),
            image_file("dup_b.png"),
            FileStorage(io.BytesIO(b"broken"), filename="bad.png", content_type="image/png"),
        ],
        "supervisor",
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    grouped = [row for row in ready["identities"] if row["validation_error"] is None]
    failed = [row for row in ready["identities"] if row["validation_error"]]
    assert len(grouped) == 1 and grouped[0]["grouped_photo_count"] == 2
    assert len(failed) == 1 and failed[0]["proposed_action"] == "skip"


# --- coordinator bounds ------------------------------------------------------

def test_maximum_size_batch_is_fully_processed(profile_environment, monkeypatch):
    """Back pressure must slow the producer, never fail valid images."""
    build, _database, _media = profile_environment
    monkeypatch.setenv("PROFILE_IMPORT_MAX_FILES", "40")
    monkeypatch.setenv("PROFILE_IMPORT_QUEUE_CAPACITY", "4")
    manager = build(TaggedFaceEngine(delay=0.002))
    assert manager.max_files == 40
    batch = manager.create_batch(
        [image_file(f"p{index}.png") for index in range(40)], "supervisor"
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor", timeout=180)
    states = [item["state"] for item in ready["items"]]
    assert len(states) == 40
    assert all(state == "valid" for state in states), states
    errors = [item["error"] for item in ready["items"] if item["error"]]
    assert not any("queue capacity" in str(error) for error in errors)


def test_worker_pool_stays_bounded(profile_environment, monkeypatch):
    build, _database, _media = profile_environment
    monkeypatch.setenv("PROFILE_IMPORT_WORKERS", "64")
    manager = build()
    assert len(manager._workers) == 2


# --- preview isolation -------------------------------------------------------

def test_preview_makes_no_durable_change(profile_environment):
    build, database, _media = profile_environment
    manager = build()
    commit_one(manager, "supervisor", "Seed.png", "Seed")
    baseline = census(database)
    manager.wait_ready(
        manager.create_batch(
            [image_file("A.png"), image_file("B.png"), FileStorage(io.BytesIO(b"x"), filename="c.png")],
            "supervisor",
        )["batch_id"],
        "supervisor",
    )
    assert census(database) == baseline


# --- rename consent ----------------------------------------------------------

def test_attach_without_consent_does_not_rename(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Khalifa.png", "Khalifa Bouneb")["person_id"]
    batch = manager.create_batch([image_file("IMG_20260729.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    assert identity["proposed_action"] == "attach_existing"
    manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": identity["proposed_name"],
        "action": "attach_existing",
        "existing_person_id": person_id,
        "primary_source_id": identity["primary_source_id"],
    }])
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory.get_person(person_id)["name"] == "Khalifa Bouneb"
    finally:
        memory.close()


# --- corrected narrow-audit regressions -------------------------------------

def _normalize_for_test(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.linalg.norm(value)


def test_content_set_key_uses_only_sorted_unique_sha256_values():
    first = "a" * 64
    second = "b" * 64
    assert content_set_fingerprint([second, first]) == content_set_fingerprint(
        [first, second, first]
    )


def test_complete_linkage_tie_joins_first_stable_group(profile_environment):
    build, _database, _media = profile_environment
    manager = build()
    manager.batch_duplicate_threshold = 0.7
    groups = manager.group_by_complete_linkage([
        {"index": 0, "label": "left", "_embedding": unit(0)},
        {"index": 1, "label": "right", "_embedding": unit(1)},
        {
            "index": 2,
            "label": "candidate",
            "_embedding": _normalize_for_test(unit(0) + unit(1)),
        },
    ])
    assert [[row["label"] for row in group] for group in groups] == [
        ["left", "candidate"],
        ["right"],
    ]


def test_cross_intent_and_different_target_are_http_409_without_mutation(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_a = commit_one(manager, "supervisor", "Person_A.png", "Person A")["person_id"]
    person_b = commit_one(manager, "supervisor", "Person_B.png", "Person B", tag=2)["person_id"]
    first_batch = manager.create_batch([image_file("attach-once.png")], "supervisor")
    first_ready = manager.wait_ready(first_batch["batch_id"], "supervisor")
    first_identity = first_ready["identities"][0]
    attached = manager.commit(first_batch["batch_id"], "supervisor", [{
        "identity_id": first_identity["identity_id"],
        "action": "attach_existing",
        "existing_person_id": person_a,
    }])
    assert attached["results"][0]["status"] == "updated"

    def counts():
        memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
        try:
            return (
                memory.get_person(person_a)["embedding_count"],
                memory.get_person(person_b)["embedding_count"],
                memory._conn.execute("SELECT COUNT(*) FROM face_photo_sources").fetchone()[0],
                memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0],
            )
        finally:
            memory.close()

    before = counts()
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()

    created_batch = manager.create_batch([image_file("Person_A.png")], "supervisor")
    created_ready = manager.wait_ready(created_batch["batch_id"], "supervisor")
    created_identity = created_ready["identities"][0]
    create_then_attach = client.post(
        f"/api/profiles/import/{created_batch['batch_id']}/commit",
        headers={"X-WAYCON-Supervisor": "supervisor"},
        json={"identities": [{
            "identity_id": created_identity["identity_id"],
            "action": "attach_existing",
            "existing_person_id": person_b,
        }]},
    )
    assert create_then_attach.status_code == 409
    assert create_then_attach.get_json()["code"] == "operation_conflict"
    assert counts() == before

    def conflicting_response(action, target=None):
        batch = manager.create_batch([image_file("attach-once.png")], "supervisor")
        ready = manager.wait_ready(batch["batch_id"], "supervisor")
        identity = ready["identities"][0]
        return client.post(
            f"/api/profiles/import/{batch['batch_id']}/commit",
            headers={"X-WAYCON-Supervisor": "supervisor"},
            json={"identities": [{
                "identity_id": identity["identity_id"],
                "name": "Must conflict",
                "action": action,
                "existing_person_id": target,
            }]},
        )

    different_target = conflicting_response("attach_existing", person_b)
    assert different_target.status_code == 409
    assert different_target.get_json()["code"] == "operation_conflict"
    assert counts() == before
    changed_action = conflicting_response("create_new")
    assert changed_action.status_code == 409
    assert changed_action.get_json()["code"] == "operation_conflict"
    assert counts() == before


def test_same_bytes_different_filename_same_attach_intent_replays(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Holder.png", "Holder")["person_id"]
    original = image_file("content-origin.png")
    retry = same_image_bytes(original, "renamed-content.png")
    original.stream.seek(0)

    def attach(file):
        batch = manager.create_batch([file], "supervisor")
        ready = manager.wait_ready(batch["batch_id"], "supervisor")
        identity = ready["identities"][0]
        return manager.commit(batch["batch_id"], "supervisor", [{
            "identity_id": identity["identity_id"],
            "action": "attach_existing",
            "existing_person_id": person_id,
        }])["results"][0]

    first = attach(original)
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        before = (
            memory.get_person(person_id)["embedding_count"],
            memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0],
        )
    finally:
        memory.close()
    replay = attach(retry)
    assert replay["person_id"] == first["person_id"]
    assert replay["idempotent_replay"] is True
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert (
            memory.get_person(person_id)["embedding_count"],
            memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0],
        ) == before
    finally:
        memory.close()


def test_partial_overlap_applies_only_the_retained_embedding(profile_environment):
    build, database, media = profile_environment
    vector_b = unit(0)
    vector_c = mixed(math.acos(0.90))
    manager = build(TaggedFaceEngine(
        embeddings={0: vector_b, 1: vector_c},
        default=vector_b,
    ))

    first_batch = manager.create_batch(
        [image_file("A.png", tag=0), image_file("B.png", tag=0)],
        "supervisor",
    )
    first_ready = manager.wait_ready(first_batch["batch_id"], "supervisor")
    first_identity = first_ready["identities"][0]
    created = manager.commit(first_batch["batch_id"], "supervisor", [{
        "identity_id": first_identity["identity_id"],
        "name": "Partial Overlap",
        "action": "create_new",
    }])["results"][0]
    person_id = created["person_id"]

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        before = memory.get_person(person_id)
        before_embedding = np.asarray(before["embedding"], dtype=np.float32)
        before_count = before["embedding_count"]
        before_phone = memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources WHERE person_id=?", (person_id,)
        ).fetchone()[0]
        before_evidence = memory._conn.execute(
            "SELECT COUNT(*) FROM identity_evidence WHERE person_id=?", (person_id,)
        ).fetchone()[0]
        before_gallery = memory._conn.execute(
            "SELECT COUNT(*) FROM person_gallery WHERE person_id=?", (person_id,)
        ).fetchone()[0]
        before_log = memory._conn.execute(
            "SELECT COUNT(*) FROM recognition_log WHERE person_id=?", (person_id,)
        ).fetchone()[0]
    finally:
        memory.close()

    overlap_batch = manager.create_batch(
        [image_file("B.png", tag=0), image_file("C.png", tag=1)],
        "supervisor",
    )
    overlap_ready = manager.wait_ready(overlap_batch["batch_id"], "supervisor")
    overlap_identity = overlap_ready["identities"][0]
    result = manager.commit(overlap_batch["batch_id"], "supervisor", [{
        "identity_id": overlap_identity["identity_id"],
        "action": "attach_existing",
        "existing_person_id": person_id,
    }])["results"][0]
    assert result["status"] == "updated"

    expected = _normalize_for_test(before_embedding * before_count + vector_c)
    grouped_bc = _normalize_for_test(np.mean([vector_b, vector_c], axis=0))
    incorrect = _normalize_for_test(before_embedding * before_count + grouped_bc)
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        after = memory.get_person(person_id)
        stored = np.asarray(after["embedding"], dtype=np.float32)
        assert after["embedding_count"] == before_count + 1
        assert np.allclose(stored, expected, rtol=1e-7, atol=1e-7)
        assert not np.allclose(stored, incorrect, rtol=1e-5, atol=1e-5)
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before_phone + 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM identity_evidence WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before_evidence + 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM person_gallery WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before_gallery + 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM recognition_log WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before_log + 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources "
            "WHERE person_id=? AND source_filename='B.png'",
            (person_id,),
        ).fetchone()[0] == 1
    finally:
        memory.close()


def test_all_owned_subset_returns_without_identity_update(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "seed-owned.png", "Owned")["person_id"]

    batch = manager.create_batch(
        [image_file("owned-A.png"), image_file("owned-B.png")], "supervisor"
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "action": "attach_existing",
        "existing_person_id": person_id,
    }])

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        before = (
            memory.get_person(person_id)["embedding_count"],
            np.asarray(memory.get_person(person_id)["embedding"], dtype=np.float32),
            memory._conn.execute(
                "SELECT COUNT(*) FROM face_photo_sources WHERE person_id=?", (person_id,)
            ).fetchone()[0],
            memory._conn.execute(
                "SELECT COUNT(*) FROM identity_evidence WHERE person_id=?", (person_id,)
            ).fetchone()[0],
        )
    finally:
        memory.close()

    subset = manager.create_batch([image_file("owned-A.png")], "supervisor")
    subset_ready = manager.wait_ready(subset["batch_id"], "supervisor")
    subset_identity = subset_ready["identities"][0]
    replay = manager.commit(subset["batch_id"], "supervisor", [{
        "identity_id": subset_identity["identity_id"],
        "action": "attach_existing",
        "existing_person_id": person_id,
    }])["results"][0]
    assert replay["idempotent_replay"] is True
    assert replay["duplicate_evidence_skipped"] is True

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        after = memory.get_person(person_id)
        assert after["embedding_count"] == before[0]
        assert np.array_equal(np.asarray(after["embedding"], dtype=np.float32), before[1])
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before[2]
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM identity_evidence WHERE person_id=?", (person_id,)
        ).fetchone()[0] == before[3]
    finally:
        memory.close()


def test_two_retained_embeddings_are_order_independent(profile_environment):
    build, database, media = profile_environment
    vector_b = unit(0)
    angle = 0.10
    vector_c = mixed(angle)
    vector_d = mixed(-angle)
    engine = TaggedFaceEngine(
        embeddings={0: vector_b, 1: vector_c, 2: vector_d, 3: unit(3)},
        default=vector_b,
    )
    manager = build(engine)

    def attach_with_order(prefix, ordered_new):
        person_id = commit_one(
            manager, "supervisor", f"{prefix}-seed.png", prefix, tag=3
        )["person_id"]
        existing_name = f"{prefix}-B.png"
        existing_batch = manager.create_batch(
            [image_file(existing_name, tag=0)], "supervisor"
        )
        existing_ready = manager.wait_ready(existing_batch["batch_id"], "supervisor")
        existing_identity = existing_ready["identities"][0]
        manager.commit(existing_batch["batch_id"], "supervisor", [{
            "identity_id": existing_identity["identity_id"],
            "action": "attach_existing",
            "existing_person_id": person_id,
        }])
        memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
        try:
            before = memory.get_person(person_id)
        finally:
            memory.close()
        files = [image_file(existing_name, tag=0)] + [
            image_file(f"{prefix}-{name}.png", tag=tag) for name, tag in ordered_new
        ]
        second_batch = manager.create_batch(files, "supervisor")
        second_ready = manager.wait_ready(second_batch["batch_id"], "supervisor")
        second_identity = second_ready["identities"][0]
        manager.commit(second_batch["batch_id"], "supervisor", [{
            "identity_id": second_identity["identity_id"],
            "action": "attach_existing",
            "existing_person_id": person_id,
        }])
        memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
        try:
            after = memory.get_person(person_id)
            assert after["embedding_count"] == before["embedding_count"] + 2
            retained = _normalize_for_test(np.mean([vector_c, vector_d], axis=0))
            expected = _normalize_for_test(
                np.asarray(before["embedding"], dtype=np.float32)
                * before["embedding_count"]
                + retained * 2
            )
            assert np.allclose(after["embedding"], expected, rtol=1e-7, atol=1e-7)
            return np.asarray(after["embedding"], dtype=np.float32)
        finally:
            memory.close()

    forward = attach_with_order("forward", [("C", 1), ("D", 2)])
    reverse = attach_with_order("reverse", [("D", 2), ("C", 1)])
    assert np.allclose(forward, reverse, rtol=1e-7, atol=1e-7)


def test_concurrent_different_intents_allow_one_and_conflict_one(profile_environment):
    build, database, media = profile_environment
    target = commit_one(build(), "supervisor", "Concurrent_Target.png", "Target")["person_id"]
    batches = []
    for manager in (build(), build()):
        batch = manager.create_batch([image_file("raced-content.png")], "supervisor")
        ready = manager.wait_ready(batch["batch_id"], "supervisor")
        batches.append((manager, batch, ready["identities"][0]))
    barrier = threading.Barrier(2)
    successes, conflicts = [], []

    def run(index):
        manager, batch, identity = batches[index]
        change = {
            "identity_id": identity["identity_id"],
            "name": "Race",
            "action": "create_new" if index == 0 else "attach_existing",
            "existing_person_id": target if index == 1 else None,
        }
        barrier.wait()
        try:
            successes.append(manager.commit(batch["batch_id"], "supervisor", [change]))
        except pm.OperationConflict as exc:
            conflicts.append(str(exc))

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1 and len(conflicts) == 1
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources "
            "WHERE source_filename='raced-content.png'"
        ).fetchone()[0] == 1
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM profile_import_content_sets"
        ).fetchone()[0] == 2
    finally:
        memory.close()


def _create_pending_review(build):
    candidate = commit_one(
        build(), "supervisor", "Review_Candidate.png", "Candidate"
    )["person_id"]
    manager = build(TaggedFaceEngine(default=mixed(math.acos(0.72))))
    batch = manager.create_batch([image_file("Review_Evidence.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Approved Review Name",
        "action": "review_required",
    }])["results"][0]
    return manager, candidate, result["review_key"]


@pytest.mark.parametrize("action", ["attach_existing", "create_new", "skip"])
def test_durable_review_resolution_actions_retry_and_conflict(profile_environment, action):
    build, database, media = profile_environment
    manager, candidate, review_key = _create_pending_review(build)
    manager.close()
    restarted = build()
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=restarted)
    client = app.test_client()
    before_people = len(client.get("/api/profiles?state=all").get_json()["profiles"])
    payload = {
        "action": action,
        "target_person_id": candidate if action == "attach_existing" else None,
        "name": "Created From Review" if action == "create_new" else "",
        "notes": "review resolution",
    }
    response = client.post(f"/api/profiles/reviews/{review_key}/resolve", json=payload)
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert "embedding" not in json.dumps(result).lower()
    assert result["status"] == {
        "attach_existing": "updated",
        "create_new": "created",
        "skip": "skipped",
    }[action]
    assert client.get("/api/profiles/reviews").get_json()["reviews"] == []
    replay = client.post(f"/api/profiles/reviews/{review_key}/resolve", json=payload)
    assert replay.status_code == 200
    assert replay.get_json()["idempotent_replay"] is True
    conflicting_action = "skip" if action != "skip" else "create_new"
    conflict = client.post(f"/api/profiles/reviews/{review_key}/resolve", json={
        "action": conflicting_action,
        "name": "Other resolution",
    })
    assert conflict.status_code == 409
    assert conflict.get_json()["code"] == "operation_conflict"

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        stored = memory._conn.execute(
            "SELECT status, resolution_action, embedding "
            "FROM profile_import_reviews WHERE review_key=?",
            (review_key,),
        ).fetchone()
        assert stored["status"] == ("rejected" if action == "skip" else "accepted")
        assert stored["resolution_action"] == action
        assert stored["embedding"] is not None
        assert len(memory.list_all()) == before_people + (action == "create_new")
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM face_photo_sources WHERE import_batch_id=?",
            (f"review:{review_key}",),
        ).fetchone()[0] == (0 if action == "skip" else 1)
    finally:
        memory.close()


def _insert_appearance(connection, person_id, day, **values):
    columns = ["person_id", "date", *values]
    connection.execute(
        f"INSERT INTO appearances({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        [person_id, day, *values.values()],
    )


def test_merge_reconciles_no_collision_and_multiple_same_date_collisions(profile_environment):
    build, database, media = profile_environment
    manager, source, target = build_merge_pair(build, database, media)
    memory = GlobalMemory(db_path=str(database), media_root=media)
    try:
        _insert_appearance(
            memory._conn, source, "2026-01-01", top="source only",
            clothing_status="ok", best_body_crops=json.dumps(["source-only-body"]),
            video_sources=json.dumps(["source-only-video"]),
        )
        _insert_appearance(
            memory._conn, target, "2026-01-02", top="target top", top_color="red",
            clothing_status="ok", best_body_crops=json.dumps(["shared", "target-body"]),
            video_sources=json.dumps(["target-video"]),
        )
        _insert_appearance(
            memory._conn, source, "2026-01-02", top="source top",
            bottom="source bottom", bottom_color="blue", clothing_status="ok",
            best_body_crops=json.dumps(["shared", "source-body"]),
            video_sources=json.dumps(["target-video", "source-video"]),
        )
        _insert_appearance(
            memory._conn, target, "2026-01-03",
            top="target wins tie", clothing_status="failed",
        )
        _insert_appearance(
            memory._conn, source, "2026-01-03",
            top="source loses tie", clothing_status="ok",
        )
    finally:
        memory.close()
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    response = app.test_client().post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    })
    assert response.status_code == 200, response.get_json()
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM appearances WHERE person_id=?", (source,)
        ).fetchone()[0] == 0
        rows = {
            row["date"]: row for row in memory._conn.execute(
                "SELECT * FROM appearances WHERE person_id=?", (target,)
            ).fetchall()
        }
        assert set(rows) == {"2026-01-01", "2026-01-02", "2026-01-03"}
        assert rows["2026-01-01"]["top"] == "source only"
        assert rows["2026-01-02"]["top"] == "source top"
        assert rows["2026-01-02"]["top_color"] == "red"
        assert rows["2026-01-02"]["bottom"] == "source bottom"
        assert json.loads(rows["2026-01-02"]["best_body_crops"]) == [
            "shared", "target-body", "source-body",
        ]
        assert json.loads(rows["2026-01-02"]["video_sources"]) == [
            "target-video", "source-video",
        ]
        assert rows["2026-01-03"]["top"] == "target wins tie"
    finally:
        memory.close()


def test_same_date_reconciliation_failure_rolls_back_and_retry_succeeds(
    profile_environment, monkeypatch
):
    build, database, media = profile_environment
    manager, source, target = build_merge_pair(build, database, media)
    memory = GlobalMemory(db_path=str(database), media_root=media)
    try:
        _insert_appearance(memory._conn, source, "2026-04-01", top="source")
        _insert_appearance(memory._conn, target, "2026-04-01", bottom="target")
    finally:
        memory.close()
    before = census(database)
    original = pm._merge_same_date_appearance
    armed = {"value": True}

    def fail_once(*args, **kwargs):
        if armed["value"]:
            armed["value"] = False
            raise sqlite3.IntegrityError("injected same-date reconciliation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "_merge_same_date_appearance", fail_once)
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    request_body = {
        "source_person_id": source, "target_person_id": target, "confirm": True,
    }
    assert client.post("/api/profiles/merge", json=request_body).status_code == 409
    assert census(database) == before
    assert client.post("/api/profiles/merge", json=request_body).status_code == 200
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM appearances WHERE person_id=?", (source,)
        ).fetchone()[0] == 0
    finally:
        memory.close()


def test_wal_only_mutation_is_detected_by_protected_store_guard(tmp_path):
    database = tmp_path / "protected.db"
    wal = Path(str(database) + "-wal")
    shm = Path(str(database) + "-shm")
    database.write_bytes(b"main")
    wal.write_bytes(b"wal-before")
    shm.write_bytes(b"shm")
    paths = (database, wal, shm)
    before = protected_store_state(paths)
    wal.write_bytes(b"wal-after-with-different-size")
    with pytest.raises(AssertionError, match="DB/WAL/SHM"):
        assert_protected_store_unchanged(before, paths)


def test_profile_memory_refuses_protected_runtime_path_before_sqlite_open(
    tmp_path, monkeypatch
):
    protected = tmp_path / "must-not-open.db"
    monkeypatch.setenv("PROFILE_IMPORT_PROTECTED_RUNTIME_DB", str(protected))
    with pytest.raises(RuntimeError, match="refused to open"):
        pm._profile_memory(db_path=str(protected))
    assert not protected.exists()
    assert not Path(str(protected) + "-wal").exists()
    assert not Path(str(protected) + "-shm").exists()


def test_attach_with_explicit_consent_renames(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Khalifa.png", "Khalifa Bouneb")["person_id"]
    batch = manager.create_batch([image_file("IMG_20260730.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Khalifa B.",
        "action": "attach_existing",
        "existing_person_id": person_id,
        "update_existing_name": True,
        "primary_source_id": identity["primary_source_id"],
    }])
    assert result["results"][0]["renamed_existing"] is True
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory.get_person(person_id)["name"] == "Khalifa B."
    finally:
        memory.close()


def test_batch_retry_does_not_repeat_a_rename(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Khalifa.png", "Khalifa Bouneb")["person_id"]
    batch = manager.create_batch([image_file("IMG_20260731.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    change = [{
        "identity_id": identity["identity_id"],
        "name": "Renamed Once",
        "action": "attach_existing",
        "existing_person_id": person_id,
        "update_existing_name": True,
        "primary_source_id": identity["primary_source_id"],
    }]
    manager.commit(batch["batch_id"], "supervisor", change)
    memory = GlobalMemory(db_path=str(database), media_root=media)
    try:
        memory._conn.execute(
            "UPDATE persons SET name='Manually Corrected' WHERE person_id=?", (person_id,)
        )
    finally:
        memory.close()
    replay = manager.commit(batch["batch_id"], "supervisor", change)
    assert replay["results"][0]["idempotent_replay"] is True
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory.get_person(person_id)["name"] == "Manually Corrected"
    finally:
        memory.close()


# --- idempotency -------------------------------------------------------------

def test_fingerprint_is_content_addressed_not_batch_addressed():
    left = commit_fingerprint(action="create_new", scope="new", content_hashes=["b", "a"], name_component="X")
    right = commit_fingerprint(action="create_new", scope="new", content_hashes=["a", "b"], name_component="X")
    assert left == right
    assert left != commit_fingerprint(
        action="create_new", scope="new", content_hashes=["a", "b"], name_component="Y"
    )


def test_commit_creates_one_profile_is_idempotent_and_preserves_phone_history(profile_environment):
    build, database, media = profile_environment
    manager = build()
    batch = manager.create_batch(
        [image_file("Hadil_Karous.png"), image_file("Hadil_alt.png")], "supervisor"
    )
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    change = {
        "identity_id": identity["identity_id"],
        "name": "Hadil Karous",
        "action": "create_new",
        "primary_source_id": identity["photos"][1]["source_id"],
    }
    first = manager.commit(batch["batch_id"], "supervisor", [change])
    replay = manager.commit(batch["batch_id"], "supervisor", [change])
    assert first["summary"]["profiles_created"] == 1
    assert replay["results"][0]["idempotent_replay"] is True

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        people = memory.list_all()
        assert len(people) == 1 and people[0]["name"] == "Hadil Karous"
        rows = memory._conn.execute(
            "SELECT source_id, face_crop_path, is_primary FROM face_photo_sources"
        ).fetchall()
        assert len(rows) == 2 and sum(row["is_primary"] for row in rows) == 1
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0] == 2
        assert memory._conn.execute("SELECT COUNT(*) FROM profile_import_commits").fetchone()[0] == 1
    finally:
        memory.close()


def test_idempotency_survives_coordinator_restart_and_reupload(profile_environment):
    build, database, media = profile_environment
    first_manager = build()
    person_id = commit_one(first_manager, "supervisor", "Zed.png", "Zed")["person_id"]

    def counts():
        memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
        try:
            return {
                "persons": len(memory.list_all()),
                "face_photo_sources": memory._conn.execute("SELECT COUNT(*) FROM face_photo_sources").fetchone()[0],
                "identity_evidence": memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0],
                "person_gallery": memory._conn.execute("SELECT COUNT(*) FROM person_gallery").fetchone()[0],
                "recognition_log": memory._conn.execute("SELECT COUNT(*) FROM recognition_log").fetchone()[0],
                "embedding_count": memory.get_person(person_id)["embedding_count"],
                "primary": memory._conn.execute(
                    "SELECT COUNT(*) FROM face_photo_sources WHERE is_primary=1"
                ).fetchone()[0],
            }
        finally:
            memory.close()

    before = counts()
    first_manager.close()

    # Brand-new coordinator: no in-memory batch state survives.
    second_manager = build()
    batch = second_manager.create_batch([image_file("Zed.png")], "supervisor")
    ready = second_manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    result = second_manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Zed changed filename proposal",
        "action": "create_new",
        "primary_source_id": identity["primary_source_id"],
    }])
    assert result["results"][0]["person_id"] == person_id
    assert result["results"][0]["idempotent_replay"] is True
    assert counts() == before

    conflict_batch = second_manager.create_batch([image_file("Zed.png")], "supervisor")
    conflict_ready = second_manager.wait_ready(conflict_batch["batch_id"], "supervisor")
    conflict_identity = conflict_ready["identities"][0]
    with pytest.raises(pm.OperationConflict, match="operation_conflict"):
        second_manager.commit(conflict_batch["batch_id"], "supervisor", [{
            "identity_id": conflict_identity["identity_id"],
            "action": "attach_existing",
            "existing_person_id": person_id,
        }])
    assert counts() == before


def test_concurrent_duplicate_commit_returns_the_same_durable_result(profile_environment):
    build, database, media = profile_environment
    manager = build()
    batch = manager.create_batch([image_file("Conc.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    change = [{
        "identity_id": identity["identity_id"],
        "name": "Conc",
        "action": "create_new",
        "primary_source_id": identity["primary_source_id"],
    }]
    outcomes: list[dict] = []
    failures: list[str] = []
    barrier = threading.Barrier(6)

    def run():
        barrier.wait()
        try:
            outcomes.append(manager.commit(batch["batch_id"], "supervisor", change))
        except Exception as exc:  # pragma: no cover - must not happen
            failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures
    assert len(outcomes) == 6
    person_ids = {row["results"][0]["person_id"] for row in outcomes}
    assert len(person_ids) == 1
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert len(memory.list_all()) == 1
        assert memory._conn.execute("SELECT COUNT(*) FROM face_photo_sources").fetchone()[0] == 1
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_evidence").fetchone()[0] == 1
        assert memory._conn.execute("SELECT COUNT(*) FROM profile_import_commits").fetchone()[0] == 1
    finally:
        memory.close()


# --- identity policy ---------------------------------------------------------

def test_strong_uncertain_and_new_profile_proposals(profile_environment):
    build, database, media = profile_environment
    seed = build()
    commit_one(seed, "supervisor", "Existing.png", "Existing")

    strong = build()
    ready = strong.wait_ready(
        strong.create_batch([image_file("same.png")], "supervisor")["batch_id"], "supervisor"
    )
    assert ready["identities"][0]["proposed_action"] == "attach_existing"
    assert ready["identities"][0]["existing_candidate"]["name"] == "Existing"

    uncertain = build(TaggedFaceEngine(default=mixed(math.acos(0.72))))
    ready = uncertain.wait_ready(
        uncertain.create_batch([image_file("uncertain.png")], "supervisor")["batch_id"], "supervisor"
    )
    assert ready["identities"][0]["proposed_action"] == "review_required"
    assert ready["identities"][0]["memory_match"]["reason"] == "similarity_between_thresholds"

    fresh = build(TaggedFaceEngine(default=unit(2)))
    ready = fresh.wait_ready(
        fresh.create_batch([image_file("new.png")], "supervisor")["batch_id"], "supervisor"
    )
    assert ready["identities"][0]["proposed_action"] == "create_new"

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert len(memory.list_all()) == 1
    finally:
        memory.close()


def test_observation_policy_is_explicit_not_falsified(profile_environment, monkeypatch):
    """The true observation count is reported; only the documented trusted
    phone-enrolment rule may relax the observation gate."""
    build, _database, _media = profile_environment
    seed = build()
    commit_one(seed, "supervisor", "Base.png", "Base")

    trusted = build()
    ready = trusted.wait_ready(
        trusted.create_batch([image_file("one.png")], "supervisor")["batch_id"], "supervisor"
    )
    match = ready["identities"][0]["memory_match"]
    assert match["observation_count"] == 1
    assert match["minimum_face_observations"] == 3
    assert match["policy_decision"] == "review_required"
    assert match["reason"] == "insufficient_face_observations"
    assert match["trusted_enrolment_applied"] is True
    assert match["identity_source"] == "phone_supervised"
    assert ready["identities"][0]["proposed_action"] == "attach_existing"

    monkeypatch.setenv("PROFILE_IMPORT_TRUSTED_PHONE_ENROLMENT", "0")
    untrusted = build()
    assert untrusted.trusted_phone_enrolment is False
    ready = untrusted.wait_ready(
        untrusted.create_batch([image_file("one.png")], "supervisor")["batch_id"], "supervisor"
    )
    match = ready["identities"][0]["memory_match"]
    assert match["trusted_enrolment_applied"] is False
    assert ready["identities"][0]["proposed_action"] == "review_required"


def test_trusted_enrolment_never_overrides_similarity_or_margin(profile_environment):
    build, _database, _media = profile_environment
    seed = build()
    commit_one(seed, "supervisor", "Base.png", "Base")
    borderline = build(TaggedFaceEngine(default=mixed(math.acos(0.72))))
    ready = borderline.wait_ready(
        borderline.create_batch([image_file("mid.png")], "supervisor")["batch_id"], "supervisor"
    )
    match = ready["identities"][0]["memory_match"]
    assert match["reason"] == "similarity_between_thresholds"
    assert match["trusted_enrolment_applied"] is False
    assert ready["identities"][0]["proposed_action"] == "review_required"


# --- durable review ----------------------------------------------------------

def test_review_required_is_durable_and_not_duplicated(profile_environment):
    build, database, media = profile_environment
    seed = build()
    candidate = commit_one(seed, "supervisor", "Base.png", "Base")["person_id"]

    manager = build(TaggedFaceEngine(default=mixed(math.acos(0.72))))
    batch = manager.create_batch([image_file("Unclear.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    assert identity["proposed_action"] == "review_required"
    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Unclear Person",
        "action": "review_required",
    }])
    row = result["results"][0]
    assert row["status"] == "review_required"
    review_key = row["review_key"]

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert len(memory.list_all()) == 1, "review must not create a synthetic person"
        stored = memory._conn.execute(
            "SELECT * FROM profile_import_reviews WHERE review_key=?", (review_key,)
        ).fetchone()
        assert stored is not None
        assert stored["status"] == "pending"
        assert stored["candidate_person_id"] == candidate
        evidence = json.loads(stored["evidence_json"])
        assert len(evidence) == 1
        assert (media / evidence[0]["face_crop_path"]).is_file()
        assert memory._conn.execute(
            "SELECT outcome FROM profile_import_commits WHERE commit_key=?", (review_key,)
        ).fetchone()["outcome"] == "review"
    finally:
        memory.close()

    # Survives coordinator eviction and restart, and a retry does not duplicate.
    manager.close()
    restarted = build(TaggedFaceEngine(default=mixed(math.acos(0.72))))
    retry_batch = restarted.create_batch([image_file("Unclear.png")], "supervisor")
    retry_ready = restarted.wait_ready(retry_batch["batch_id"], "supervisor")
    retry = restarted.commit(retry_batch["batch_id"], "supervisor", [{
        "identity_id": retry_ready["identities"][0]["identity_id"],
        "name": "Unclear Person",
        "action": "review_required",
    }])
    assert retry["results"][0]["idempotent_replay"] is True

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory._conn.execute("SELECT COUNT(*) FROM profile_import_reviews").fetchone()[0] == 1
        assert len(memory.list_all()) == 1
    finally:
        memory.close()

    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=restarted)
    client = app.test_client()
    listed = client.get("/api/profiles/reviews").get_json()["reviews"]
    assert len(listed) == 1 and listed[0]["review_key"] == review_key
    detail = client.get(f"/api/profiles/{candidate}").get_json()
    imports = [row for row in detail["pending_review_suggestions"] if row["kind"] == "phone_import"]
    assert len(imports) == 1


# --- profile-image priority --------------------------------------------------

def register_video_person(database, media, sharpness=100000.0):
    relative = "person_001/face_crops/video_best.jpg"
    path = media / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((100, 100, 3), 128, np.uint8))
    memory = GlobalMemory(db_path=str(database), media_root=media)
    try:
        return memory.register({
            "face_embedding": unit(0).tolist(),
            "face_crops": [relative],
            "face_crop_sharpness": {relative: sharpness},
            "appearance": {"date": "2026-01-01"},
            "video_sources": ["cam1.mp4"],
            "cameras": ["cam1"],
        })
    finally:
        memory.close()


def resolved_for(database, media, person_id):
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        return resolve_profile_image(memory._conn, person_id, media)
    finally:
        memory.close()


def attach_photo(manager, person_id, filename):
    batch = manager.create_batch([image_file(filename)], "supervisor", kind="profile_photo")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]
    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "action": "attach_existing",
        "existing_person_id": person_id,
        "primary_source_id": identity["primary_source_id"],
    }])
    assert result["results"][0]["status"] == "updated", result["results"][0]
    return result["results"][0]


def test_supervisor_primary_selection_survives_later_attachments(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = register_video_person(database, media)
    assert resolved_for(database, media, person_id)["origin"] == "video_crop"

    attach_photo(manager, person_id, "phone_first.png")
    time.sleep(1.1)
    assert resolved_for(database, media, person_id)["origin"] == "newest_phone_crop"

    attach_photo(manager, person_id, "phone_second.png")
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    detail = client.get(f"/api/profiles/{person_id}").get_json()
    older = min(detail["phone_photos"], key=lambda row: (row["created_at"], row["source_filename"]))
    assert client.post(
        f"/api/profiles/{person_id}/primary-photo", json={"source_id": older["source_id"]}
    ).status_code == 200
    chosen = resolved_for(database, media, person_id)
    assert chosen["origin"] == "supervisor_phone_crop"
    assert chosen["path"] == older["face_crop_path"]

    time.sleep(1.1)
    outcome = attach_photo(manager, person_id, "phone_third.png")
    assert outcome["primary_preserved"] is True
    final = resolved_for(database, media, person_id)
    assert final["origin"] == "supervisor_phone_crop"
    assert final["path"] == older["face_crop_path"]

    detail = client.get(f"/api/profiles/{person_id}").get_json()
    assert detail["effective_profile_image"] == older["face_crop_path"]
    assert detail["profile_image_origin"] == "supervisor_phone_crop"
    assert len(detail["phone_photos"]) == 3
    assert sum(row["is_supervisor_selected"] for row in detail["phone_photos"]) == 1


def test_phone_crop_outranks_a_sharper_video_crop(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = register_video_person(database, media, sharpness=1_000_000.0)
    attach_photo(manager, person_id, "phone.png")
    resolved = resolved_for(database, media, person_id)
    assert resolved["origin"] == "newest_phone_crop"
    assert "video_best" not in resolved["path"]


def test_missing_phone_files_fall_back_to_video_then_placeholder(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = register_video_person(database, media)
    attach_photo(manager, person_id, "phone_a.png")
    time.sleep(1.1)
    attach_photo(manager, person_id, "phone_b.png")

    newest = resolved_for(database, media, person_id)
    assert newest["origin"] == "newest_phone_crop"
    (media / newest["path"]).unlink()

    second = resolved_for(database, media, person_id)
    assert second["origin"] == "newest_phone_crop"
    assert second["path"] != newest["path"]
    (media / second["path"]).unlink()

    video = resolved_for(database, media, person_id)
    assert video["origin"] == "video_crop"
    (media / video["path"]).unlink()

    assert resolved_for(database, media, person_id) == {"path": None, "origin": "placeholder"}


def test_archive_and_restore_preserve_the_resolved_image(profile_environment):
    build, database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Keep.png", "Keep")["person_id"]
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    before = resolved_for(database, media, person_id)
    assert client.post(f"/api/profiles/{person_id}/archive").get_json()["state"] == "archived"
    assert resolved_for(database, media, person_id) == before
    assert client.post(f"/api/profiles/{person_id}/restore").get_json()["state"] == "active"
    assert resolved_for(database, media, person_id) == before


# --- merge -------------------------------------------------------------------

def build_merge_pair(build, database, media):
    left = build()
    right = build(TaggedFaceEngine(default=unit(5)))
    source = commit_one(left, "supervisor", "Src.png", "Src")["person_id"]
    target = commit_one(right, "supervisor", "Tgt.png", "Tgt")["person_id"]
    return left, source, target


def merge_snapshot(database, media, source, target):
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        person = memory.get_person(source)
        return {
            "source_active": person["is_active"],
            "merged_into": person["merged_into_person_id"],
            "phone": {
                person_id: memory._conn.execute(
                    "SELECT COUNT(*) FROM face_photo_sources WHERE person_id=?", (person_id,)
                ).fetchone()[0]
                for person_id in (source, target)
            },
            "evidence": {
                person_id: memory._conn.execute(
                    "SELECT COUNT(*) FROM identity_evidence WHERE person_id=?", (person_id,)
                ).fetchone()[0]
                for person_id in (source, target)
            },
            "audit": memory._conn.execute(
                "SELECT COUNT(*) FROM identity_merge_audit"
            ).fetchone()[0],
        }
    finally:
        memory.close()


@pytest.mark.parametrize("stage", [
    "before_evidence_movement",
    "phone_source_movement",
    "primary_reconciliation",
    "audit_creation",
])
def test_merge_rolls_back_completely_on_injected_failure(profile_environment, monkeypatch, stage):
    build, database, media = profile_environment
    manager, source, target = build_merge_pair(build, database, media)
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    before = merge_snapshot(database, media, source, target)
    assert before["audit"] == 0

    targets = {
        "before_evidence_movement": (pm, "merge_phone_evidence"),
        "phone_source_movement": (pm, "_merge_move_phone_sources"),
        "primary_reconciliation": (pm, "_merge_reconcile_primary"),
        "audit_creation": (global_memory_store.GlobalMemory, "_insert_person_merge_audit"),
    }
    owner, attribute = targets[stage]
    original = getattr(owner, attribute)
    armed = {"value": True}

    # Fires exactly once, so the retry exercises the real code path.  Note this
    # deliberately avoids monkeypatch.undo(), which would also revert the
    # fixture's redirection away from the real runtime database.
    def boom(*args, **kwargs):
        if armed["value"]:
            armed["value"] = False
            raise sqlite3.IntegrityError(f"injected failure at {stage}")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, boom)

    response = client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    })
    assert response.status_code == 409
    assert armed["value"] is False
    assert merge_snapshot(database, media, source, target) == before

    retry = client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    })
    assert retry.status_code == 200, retry.get_json()
    after = merge_snapshot(database, media, source, target)
    assert after["source_active"] is False
    assert after["merged_into"] == target
    assert after["phone"][source] == 0 and after["phone"][target] == 2
    assert after["evidence"][source] == 0 and after["evidence"][target] == 2
    assert after["audit"] == 1


def test_merge_rejects_self_and_unconfirmed_requests(profile_environment):
    build, database, media = profile_environment
    manager, source, target = build_merge_pair(build, database, media)
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    assert client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": source, "confirm": True,
    }).status_code == 409
    assert client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target,
    }).status_code == 400
    assert client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    }).status_code == 200
    assert client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    }).status_code in {200, 409}


def test_merge_preserves_a_supervisor_selected_target_photo(profile_environment):
    build, database, media = profile_environment
    manager, source, target = build_merge_pair(build, database, media)
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    target_photo = client.get(f"/api/profiles/{target}").get_json()["phone_photos"][0]
    client.post(f"/api/profiles/{target}/primary-photo", json={"source_id": target_photo["source_id"]})
    assert client.post("/api/profiles/merge", json={
        "source_person_id": source, "target_person_id": target, "confirm": True,
    }).status_code == 200
    resolved = resolved_for(database, media, target)
    assert resolved["origin"] == "supervisor_phone_crop"
    assert resolved["path"] == target_photo["face_crop_path"]


# --- profile API -------------------------------------------------------------

def test_profile_api_listing_search_edit_and_photo_history(profile_environment):
    build, _database, _media = profile_environment
    manager = build()
    right = build(TaggedFaceEngine(default=unit(5)))
    source = commit_one(manager, "127.0.0.1", "Source.png", "Source")["person_id"]
    target = commit_one(right, "127.0.0.1", "Target.png", "Target")["person_id"]

    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()

    listed = client.get("/api/profiles?state=all")
    assert listed.status_code == 200
    assert {row["person_id"] for row in listed.get_json()["profiles"]} == {source, target}
    searched = client.get("/api/profiles?state=all&q=source").get_json()["profiles"]
    assert [row["person_id"] for row in searched] == [source]

    edited = client.patch(f"/api/profiles/{source}", json={"name": "Renamed", "notes": "Note"})
    assert edited.status_code == 200
    assert edited.get_json()["notes"] == "Note"
    assert "embedding" not in edited.get_json()
    assert client.patch(f"/api/profiles/{source}", json={"nickname": "x"}).status_code == 400

    detail = client.get(f"/api/profiles/{source}").get_json()
    assert len(detail["phone_photos"]) == 1
    assert detail["video_evidence"] == []
    assert detail["effective_profile_image"] == detail["phone_photos"][0]["face_crop_path"]

    client.post(f"/api/profiles/{source}/archive")
    assert [row["person_id"] for row in client.get("/api/profiles?state=archived").get_json()["profiles"]] == [source]
    assert [row["person_id"] for row in client.get("/api/profiles?state=active").get_json()["profiles"]] == [target]
    client.post(f"/api/profiles/{source}/restore")
    assert client.get("/api/profiles?state=bogus").status_code == 400


def test_api_exposes_no_embeddings_or_absolute_paths(profile_environment):
    build, _database, media = profile_environment
    manager = build()
    person_id = commit_one(manager, "supervisor", "Leak.png", "Leak")["person_id"]
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    preview = manager.wait_ready(
        manager.create_batch([image_file("P.png")], "supervisor")["batch_id"], "supervisor"
    )
    payloads = {
        "preview": json.dumps(preview),
        "list": client.get("/api/profiles?state=all").get_data(as_text=True),
        "detail": client.get(f"/api/profiles/{person_id}").get_data(as_text=True),
        "reviews": client.get("/api/profiles/reviews").get_data(as_text=True),
    }
    for label, text in payloads.items():
        assert str(media) not in text, f"{label} leaked the media root"
        assert "_embedding" not in text, f"{label} leaked a raw embedding"
    detail = client.get(f"/api/profiles/{person_id}").get_json()
    assert "embedding" not in detail
    assert "embedding" not in detail["phone_photos"][0]


# --- staging lifecycle -------------------------------------------------------

def test_cancel_removes_staged_media(profile_environment):
    build, _database, media = profile_environment
    manager = build()
    batch = manager.create_batch([image_file("c1.png"), image_file("c2.png")], "supervisor")
    manager.wait_ready(batch["batch_id"], "supervisor")
    directory = media / "_profile_imports" / batch["batch_id"]
    assert list(directory.rglob("*.png"))
    cancelled = manager.cancel(batch["batch_id"], "supervisor")
    assert cancelled["state"] == "cancelled"
    assert not directory.exists()
    with pytest.raises(KeyError):
        manager.public_batch(batch["batch_id"], "supervisor")


def test_ttl_sweep_reclaims_terminal_batches(profile_environment, monkeypatch):
    build, _database, media = profile_environment
    monkeypatch.setenv("PROFILE_IMPORT_BATCH_TTL_SECONDS", "60")
    manager = build()
    batch = manager.create_batch([image_file("t.png")], "supervisor")
    manager.wait_ready(batch["batch_id"], "supervisor")
    directory = media / "_profile_imports" / batch["batch_id"]
    assert directory.is_dir()
    assert manager.sweep() == 0

    manager._batches[batch["batch_id"]]["monotonic_created_at"] -= 10_000
    assert manager.sweep() == 1
    assert not directory.exists()
    assert batch["batch_id"] not in manager._batches


def test_ready_batches_do_not_permanently_consume_capacity(profile_environment, monkeypatch):
    build, _database, media = profile_environment
    monkeypatch.setenv("PROFILE_IMPORT_MAX_BATCHES", "3")
    manager = build()
    created = []
    for index in range(6):
        batch = manager.create_batch([image_file(f"r{index}.png")], "supervisor")
        manager.wait_ready(batch["batch_id"], "supervisor")
        created.append(batch["batch_id"])
    assert len(manager._batches) <= 3
    staged = list((media / "_profile_imports").iterdir())
    assert len(staged) <= 3
    assert created[-1] in manager._batches


def test_failed_manual_preview_leaves_no_staged_media(profile_environment):
    build, _database, media = profile_environment
    manager = build()
    from forensics.person_creation.service import app
    app.config.update(TESTING=True, PROFILE_IMPORT_MANAGER=manager)
    client = app.test_client()
    response = client.post("/api/profiles/manual/preview", data={
        "name": "  ",
        "photo": (io.BytesIO(image_file("m.png").stream.read()), "m.png"),
    }, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "name is required"
    staging = media / "_profile_imports"
    assert not staging.exists() or not list(staging.iterdir())
    assert manager._batches == {}


# --- write-path security -----------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="requires real POSIX symlinks")
def test_symlinked_person_directory_cannot_receive_writes(profile_environment):
    build, database, media = profile_environment
    manager = build()
    outside = media.parent / "OUTSIDE"
    outside.mkdir()
    batch = manager.create_batch([image_file("s.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]

    media.mkdir(parents=True, exist_ok=True)
    (media / "person_001").symlink_to(outside, target_is_directory=True)

    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Symlinked",
        "action": "create_new",
        "primary_source_id": identity["primary_source_id"],
    }])
    assert result["results"][0]["status"] == "failed"
    assert "symlink" in result["results"][0]["error"]
    assert [path for path in outside.rglob("*") if path.is_file()] == []

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory.list_all(include_inactive=True) == []
        assert memory._conn.execute("SELECT COUNT(*) FROM face_photo_sources").fetchone()[0] == 0
    finally:
        memory.close()


@pytest.mark.skipif(os.name == "nt", reason="requires real POSIX symlinks")
def test_image_api_refuses_symlink_and_traversal_reads(profile_environment):
    build, _database, media = profile_environment
    build()
    media.mkdir(parents=True, exist_ok=True)
    secret = media.parent / "secret.jpg"
    cv2.imwrite(str(secret), np.full((10, 10, 3), 7, np.uint8))
    (media / "leak.jpg").symlink_to(secret)
    from forensics.person_creation.service import app
    app.config.update(TESTING=True)
    client = app.test_client()
    for probe in ("leak.jpg", "../secret.jpg", "../../etc/passwd", "/etc/passwd",
                  "person_001/../../secret.jpg", "C:/Windows/win.ini", "file:///etc/passwd"):
        assert client.get("/api/images", query_string={"path": probe}).status_code != 200


def test_rollback_removes_copied_media(profile_environment, monkeypatch):
    build, database, media = profile_environment
    manager = build()
    batch = manager.create_batch([image_file("rb.png")], "supervisor")
    ready = manager.wait_ready(batch["batch_id"], "supervisor")
    identity = ready["identities"][0]

    original_execute = pm.ProfileImportManager._materialize_phone_files

    def fail_after_copy(self, person_id, items, created):
        result = original_execute(self, person_id, items, created)
        assert [path for path in created if path.is_file()]
        raise sqlite3.IntegrityError("injected failure after media copy")

    monkeypatch.setattr(pm.ProfileImportManager, "_materialize_phone_files", fail_after_copy)
    result = manager.commit(batch["batch_id"], "supervisor", [{
        "identity_id": identity["identity_id"],
        "name": "Rollback",
        "action": "create_new",
        "primary_source_id": identity["primary_source_id"],
    }])
    assert result["results"][0]["status"] == "failed"
    monkeypatch.undo()

    leftovers = list((media / "person_001").rglob("*")) if (media / "person_001").exists() else []
    assert [path for path in leftovers if path.is_file()] == []
    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        assert memory.list_all(include_inactive=True) == []
    finally:
        memory.close()


# --- migration ---------------------------------------------------------------

LEGACY_SCHEMA = """
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_count INTEGER NOT NULL DEFAULT 1,
    enrolled_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    cameras TEXT NOT NULL DEFAULT '[]',
    profile_image TEXT DEFAULT NULL,
    profile_image_source TEXT NOT NULL DEFAULT 'auto'
);
CREATE TABLE counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
CREATE TABLE appearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    date TEXT NOT NULL,
    top TEXT,
    bottom TEXT,
    shoes TEXT,
    full_description TEXT,
    top_color TEXT,
    bottom_color TEXT,
    best_body_crops TEXT NOT NULL DEFAULT '[]',
    video_sources TEXT NOT NULL DEFAULT '[]',
    UNIQUE(person_id, date)
);
CREATE TABLE person_gallery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    crop_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sharpness REAL NOT NULL DEFAULT 0,
    session_date TEXT,
    video_source TEXT,
    width INTEGER,
    height INTEGER,
    UNIQUE(person_id, crop_type, path)
);
CREATE TABLE recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    similarity REAL,
    embedding_count_before INTEGER,
    embedding_count_after INTEGER,
    video_sources TEXT NOT NULL DEFAULT '[]',
    ts TEXT NOT NULL
);
"""


def test_migration_upgrades_an_existing_database_idempotently(tmp_path, monkeypatch):
    database = tmp_path / "legacy.db"
    media = tmp_path / "media"
    connection = sqlite3.connect(str(database))
    connection.executescript(LEGACY_SCHEMA)
    connection.execute(
        "INSERT INTO persons(person_id, name, embedding, embedding_count, enrolled_at, updated_at) "
        "VALUES ('person_001', 'Khalifa', ?, 168, '2026-01-01', '2026-01-02')",
        (unit(0).tobytes(),),
    )
    connection.execute("INSERT INTO counters(key, value) VALUES ('person_count', 1)")
    connection.execute("INSERT INTO appearances(person_id, date) VALUES ('person_001', '2026-01-01')")
    connection.execute(
        "INSERT INTO person_gallery(person_id, crop_type, path, sharpness) "
        "VALUES ('person_001', 'face', 'person_001/face_crops/a.jpg', 12.5)"
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr("forensics.global_memory.config.DB_PATH", str(database))
    for _pass in range(3):
        global_memory_store._INITIALIZED_DATABASES.clear()
        GlobalMemory(db_path=str(database), media_root=media).close()

    memory = GlobalMemory(db_path=str(database), read_only=True, media_root=media)
    try:
        person = memory.get_person("person_001")
        assert person["name"] == "Khalifa"
        assert person["embedding_count"] == 168
        assert person["notes"] == ""
        assert person["identity_source"] == "video"
        assert person["is_active"] is True
        columns = {row["name"] for row in memory._conn.execute("PRAGMA table_info(face_photo_sources)")}
        assert {"is_supervisor_selected", "content_sha256"} <= columns
        commit_columns = {row["name"] for row in memory._conn.execute("PRAGMA table_info(profile_import_commits)")}
        assert {"result_json", "content_set_key"} <= commit_columns
        review_columns = {
            row["name"]
            for row in memory._conn.execute("PRAGMA table_info(profile_import_reviews)")
        }
        assert {
            "content_set_key", "embedding", "resolution_action",
            "resolution_target_person_id", "resolution_result_json",
        } <= review_columns
        tables = {row["name"] for row in memory._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "face_photo_sources", "profile_import_commits",
            "profile_import_content_sets", "profile_import_reviews",
            "profile_import_review_evidence",
        } <= tables
        assert memory._conn.execute("SELECT COUNT(*) FROM appearances").fetchone()[0] == 1
        assert memory._conn.execute("SELECT COUNT(*) FROM person_gallery").fetchone()[0] == 1
    finally:
        memory.close()


def test_one_supervisor_primary_and_one_primary_per_person(tmp_path, monkeypatch):
    database = tmp_path / "memory.db"
    media = tmp_path / "media"
    monkeypatch.setattr("forensics.global_memory.config.DB_PATH", str(database))
    memory = GlobalMemory(db_path=str(database), media_root=media)
    try:
        memory._conn.execute(
            "INSERT INTO persons(person_id, name, embedding, embedding_count, enrolled_at, updated_at) "
            "VALUES ('person_001', 'P', ?, 1, '2026-01-01', '2026-01-01')",
            (unit(0).tobytes(),),
        )

        def insert(source_id, primary, supervisor):
            memory._conn.execute(
                "INSERT INTO face_photo_sources(source_id, person_id, original_image_path, "
                "face_crop_path, face_bbox, quality_info, embedding, created_at, source_filename, "
                "import_batch_id, is_primary, is_supervisor_selected) "
                "VALUES (?, 'person_001', 'a.jpg', 'b.jpg', '[]', '{}', ?, '2026-01-01', 'f.png', NULL, ?, ?)",
                (source_id, b"\x00" * 4, primary, supervisor),
            )

        insert("s1", 1, 1)
        with pytest.raises(sqlite3.IntegrityError):
            insert("s2", 1, 0)
        with pytest.raises(sqlite3.IntegrityError):
            insert("s3", 0, 1)
        insert("s4", 0, 0)
        assert memory._conn.execute("SELECT COUNT(*) FROM face_photo_sources").fetchone()[0] == 2
    finally:
        memory.close()
