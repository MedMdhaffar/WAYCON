"""Phase 4 step 1: continuous Phase 3E decisions taken while capture is live."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from forensics.global_memory import GlobalMemory
from forensics.identity_evidence import identity_evidence_key
from forensics.media_paths import MediaPathError
from forensics.person_creation.live_analysis import LiveRollingAnalysisSession
from forensics.person_creation.live_session import FrozenAnalysisSnapshot, FrozenRecord


def _record(value: dict) -> FrozenRecord:
    return FrozenRecord.from_mapping(value)


@pytest.fixture
def media_root(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    root.mkdir()
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(tmp_path / "memory.db"))
    return root


@pytest.fixture
def database(tmp_path):
    return tmp_path / "memory.db"


def _face_paths(media_root: Path, count: int, *, prefix: str = "live") -> tuple[str, ...]:
    directory = media_root / "session" / "_staging" / "face_crops"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        crop = directory / f"{prefix}_{index}.jpg"
        crop.write_bytes(f"jpeg:{prefix}:{index}".encode())
        paths.append(str(crop))
    return tuple(paths)


def _snapshot(version: int, paths: tuple[str, ...], *, chunk: int = 0):
    embeddings = tuple(
        _record({
            "crop_path": path,
            "embedding": [1.0, float(index) / 1000.0],
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
        person_name="Live Subject",
        video_paths=("camera-source",),
        identity_clustering_config=_record({"eps": 0.4, "min_samples": 2}),
        quality_body_crops=(),
        quality_face_crops=faces,
        face_embeddings=embeddings,
        face_chunk_membership=tuple((path, chunk) for path in paths),
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


def _session(database: Path, provider, **kwargs) -> LiveRollingAnalysisSession:
    return LiveRollingAnalysisSession(
        snapshot_provider=provider,
        join_timeout_seconds=5.0,
        database_path=database,
        cluster=_cluster_all,
        associate=_association,
        job_id=kwargs.pop("job_id", "job-live-1"),
        identity_decisions=True,
        **kwargs,
    )


def _wait_for(session: LiveRollingAnalysisSession, version: int) -> dict:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = session.public_snapshot()
        if snapshot["analysis_version"] >= version:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"analysis version {version} was not completed")


def _seed_person(database: Path, embedding: list[float], paths: tuple[str, ...]) -> str:
    memory = GlobalMemory(str(database))
    try:
        result = memory.register_with_identity_policy(
            {
                "name": "Seed",
                "face_embedding": embedding,
                "face_crops": list(paths),
                "body_crops": [],
                "best_body_crops": [],
                "video_sources": ["seed"],
                "appearance": {"date": "2026-01-01"},
            },
            observation_count=8,
            low_confidence=False,
        )
        return result.person_id
    finally:
        memory.close()


def _count(database: Path, table: str) -> int:
    if not database.exists():
        return 0
    connection = sqlite3.connect(str(database))
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.OperationalError:
        return 0  # schema is only created on the first write
    finally:
        connection.close()


def _person_count(database: Path) -> int:
    return _count(database, "persons")


def _suggestion_count(database: Path) -> int:
    return _count(database, "identity_match_suggestions")


def _embedding(database: Path, person_id: str) -> tuple[np.ndarray, int]:
    connection = sqlite3.connect(str(database))
    try:
        blob, count = connection.execute(
            "SELECT embedding, embedding_count FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        return np.frombuffer(blob, dtype=np.float32).copy(), int(count)
    finally:
        connection.close()


def _canonical_crop(
    media_root: Path,
    person_id: str,
    crop_type: str,
    name: str,
    content: bytes,
) -> str:
    path = media_root / person_id / f"{crop_type}_crops" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return f"{person_id}/{crop_type}_crops/{name}"


def _run(session, provider_state, versions):
    session.start()
    try:
        for version in versions:
            provider_state["snapshot"] = version[1]
            session.request_version(version[0])
            _wait_for(session, version[0])
        return session.public_snapshot()
    finally:
        session.finish(versions[-1][0])


def _identity(snapshot: dict) -> dict:
    identities = snapshot["live_identities"]
    assert identities, "expected one live identity"
    return identities[0]


# --- live rolling identity decision -------------------------------------------


def test_live_rolling_identity_decision_publishes_phase3e_fields(media_root, database):
    paths = _face_paths(media_root, 6)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    snapshot = _run(session, state, [(1, _snapshot(1, paths))])
    identity = _identity(snapshot)

    assert identity["live_identity_id"] == identity["session_person_id"]
    assert identity["decision"] == "new_person"
    assert identity["canonical_person_id"].startswith("person_")
    assert identity["state"] == "new_person"
    assert identity["version"] == 1
    assert identity["decision_version"] == 1
    assert identity["face_count"] == 6
    assert identity["body_count"] == 0
    assert identity["best_face_path"]
    assert identity["first_seen"] == 0 and identity["last_seen"] == 0
    assert identity["suggestion_id"] is None
    for field in (
        "candidate_person_id", "candidate_similarity",
        "second_candidate_person_id", "second_candidate_similarity", "margin",
    ):
        assert field in identity


def test_decision_is_withheld_until_minimum_observations(media_root, database):
    paths = _face_paths(media_root, 2)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    snapshot = _run(session, state, [(1, _snapshot(1, paths))])
    identity = _identity(snapshot)

    assert identity["decision"] is None
    assert identity["state"] == "observing"
    assert identity["canonical_person_id"] is None
    assert _person_count(database) == 0


# --- idempotency ---------------------------------------------------------------


def test_new_person_creates_one_canonical_identity_only(media_root, database):
    paths = _face_paths(media_root, 3)
    # Evidence accumulates: the second pass is a superset of the first, so the
    # stable live identity is retained and must not be registered twice.
    grown = paths + _face_paths(media_root, 2, prefix="grow")
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    snapshot = _run(session, state, [
        (1, _snapshot(1, paths)),
        (2, _snapshot(2, grown)),
    ])
    identity = _identity(snapshot)

    assert identity["decision"] == "new_person"
    assert _person_count(database) == 1
    assert _suggestion_count(database) == 0


def test_attach_existing_is_idempotent(media_root, database):
    seed_paths = _face_paths(media_root, 3, prefix="seed")
    seeded = _seed_person(database, [1.0, 0.0], seed_paths)
    paths = _face_paths(media_root, 3)
    grown = paths + _face_paths(media_root, 2, prefix="grow")
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    snapshot = _run(session, state, [
        (1, _snapshot(1, paths)),
        (2, _snapshot(2, grown)),
    ])
    identity = _identity(snapshot)

    assert identity["decision"] == "attach_existing"
    assert identity["canonical_person_id"] == seeded
    assert identity["candidate_person_id"] == seeded
    assert _person_count(database) == 1
    assert _suggestion_count(database) == 0


def test_review_required_creates_one_source_and_one_suggestion(media_root, database):
    seed_paths = _face_paths(media_root, 3, prefix="seed")
    seeded = _seed_person(database, [0.75, 0.6614], seed_paths)
    paths = _face_paths(media_root, 3)
    grown = paths + _face_paths(media_root, 2, prefix="grow")
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    snapshot = _run(session, state, [
        (1, _snapshot(1, paths)),
        (2, _snapshot(2, grown)),
    ])
    identity = _identity(snapshot)

    assert identity["decision"] == "review_required"
    assert identity["candidate_person_id"] == seeded
    assert identity["suggestion_id"] is not None
    assert identity["canonical_person_id"] != seeded
    # one seeded person + exactly one live source profile
    assert _person_count(database) == 2
    assert _suggestion_count(database) == 1


def test_identical_snapshot_replay_does_not_advance_stability(media_root, database):
    paths = _face_paths(media_root, 3)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    session.start()
    try:
        for version in (1, 2, 3, 4):
            state["snapshot"] = _snapshot(version, paths)
            session.request_version(version)
            _wait_for(session, version)
        snapshot = session.public_snapshot()
    finally:
        session.finish(4)

    identity = _identity(snapshot)
    assert identity["decision"] == "new_person"
    assert identity["provisional"] is True
    assert identity["decision_version"] == 1
    assert identity["version"] == 1
    assert _person_count(database) == 0
    assert len(session.identity_decisions()) == 0


def test_provisional_outcome_can_change_before_persistence(media_root, database):
    seed_paths = _face_paths(media_root, 3, prefix="seed-change")
    _seed_person(database, [1.0, 0.0], seed_paths)
    initial = _face_paths(media_root, 3, prefix="change")
    grown = initial + _face_paths(media_root, 1, prefix="change-new")
    final = grown + _face_paths(media_root, 1, prefix="change-final")

    def changing_cluster(state):
        result = _cluster_all(state)
        if result["identity_clusters"]:
            result["identity_clusters"][0]["representative_embedding"] = (
                [1.0, 0.0] if len(state["all_face_embeddings"]) == 3 else [0.0, 1.0]
            )
        return result

    state = {"snapshot": _snapshot(1, initial)}
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        cluster=changing_cluster,
        associate=_association,
        job_id="job-changing",
        identity_decisions=True,
    )
    session.start()
    try:
        session.request_version(1)
        first = _identity(_wait_for(session, 1))
        assert first["decision"] == "attach_existing"
        assert first["provisional"] is True

        state["snapshot"] = _snapshot(2, grown)
        session.request_version(2)
        second = _identity(_wait_for(session, 2))
        assert second["decision"] == "new_person"
        assert second["provisional"] is True
        assert _person_count(database) == 1

        state["snapshot"] = _snapshot(3, final)
        session.request_version(3)
        third = _identity(_wait_for(session, 3))
        assert third["decision"] == "new_person"
        assert third["provisional"] is False
        assert _person_count(database) == 2
    finally:
        session.finish(3)


def test_evidence_key_uses_contents_and_survives_relocation(media_root):
    left = media_root / "session-a" / "same.jpg"
    right = media_root / "session-b" / "same.jpg"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"left-content")
    right.write_bytes(b"right-content")

    left_key = identity_evidence_key(left, "face")
    right_key = identity_evidence_key(right, "face")
    assert left_key != right_key
    assert identity_evidence_key(left, "body") != left_key

    relocated = media_root / "person_001" / "face_crops" / "renamed.jpg"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(left.read_bytes())
    assert identity_evidence_key(relocated, "face") == left_key


def test_append_blends_only_unseen_faces_and_replay_is_database_noop(
    media_root,
    database,
):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 3, prefix="append-seed"),
    )
    face_a = _canonical_crop(media_root, seeded, "face", "a.jpg", b"append-a")
    face_b = _canonical_crop(media_root, seeded, "face", "b.jpg", b"append-b")
    keys = [
        identity_evidence_key(path, "face", media_root=media_root)
        for path in (face_a, face_b)
    ]

    memory = GlobalMemory(str(database), media_root=media_root)
    try:
        first = memory.append_identity_evidence(
            seeded,
            embedding=[0.0, 1.0],
            observation_count=2,
            face_crops=[face_a, face_b],
            evidence_keys=keys,
        )
        rows_after_first = {
            table: _count(database, table)
            for table in (
                "identity_evidence", "person_gallery", "appearances", "recognition_log"
            )
        }
        vector_after_first, count_after_first = _embedding(database, seeded)
        replay = memory.append_identity_evidence(
            seeded,
            embedding=[-1.0, 0.0],
            observation_count=2,
            face_crops=[face_a, face_b],
            evidence_keys=keys,
        )
    finally:
        memory.close()

    expected = np.asarray([3.0, 2.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert first.appended is True
    assert count_after_first == 5
    assert np.allclose(vector_after_first, expected, atol=1e-6)
    assert replay.idempotent_replay is True
    assert _embedding(database, seeded)[1] == count_after_first
    assert np.allclose(_embedding(database, seeded)[0], vector_after_first)
    assert {
        table: _count(database, table) for table in rows_after_first
    } == rows_after_first


def test_body_only_append_is_ledgered_without_embedding_change(media_root, database):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 2, prefix="body-seed"),
    )
    body = _canonical_crop(media_root, seeded, "body", "body.jpg", b"body-only")
    key = identity_evidence_key(body, "body", media_root=media_root)
    before_vector, before_count = _embedding(database, seeded)

    with GlobalMemory(str(database), media_root=media_root) as memory:
        result = memory.append_identity_evidence(
            seeded,
            body_crops=[body],
            evidence_keys=[key],
        )

    after_vector, after_count = _embedding(database, seeded)
    assert result.appended is True
    assert after_count == before_count
    assert np.array_equal(after_vector, before_vector)
    connection = sqlite3.connect(str(database))
    try:
        row = connection.execute(
            "SELECT crop_type, embedding_applied, observation_weight "
            "FROM identity_evidence WHERE evidence_key=?",
            (key,),
        ).fetchone()
    finally:
        connection.close()
    assert row == ("body", 0, 0)
    assert _count(database, "person_gallery") == 3


def test_append_failure_rolls_back_ledger_and_identity_updates(
    media_root,
    database,
    monkeypatch,
):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 2, prefix="rollback-seed"),
    )
    face = _canonical_crop(media_root, seeded, "face", "fail.jpg", b"fail-face")
    key = identity_evidence_key(face, "face", media_root=media_root)
    before_vector, before_count = _embedding(database, seeded)
    before_gallery = _count(database, "person_gallery")
    before_log = _count(database, "recognition_log")

    with GlobalMemory(str(database), media_root=media_root) as memory:
        monkeypatch.setattr(
            memory,
            "_update_embedding",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("blend failed")),
        )
        with pytest.raises(RuntimeError, match="blend failed"):
            memory.append_identity_evidence(
                seeded,
                embedding=[0.0, 1.0],
                observation_count=1,
                face_crops=[face],
                evidence_keys=[key],
            )

    after_vector, after_count = _embedding(database, seeded)
    assert after_count == before_count
    assert np.array_equal(after_vector, before_vector)
    assert _count(database, "identity_evidence") == 0
    assert _count(database, "person_gallery") == before_gallery
    assert _count(database, "recognition_log") == before_log


def test_concurrent_append_does_not_double_count(media_root, database):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 2, prefix="concurrent-seed"),
    )
    face = _canonical_crop(media_root, seeded, "face", "one.jpg", b"one-new-face")
    key = identity_evidence_key(face, "face", media_root=media_root)
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def append_once():
        memory = GlobalMemory(str(database), media_root=media_root)
        try:
            barrier.wait(timeout=5)
            results.append(memory.append_identity_evidence(
                seeded,
                embedding=[0.0, 1.0],
                observation_count=1,
                face_crops=[face],
                evidence_keys=[key],
            ))
        except BaseException as exc:
            errors.append(exc)
        finally:
            memory.close()

    threads = [threading.Thread(target=append_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    assert sum(result.appended for result in results) == 1
    assert _embedding(database, seeded)[1] == 3
    assert _count(database, "identity_evidence") == 1


def test_append_rejects_noncanonical_media_paths(media_root, database):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 2, prefix="security-seed"),
    )
    staging = media_root / "session" / "_staging" / "face_crops" / "bad.jpg"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(b"bad")
    with GlobalMemory(str(database), media_root=media_root) as memory:
        with pytest.raises(MediaPathError):
            memory.append_identity_evidence(
                seeded,
                embedding=[0.0, 1.0],
                observation_count=1,
                face_crops=[str(staging)],
            )


def test_old_database_is_upgraded_with_identity_evidence_ledger(
    media_root,
    database,
):
    connection = sqlite3.connect(str(database))
    try:
        connection.execute(
            """
            CREATE TABLE persons (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_count INTEGER NOT NULL DEFAULT 1,
                enrolled_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cameras TEXT NOT NULL DEFAULT '[]',
                profile_image TEXT DEFAULT NULL,
                profile_image_source TEXT NOT NULL DEFAULT 'auto',
                is_active INTEGER NOT NULL DEFAULT 1,
                merged_into_person_id TEXT DEFAULT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    for _ in range(2):
        with GlobalMemory(str(database), media_root=media_root):
            pass

    connection = sqlite3.connect(str(database))
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(identity_evidence)")
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(identity_evidence)")
        }
    finally:
        connection.close()
    assert {
        "id", "person_id", "evidence_key", "crop_type", "canonical_path",
        "embedding_applied", "observation_weight", "created_at",
    } <= columns
    assert "idx_identity_evidence_person" in indexes


# --- canonical media during live persistence -----------------------------------


def _stored_media_values(database: Path) -> list[str]:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    values: list[str] = []
    try:
        for row in connection.execute("SELECT path FROM person_gallery"):
            values.append(str(row["path"] or ""))
        for row in connection.execute("SELECT profile_image FROM persons"):
            values.append(str(row["profile_image"] or ""))
        for row in connection.execute("SELECT best_face_crop FROM recognition_log"):
            values.append(str(row["best_face_crop"] or ""))
        for row in connection.execute("SELECT best_body_crops FROM appearances"):
            values.extend(str(item) for item in json.loads(row["best_body_crops"] or "[]"))
    finally:
        connection.close()
    return [value for value in values if value]


def test_live_persistence_stores_only_canonical_person_paths(media_root, database):
    paths = _face_paths(media_root, 6)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    _run(session, state, [(1, _snapshot(1, paths))])

    stored = _stored_media_values(database)
    assert stored, "live persistence stored no media references"
    for value in stored:
        assert not value.startswith("/"), value
        assert "\\" not in value, value
        assert not re.match(r"^[A-Za-z]:", value), value
        assert "_staging" not in value, value
        assert not re.search(r"cluster_\d", value), value
        assert re.match(r"^person_\d+/", value), value
        # the canonical file really exists under the media root
        assert (media_root / value).is_file(), value


def test_decision_receipt_contains_required_fields(media_root, database):
    paths = _face_paths(media_root, 6)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"], job_id="job-receipt")

    _run(session, state, [(1, _snapshot(1, paths))])
    receipts = session.identity_decisions()

    assert len(receipts) == 1
    receipt = receipts[0]
    for field in (
        "job_id", "live_identity_id", "decision_version", "decision",
        "canonical_person_id", "suggestion_id", "evidence_keys",
        "persisted_evidence_keys", "persisted_observation_count",
        "persisted_face_crops", "persisted_body_crops",
        "canonical_face_paths", "canonical_body_paths",
        "persisted_face_count", "persisted_body_count",
        "last_appended_analysis_version",
    ):
        assert field in receipt, field
    assert receipt["job_id"] == "job-receipt"
    assert receipt["decision_version"] == 1
    assert receipt["persisted_face_count"] == 6
    assert receipt["persisted_observation_count"] == 6
    assert receipt["evidence_keys"] == receipt["persisted_evidence_keys"]
    assert all(key.startswith("face:") for key in receipt["evidence_keys"])
    assert all(
        value.startswith(f"{receipt['canonical_person_id']}/")
        for value in receipt["persisted_face_crops"]
    )


# --- finalize reconciliation ---------------------------------------------------


def test_finalize_reuses_live_receipt_without_creating_duplicates(media_root, database):
    from forensics.person_creation.nodes.finalize import finalize

    paths = _face_paths(media_root, 6)
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])
    _run(session, state, [(1, _snapshot(1, paths))])

    receipt = session.identity_decisions()[0]
    person_id = receipt["canonical_person_id"]
    before = {
        "persons": _person_count(database),
        "gallery": _count(database, "person_gallery"),
        "appearances": _count(database, "appearances"),
        "suggestions": _suggestion_count(database),
        "recognition": _count(database, "recognition_log"),
    }

    # Simulate the batch tail: the same evidence, promoted to cluster paths.
    output_dir = media_root / "session"
    promoted_dir = output_dir / "cluster_0" / "face_crops"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    promoted = []
    for source in paths:
        destination = promoted_dir / Path(source).name
        destination.write_bytes(Path(source).read_bytes())
        promoted.append(str(destination))

    finalize({
        "output_dir": str(output_dir),
        "person_name": "Live Subject",
        "video_paths": ["camera-source"],
        "per_cluster_profiles": {
            0: {
                "id": "cluster_0",
                "name": "Live Subject",
                "face_embedding": [1.0, 0.0],
                "face_crops": promoted,
                "body_crops": [],
                "best_body_crops": [],
                "video_sources": ["camera-source"],
                "appearance": {"date": "2026-07-21"},
            },
        },
        "unresolved_faces": [],
        "unattached_bodies": [],
        "live_identity_decisions": [receipt],
    })

    # exactly one person, one suggestion set, no duplicated evidence rows
    assert _person_count(database) == before["persons"] == 1
    assert _suggestion_count(database) == before["suggestions"]
    assert _count(database, "appearances") == before["appearances"]
    assert _count(database, "recognition_log") == before["recognition"]
    assert _count(database, "person_gallery") == before["gallery"]

    # media stayed canonical for the reused person
    for value in _stored_media_values(database):
        assert value.startswith(f"{person_id}/"), value
        assert "_staging" not in value and not re.search(r"cluster_\d", value), value


def test_finalize_appends_later_evidence_once_and_replay_changes_nothing(
    media_root,
    database,
):
    from forensics.person_creation.nodes.finalize import finalize

    initial = _face_paths(media_root, 6, prefix="finalize-initial")
    provider = {"snapshot": _snapshot(1, initial)}
    session = _session(database, lambda: provider["snapshot"])
    _run(session, provider, [(1, provider["snapshot"])])
    receipt = session.identity_decisions()[0]
    person_id = receipt["canonical_person_id"]

    output_dir = media_root / "session-finalize-later"
    promoted_dir = output_dir / "cluster_0" / "face_crops"
    promoted_dir.mkdir(parents=True)
    promoted = []
    records = []
    for index, source in enumerate(initial):
        destination = promoted_dir / Path(source).name
        destination.write_bytes(Path(source).read_bytes())
        promoted.append(str(destination))
        records.append({"crop_path": str(destination), "embedding": [1.0, 0.0]})
    for index in range(2):
        destination = promoted_dir / f"later_{index}.jpg"
        destination.write_bytes(f"later:{index}".encode())
        promoted.append(str(destination))
        records.append({"crop_path": str(destination), "embedding": [0.0, 1.0]})

    state = {
        "output_dir": str(output_dir),
        "person_name": "Live Subject",
        "video_paths": ["camera-source"],
        "rolling_analysis": {"analysis_version": 9},
        "identity_clusters": [{
            "cluster_id": 0,
            "face_records": records,
        }],
        "per_cluster_profiles": {
            0: {
                "id": "cluster_0",
                "name": "Live Subject",
                "face_embedding": [1.0, 0.0],
                "face_crops": promoted,
                "body_crops": [],
                "best_body_crops": [],
                "video_sources": ["camera-source"],
                "appearance": {"date": "2026-07-22"},
            },
        },
        "unresolved_faces": [],
        "unattached_bodies": [],
        "live_identity_decisions": [receipt],
    }
    before_count = _embedding(database, person_id)[1]
    finalize(state)
    after_first = {
        "embedding_count": _embedding(database, person_id)[1],
        "persons": _person_count(database),
        "suggestions": _suggestion_count(database),
        "ledger": _count(database, "identity_evidence"),
        "gallery": _count(database, "person_gallery"),
        "appearances": _count(database, "appearances"),
        "recognition": _count(database, "recognition_log"),
    }
    finalize(state)
    after_replay = {
        "embedding_count": _embedding(database, person_id)[1],
        "persons": _person_count(database),
        "suggestions": _suggestion_count(database),
        "ledger": _count(database, "identity_evidence"),
        "gallery": _count(database, "person_gallery"),
        "appearances": _count(database, "appearances"),
        "recognition": _count(database, "recognition_log"),
    }

    assert after_first["embedding_count"] == before_count + 2
    assert after_first["ledger"] == 8
    assert after_replay == after_first
    assert receipt["persisted_observation_count"] == 8
    assert receipt["last_appended_analysis_version"] == 9
    assert len(receipt["canonical_face_paths"]) == 8


# --- status API exposure -------------------------------------------------------


def test_status_api_exposes_live_identity_decision_fields(media_root, monkeypatch):
    from forensics.person_creation import service

    crop = media_root / "person_004" / "face_crops" / "best.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"jpeg")

    monkeypatch.setattr(service, "_start_pipeline_thread", lambda *_a: None)
    identity = {
        "session_person_id": "live_0001",
        "live_identity_id": "live_0001",
        "state": "attach_existing",
        "version": 2,
        "face_count": 4,
        "body_count": 1,
        "best_face_path": str(crop),          # absolute on purpose
        "best_body_path": "person_004/body_crops/missing.jpg",
        "first_seen": 0,
        "last_seen": 2,
        "candidate_person_id": "person_004",
        "candidate_similarity": 0.91,
        "second_candidate_person_id": "person_009",
        "second_candidate_similarity": 0.42,
        "margin": 0.49,
        "decision": "attach_existing",
        "canonical_person_id": "person_004",
        "suggestion_id": None,
        "decision_version": 1,
    }
    with service._jobs_lock:
        service._jobs.clear()
        service._jobs["job-status"] = service.JobState(
            "job-status",
            input_type="camera_uri",
            output_dir=str(media_root / "session"),
            snapshot={"rolling_analysis": {"live_identities": [identity]}},
        )
    try:
        payload = service.app.test_client().get(
            "/api/person/status/job-status"
        ).get_json()
    finally:
        with service._jobs_lock:
            service._jobs.clear()

    published = payload["snapshot"]["rolling_analysis"]["live_identities"][0]
    for field in (
        "live_identity_id", "state", "face_count", "body_count",
        "best_face_path", "best_body_path", "first_seen", "last_seen",
        "candidate_person_id", "candidate_similarity",
        "second_candidate_person_id", "second_candidate_similarity",
        "margin", "decision", "canonical_person_id", "suggestion_id",
        "decision_version", "version",
    ):
        assert field in published, field
    assert published["decision"] == "attach_existing"
    assert published["canonical_person_id"] == "person_004"
    assert published["decision_version"] == 1
    assert published["version"] == 2

    # media values are canonical relative paths; the absolute one was rewritten
    # and the missing one was dropped rather than leaked.
    assert published["best_face_path"] == "person_004/face_crops/best.jpg"
    assert published["best_body_path"] is None
    serialized = json.dumps(payload)
    assert "_staging" not in serialized
    assert not re.search(r"[A-Za-z]:[\\/]", serialized)
    assert not re.search(r"cluster_\d", serialized)
