from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading

import numpy as np
import pytest

from forensics.global_memory import (
    AlreadyMergedConflictError,
    GlobalMemory,
    InactiveSourceError,
    InactiveTargetError,
    InvalidMergeEmbeddingError,
    InvalidMergeMetadataError,
    InvalidMergeRequestError,
    MergeAuditIntegrityError,
    PersonNotFoundError,
    RedirectChainError,
    SelfMergeError,
)
from forensics.global_memory.store import ReadOnlyGlobalMemoryError


TABLES = (
    "persons",
    "appearances",
    "person_gallery",
    "recognition_log",
    "identity_match_suggestions",
    "identity_merge_audit",
    "counters",
    "sqlite_sequence",
)


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _insert_person(
    memory: GlobalMemory,
    person_id: str,
    embedding,
    *,
    count: object,
    name: str | None = None,
    cameras: list | None = None,
    active: bool = True,
    redirect: str | None = None,
) -> None:
    vector = _unit(embedding)
    memory._conn.execute(
        """
        INSERT INTO persons(
            person_id, name, embedding, embedding_count,
            enrolled_at, updated_at, cameras, profile_image,
            profile_image_source, is_active, merged_into_person_id
        ) VALUES(?, ?, ?, ?, '2026-07-01', '2026-07-02T03:04:05', ?, ?,
                 'manual', ?, ?)
        """,
        (
            person_id,
            name or person_id,
            sqlite3.Binary(vector.astype(np.float32).tobytes()),
            count,
            json.dumps(cameras or []),
            f"{person_id}.jpg",
            int(active),
            redirect,
        ),
    )


def _snapshot(memory: GlobalMemory) -> dict[str, list[tuple]]:
    return {
        table: [
            tuple(row)
            for row in memory._conn.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid'
            ).fetchall()
        ]
        for table in TABLES
    }


def _file_hashes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _assert_clean_failure(
    memory: GlobalMemory,
    database: Path,
    before: dict[str, list[tuple]],
) -> None:
    assert _snapshot(memory) == before
    assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
    assert memory._conn.in_transaction is False
    second = sqlite3.connect(database, timeout=0.25, isolation_level=None)
    try:
        second.execute("PRAGMA busy_timeout=250")
        second.execute("BEGIN IMMEDIATE")
        second.execute("ROLLBACK")
    finally:
        second.close()


def _seed_complete_merge(memory: GlobalMemory, media_root: Path) -> None:
    media_root.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("source.jpg", b"source-image"),
        ("target.jpg", b"target-image"),
        ("shared.jpg", b"shared-gallery-image"),
        ("source-body.jpg", b"source-body"),
        ("target-body.jpg", b"target-body"),
    ):
        (media_root / name).write_bytes(content)

    _insert_person(
        memory,
        "person_001",
        (0.0, 1.0, 0.0),
        count=7,
        name="Reviewed Source",
        cameras=["camera-source", "camera-shared"],
    )
    _insert_person(
        memory,
        "person_002",
        (1.0, 0.0, 0.0),
        count=2,
        name="Reviewed Target",
        cameras=["camera-target", "camera-shared"],
    )
    _insert_person(memory, "person_003", (0.0, 0.0, 1.0), count=3)
    _insert_person(memory, "person_004", (1.0, 1.0, 0.0), count=4)
    memory._conn.execute("UPDATE counters SET value=4 WHERE key='person_count'")

    memory._conn.execute(
        """
        INSERT INTO appearances(
            person_id, date, top, bottom, full_description, clothing_status,
            best_body_crops, video_sources
        ) VALUES('person_001', '2026-07-20', 'source-top', 'source-bottom',
                 'source description', 'ok', '["source-body.jpg"]',
                 '["source-video"]')
        """
    )
    memory._conn.execute(
        """
        INSERT INTO appearances(
            person_id, date, top, bottom, full_description, clothing_status,
            best_body_crops, video_sources
        ) VALUES('person_002', '2026-07-20', 'target-top', 'target-bottom',
                 'target description', 'ok', '["target-body.jpg"]',
                 '["target-video"]')
        """
    )
    memory._conn.execute(
        """
        INSERT INTO person_gallery(
            person_id, crop_type, path, sharpness, session_date,
            video_source, width, height
        ) VALUES('person_001', 'face', 'shared.jpg', 11.0, '2026-07-20',
                 'source-video', 100, 200)
        """
    )
    memory._conn.execute(
        """
        INSERT INTO person_gallery(
            person_id, crop_type, path, sharpness, session_date,
            video_source, width, height
        ) VALUES('person_002', 'face', 'shared.jpg', 22.0, '2026-07-21',
                 'target-video', 300, 400)
        """
    )
    memory._conn.execute(
        """
        INSERT INTO recognition_log(
            person_id, event_type, similarity, embedding_count_before,
            embedding_count_after, video_sources, best_face_crop, ts
        ) VALUES('person_001', 'recognized', 0.82, 6, 7,
                 '["source-video"]', 'source.jpg', '2026-07-20T10:00:00')
        """
    )
    memory._conn.execute(
        """
        INSERT INTO recognition_log(
            person_id, event_type, similarity, embedding_count_before,
            embedding_count_after, video_sources, best_face_crop, ts
        ) VALUES('person_002', 'recognized', 0.92, 1, 2,
                 '["target-video"]', 'target.jpg', '2026-07-20T10:00:00')
        """
    )
    suggestion_sql = """
        INSERT INTO identity_match_suggestions(
            source_person_id, candidate_person_id, similarity,
            reason, status, created_at
        ) VALUES (?, ?, 0.75, 'test', ?, '2026-07-20T10:00:00')
    """
    memory._conn.execute(suggestion_sql, ("person_001", "person_002", "pending"))
    memory._conn.execute(suggestion_sql, ("person_003", "person_001", "pending"))
    memory._conn.execute(suggestion_sql, ("person_002", "person_003", "pending"))
    memory._conn.execute(suggestion_sql, ("person_001", "person_003", "accepted"))
    memory._conn.execute(suggestion_sql, ("person_004", "person_001", "rejected"))


def test_basic_logical_merge_preserves_provenance_audit_and_media(tmp_path):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    memory = GlobalMemory(database, media_root=media_root)
    try:
        _seed_complete_merge(memory, media_root)
        source_before = dict(memory._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_001'"
        ).fetchone())
        target_before = dict(memory._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_002'"
        ).fetchone())
        evidence_before = {
            table: _snapshot(memory)[table]
            for table in ("appearances", "person_gallery", "recognition_log", "counters")
        }
        media_before = _file_hashes(media_root)
        expected = memory._normalize_embedding(
            np.frombuffer(target_before["embedding"], dtype=np.float32) * 2
            + np.frombuffer(source_before["embedding"], dtype=np.float32) * 7
        )

        result = memory.merge_persons(
            "person_001",
            "person_002",
            reason="supervisor-confirmed duplicate",
            decision_source="phase-3f-test",
        )

        source_after = dict(memory._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_001'"
        ).fetchone())
        target_after = dict(memory._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_002'"
        ).fetchone())
        assert result.source_person_id == "person_001"
        assert result.target_person_id == "person_002"
        assert result.idempotent_replay is False
        assert result.source_embedding_count == 7
        assert result.target_embedding_count_before == 2
        assert result.target_embedding_count_after == 9
        assert result.staled_suggestion_count == 2
        assert result.lineage_member_count_after == 2
        assert target_after["name"] == target_before["name"] == "Reviewed Target"
        assert target_after["enrolled_at"] == target_before["enrolled_at"]
        assert target_after["profile_image"] == target_before["profile_image"]
        assert target_after["profile_image_source"] == "manual"
        assert target_after["is_active"] == 1
        assert target_after["merged_into_person_id"] is None
        assert target_after["embedding_count"] == 9
        assert np.allclose(
            np.frombuffer(target_after["embedding"], dtype=np.float32),
            expected,
            atol=1e-7,
        )
        assert json.loads(target_after["cameras"]) == [
            "camera-target",
            "camera-shared",
            "camera-source",
        ]
        assert source_after == {
            **source_before,
            "is_active": 0,
            "merged_into_person_id": "person_002",
        }
        for table, rows in evidence_before.items():
            assert _snapshot(memory)[table] == rows

        audit = memory._conn.execute("SELECT * FROM identity_merge_audit").fetchone()
        audit_before = tuple(audit)
        assert audit["merge_id"] == result.audit_id
        assert audit["source_person_id"] == "person_001"
        assert audit["target_person_id"] == "person_002"
        assert audit["reason"] == "supervisor-confirmed duplicate"
        assert audit["decision_source"] == "phase-3f-test"
        assert audit["similarity"] is None
        assert bytes(audit["source_embedding"]) == bytes(source_before["embedding"])
        assert audit["source_embedding_count"] == 7
        assert audit["source_name"] == "Reviewed Source"
        assert audit["target_name_before"] == "Reviewed Target"

        statuses = [
            tuple(row)
            for row in memory._conn.execute(
                "SELECT source_person_id,candidate_person_id,status "
                "FROM identity_match_suggestions ORDER BY id"
            ).fetchall()
        ]
        assert statuses == [
            ("person_001", "person_002", "stale"),
            ("person_003", "person_001", "stale"),
            ("person_002", "person_003", "pending"),
            ("person_001", "person_003", "accepted"),
            ("person_004", "person_001", "rejected"),
        ]
        assert [row["top"] for row in memory.get_lineage_appearances("person_002")] == [
            "target-top",
            "source-top",
        ]
        assert [row["sharpness"] for row in memory.get_lineage_gallery("person_001")] == [
            22.0,
            11.0,
        ]
        assert [
            row["original_person_id"]
            for row in memory.get_lineage_recognition_history("person_002")
        ] == ["person_002", "person_001"]
        assert _file_hashes(media_root) == media_before
        assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
        assert memory._conn.in_transaction is False

        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            memory._conn.execute(
                "UPDATE identity_merge_audit SET reason='changed' WHERE merge_id=?",
                (result.audit_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            memory._conn.execute(
                "DELETE FROM identity_merge_audit WHERE merge_id=?",
                (result.audit_id,),
            )
        assert tuple(memory._conn.execute(
            "SELECT * FROM identity_merge_audit WHERE merge_id=?",
            (result.audit_id,),
        ).fetchone()) == audit_before
    finally:
        memory.close()


def test_idempotent_replay_survives_restart_and_conflicting_replay_is_safe(tmp_path):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    memory = GlobalMemory(database, media_root=media_root)
    try:
        _seed_complete_merge(memory, media_root)
        first = memory.merge_persons("person_001", "person_002", reason="first")
        after_first = _snapshot(memory)
        media_after_first = _file_hashes(media_root)

        replay = memory.merge_persons("person_001", "person_002", reason="different")

        assert replay.idempotent_replay is True
        assert replay.audit_id == first.audit_id
        assert replay.source_embedding_count == first.source_embedding_count
        assert replay.target_embedding_count_before == first.target_embedding_count_after
        assert replay.target_embedding_count_after == first.target_embedding_count_after
        assert replay.staled_suggestion_count == 0
        assert _snapshot(memory) == after_first
        assert _file_hashes(media_root) == media_after_first
    finally:
        memory.close()

    reopened = GlobalMemory(database, media_root=media_root)
    try:
        replay = reopened.merge_persons("person_001", "person_002", reason="restart")
        assert replay.idempotent_replay is True
        assert replay.audit_id == first.audit_id
        assert _snapshot(reopened) == after_first
        with pytest.raises(AlreadyMergedConflictError):
            reopened.merge_persons("person_001", "person_003", reason="conflict")
        assert _snapshot(reopened) == after_first
    finally:
        reopened.close()


@pytest.mark.parametrize("corruption", ["missing", "wrong_embedding", "duplicate"])
def test_redirect_without_one_consistent_audit_rejects_replay(tmp_path, corruption):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0), count=4)
        memory._conn.execute(
            "UPDATE persons SET is_active=0, merged_into_person_id='person_002' "
            "WHERE person_id='person_001'"
        )
        if corruption != "missing":
            source = memory._conn.execute(
                "SELECT * FROM persons WHERE person_id='person_001'"
            ).fetchone()
            blob = bytes(source["embedding"])
            if corruption == "wrong_embedding":
                blob = _unit((0.0, 0.0, 1.0)).astype(np.float32).tobytes()
            for index in range(2 if corruption == "duplicate" else 1):
                memory._conn.execute(
                    """
                    INSERT INTO identity_merge_audit(
                        source_person_id,target_person_id,reason,decision_source,
                        similarity,source_embedding,source_embedding_count,
                        source_name,target_name_before,created_at
                    ) VALUES('person_001','person_002',?,'test',NULL,?,2,
                             'source','target','2026-07-20')
                    """,
                    (f"audit-{index}", sqlite3.Binary(blob)),
                )
        before = _snapshot(memory)

        with pytest.raises(MergeAuditIntegrityError):
            memory.merge_persons("person_001", "person_002", reason="replay")

        _assert_clean_failure(memory, tmp_path / "memory.db", before)
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("source", "target", "reason", "error"),
    [
        ("", "person_002", "reason", InvalidMergeRequestError),
        ("   ", "person_002", "reason", InvalidMergeRequestError),
        (None, "person_002", "reason", InvalidMergeRequestError),
        ("person_001", "", "reason", InvalidMergeRequestError),
        ("person_001", "   ", "reason", InvalidMergeRequestError),
        ("person_001", None, "reason", InvalidMergeRequestError),
        ("person_001", "person_002", "", InvalidMergeRequestError),
        ("person_001", "person_002", "   ", InvalidMergeRequestError),
        ("person_001", "person_001", "reason", SelfMergeError),
        ("missing", "person_002", "reason", PersonNotFoundError),
        ("person_001", "missing", "reason", PersonNotFoundError),
    ],
)
def test_invalid_requests_leave_complete_database_unchanged(
    tmp_path,
    source,
    target,
    reason,
    error,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0), count=4)
        before = _snapshot(memory)
        with pytest.raises(error):
            memory.merge_persons(source, target, reason=reason)
        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("state", "error"),
    [
        ("inactive_target", InactiveTargetError),
        ("redirected_target", InactiveTargetError),
        ("active_redirected_target", RedirectChainError),
        ("inactive_source", InactiveSourceError),
        ("different_redirect", AlreadyMergedConflictError),
        ("source_has_child", RedirectChainError),
    ],
)
def test_inactive_and_redirect_states_reject_without_mutation(tmp_path, state, error):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0), count=4)
        _insert_person(memory, "person_003", (0.0, 0.0, 1.0), count=3)
        if state == "inactive_target":
            memory._conn.execute("UPDATE persons SET is_active=0 WHERE person_id='person_002'")
        elif state == "redirected_target":
            memory._conn.execute(
                "UPDATE persons SET is_active=0, merged_into_person_id='person_003' "
                "WHERE person_id='person_002'"
            )
        elif state == "active_redirected_target":
            memory._conn.execute(
                "UPDATE persons SET merged_into_person_id='person_003' "
                "WHERE person_id='person_002'"
            )
        elif state == "inactive_source":
            memory._conn.execute("UPDATE persons SET is_active=0 WHERE person_id='person_001'")
        elif state == "different_redirect":
            memory._conn.execute(
                "UPDATE persons SET is_active=0, merged_into_person_id='person_003' "
                "WHERE person_id='person_001'"
            )
        elif state == "source_has_child":
            memory._conn.execute(
                "UPDATE persons SET is_active=0, merged_into_person_id='person_001' "
                "WHERE person_id='person_003'"
            )
        before = _snapshot(memory)

        with pytest.raises(error):
            memory.merge_persons("person_001", "person_002", reason="invalid")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


BAD_EMBEDDINGS = {
    "empty": b"",
    "malformed": b"\x00",
    "dimension_mismatch": np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
    "zero": np.asarray([0.0, 0.0, 0.0], dtype=np.float32).tobytes(),
    "nan": np.asarray([np.nan, 0.0, 0.0], dtype=np.float32).tobytes(),
    "positive_infinity": np.asarray([np.inf, 0.0, 0.0], dtype=np.float32).tobytes(),
    "negative_infinity": np.asarray([-np.inf, 0.0, 0.0], dtype=np.float32).tobytes(),
}


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("kind", sorted(BAD_EMBEDDINGS))
def test_stored_embedding_hostility_rolls_back_exactly(tmp_path, role, kind):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0), count=4)
        person_id = "person_001" if role == "source" else "person_002"
        memory._conn.execute(
            "UPDATE persons SET embedding=? WHERE person_id=?",
            (sqlite3.Binary(BAD_EMBEDDINGS[kind]), person_id),
        )
        before = _snapshot(memory)

        with pytest.raises(InvalidMergeEmbeddingError):
            memory.merge_persons("person_001", "person_002", reason="hostile")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("bad_count", [0, -1, 1.5, "invalid"])
def test_stored_embedding_count_hostility_rolls_back_exactly(
    tmp_path,
    role,
    bad_count,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0), count=4)
        person_id = "person_001" if role == "source" else "person_002"
        memory._conn.execute(
            "UPDATE persons SET embedding_count=? WHERE person_id=?",
            (bad_count, person_id),
        )
        before = _snapshot(memory)

        with pytest.raises(InvalidMergeEmbeddingError):
            memory.merge_persons("person_001", "person_002", reason="hostile")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize("role", ["source", "target"])
def test_unstorable_missing_embedding_and_boolean_count_are_typed(role):
    with pytest.raises(InvalidMergeEmbeddingError, match="missing"):
        GlobalMemory._validated_stored_merge_embedding({"embedding": None}, role)
    with pytest.raises(InvalidMergeEmbeddingError, match="positive integer"):
        GlobalMemory._validated_stored_merge_count({"embedding_count": True}, role)


@pytest.mark.parametrize("residual", [0.0, 1e-12, 1e-10, 1e-8])
def test_exact_and_near_embedding_cancellation_rejects_before_normalization(
    tmp_path,
    residual,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (-1.0, residual), count=1)
        _insert_person(memory, "person_002", (1.0, 0.0), count=1)
        source = memory._conn.execute(
            "SELECT embedding FROM persons WHERE person_id='person_001'"
        ).fetchone()
        target = memory._conn.execute(
            "SELECT embedding FROM persons WHERE person_id='person_002'"
        ).fetchone()
        weighted = (
            np.frombuffer(target["embedding"], dtype=np.float32).astype(np.float64)
            + np.frombuffer(source["embedding"], dtype=np.float32).astype(np.float64)
        )
        weighted_norm = float(np.linalg.norm(weighted))
        if residual == 0.0:
            assert weighted_norm == 0.0
        else:
            assert weighted_norm == pytest.approx(residual, rel=1e-6)
        before = _snapshot(memory)

        with pytest.raises(InvalidMergeEmbeddingError, match="cancelled"):
            memory.merge_persons("person_001", "person_002", reason="cancellation")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("target_embedding", "source_embedding", "target_count", "source_count"),
    [
        ((1.0, 0.0), (-1.0, 1e-6), 1, 1),
        ((1.0, 0.0), (-1.0, 0.0), 2, 1),
        ((1.0, 0.0), (0.0, 1.0), 1, 1),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2, 7),
    ],
)
def test_safe_small_asymmetric_orthogonal_and_normal_embeddings_merge(
    tmp_path,
    target_embedding,
    source_embedding,
    target_count,
    source_count,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(
            memory,
            "person_001",
            source_embedding,
            count=source_count,
        )
        _insert_person(
            memory,
            "person_002",
            target_embedding,
            count=target_count,
        )
        source_before = memory._conn.execute(
            "SELECT embedding FROM persons WHERE person_id='person_001'"
        ).fetchone()
        target_before = memory._conn.execute(
            "SELECT embedding FROM persons WHERE person_id='person_002'"
        ).fetchone()
        weighted = (
            np.frombuffer(target_before["embedding"], dtype=np.float32).astype(np.float64)
            * target_count
            + np.frombuffer(source_before["embedding"], dtype=np.float32).astype(np.float64)
            * source_count
        )
        assert float(np.linalg.norm(weighted)) > (target_count + source_count) * 1e-8
        expected = (weighted / np.linalg.norm(weighted)).astype(np.float32)

        result = memory.merge_persons(
            "person_001",
            "person_002",
            reason="safe weighted embedding",
        )

        target_after = memory._conn.execute(
            "SELECT embedding, embedding_count FROM persons WHERE person_id='person_002'"
        ).fetchone()
        assert result.idempotent_replay is False
        assert result.target_embedding_count_after == target_count + source_count
        assert target_after["embedding_count"] == target_count + source_count
        assert bytes(target_after["embedding"]) == expected.tobytes()
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
        assert memory._conn.in_transaction is False
    finally:
        memory.close()


SQLITE_MAX_INTEGER = 2**63 - 1


@pytest.mark.parametrize(
    ("target_count", "source_count"),
    [
        (SQLITE_MAX_INTEGER, 1),
        (SQLITE_MAX_INTEGER, SQLITE_MAX_INTEGER),
    ],
)
def test_embedding_count_overflow_is_typed_and_rolls_back_exactly(
    tmp_path,
    target_count,
    source_count,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0), count=source_count)
        _insert_person(memory, "person_002", (1.0, 0.0), count=target_count)
        before = _snapshot(memory)

        with pytest.raises(InvalidMergeEmbeddingError, match="SQLite signed INTEGER"):
            memory.merge_persons("person_001", "person_002", reason="overflow")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("target_count", "source_count", "expected_count"),
    [
        (SQLITE_MAX_INTEGER - 1, 1, SQLITE_MAX_INTEGER),
        (2, 7, 9),
    ],
)
def test_embedding_count_valid_boundaries_persist_exact_sum(
    tmp_path,
    target_count,
    source_count,
    expected_count,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0), count=source_count)
        _insert_person(memory, "person_002", (1.0, 0.0), count=target_count)

        result = memory.merge_persons("person_001", "person_002", reason="count boundary")

        assert result.target_embedding_count_after == expected_count
        assert memory._conn.execute(
            "SELECT embedding_count FROM persons WHERE person_id='person_002'"
        ).fetchone()[0] == expected_count
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
        assert memory._conn.in_transaction is False
    finally:
        memory.close()


INVALID_CAMERA_JSON = {
    "malformed": "{",
    "null": "null",
    "object": '{"camera": "camera-1"}',
    "number": "7",
    "string": '"camera-1"',
    "mixed_array": '["camera-1", 7]',
    "nested_array": '[["camera-1"]]',
    "nested_object": '[{"camera": "camera-1"}]',
}


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("bad_json", INVALID_CAMERA_JSON.values(), ids=INVALID_CAMERA_JSON)
def test_invalid_merge_camera_metadata_is_typed_and_rolls_back_exactly(
    tmp_path,
    role,
    bad_json,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(
            memory,
            "person_001",
            (0.0, 1.0),
            count=2,
            cameras=["source-camera"],
        )
        _insert_person(
            memory,
            "person_002",
            (1.0, 0.0),
            count=4,
            cameras=["target-camera"],
        )
        person_id = "person_001" if role == "source" else "person_002"
        memory._conn.execute(
            "UPDATE persons SET cameras=? WHERE person_id=?",
            (bad_json, person_id),
        )
        before = _snapshot(memory)

        with pytest.raises(InvalidMergeMetadataError, match=f"{role} cameras"):
            memory.merge_persons("person_001", "person_002", reason="bad cameras")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


@pytest.mark.parametrize("role", ["source", "target"])
def test_non_text_merge_camera_metadata_is_typed(role):
    with pytest.raises(InvalidMergeMetadataError, match=f"{role} cameras"):
        GlobalMemory._validated_merge_cameras(None, role)


@pytest.mark.parametrize(
    ("target_cameras", "source_cameras", "expected"),
    [
        ([], [], []),
        ([], ["source"], ["source"]),
        (["target"], [], ["target"]),
        (["target"], ["source"], ["target", "source"]),
        (
            ["target", "shared", "target"],
            ["source", "shared", "source"],
            ["target", "shared", "source"],
        ),
    ],
)
def test_valid_camera_arrays_merge_target_first_and_deduplicate(
    tmp_path,
    target_cameras,
    source_cameras,
    expected,
):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        _insert_person(
            memory,
            "person_001",
            (0.0, 1.0),
            count=2,
            cameras=source_cameras,
        )
        _insert_person(
            memory,
            "person_002",
            (1.0, 0.0),
            count=4,
            cameras=target_cameras,
        )

        memory.merge_persons("person_001", "person_002", reason="valid cameras")

        cameras = memory._conn.execute(
            "SELECT cameras FROM persons WHERE person_id='person_002'"
        ).fetchone()[0]
        assert json.loads(cameras) == expected
        assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
    finally:
        memory.close()


FAILURE_STAGES = (
    "_insert_person_merge_audit",
    "_update_merge_target_embedding",
    "_update_merge_target_metadata",
    "_stale_merge_suggestions",
    "_deactivate_merge_source",
    "_redirect_merge_source",
    "_before_merge_commit",
)


@pytest.mark.parametrize("stage", FAILURE_STAGES)
def test_failure_injection_matrix_restores_every_value_and_releases_lock(
    tmp_path,
    monkeypatch,
    stage,
):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    memory = GlobalMemory(database, media_root=media_root)
    try:
        _seed_complete_merge(memory, media_root)
        before = _snapshot(memory)
        media_before = _file_hashes(media_root)
        original = getattr(memory, stage)

        def fail_after(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(f"injected {stage} failure")

        monkeypatch.setattr(memory, stage, fail_after)
        with pytest.raises(RuntimeError, match="injected"):
            memory.merge_persons("person_001", "person_002", reason="rollback")

        _assert_clean_failure(memory, database, before)
        assert _file_hashes(media_root) == media_before
    finally:
        memory.close()


@pytest.mark.parametrize("trigger_target", ["audit", "target_embedding"])
def test_real_sqlite_trigger_failures_roll_back_complete_merge(tmp_path, trigger_target):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    memory = GlobalMemory(database, media_root=media_root)
    try:
        _seed_complete_merge(memory, media_root)
        if trigger_target == "audit":
            memory._conn.execute(
                """
                CREATE TRIGGER fail_merge_audit
                BEFORE INSERT ON identity_merge_audit
                BEGIN SELECT RAISE(ABORT, 'injected audit failure'); END
                """
            )
            expected_error = MergeAuditIntegrityError
        else:
            memory._conn.execute(
                """
                CREATE TRIGGER fail_merge_target_embedding
                BEFORE UPDATE OF embedding ON persons
                WHEN OLD.person_id='person_002'
                BEGIN SELECT RAISE(ABORT, 'injected target failure'); END
                """
            )
            expected_error = sqlite3.IntegrityError
        before = _snapshot(memory)
        media_before = _file_hashes(media_root)

        with pytest.raises(expected_error):
            memory.merge_persons("person_001", "person_002", reason="trigger")

        _assert_clean_failure(memory, database, before)
        assert _file_hashes(media_root) == media_before
    finally:
        memory.close()


def test_read_only_allows_lineage_reads_but_rejects_merge_without_changes(tmp_path):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    writer = GlobalMemory(database, media_root=media_root)
    try:
        _insert_person(writer, "person_001", (0.0, 1.0, 0.0), count=2)
        _insert_person(writer, "person_002", (1.0, 0.0, 0.0), count=4)
        before = _snapshot(writer)
    finally:
        writer.close()

    read_only = GlobalMemory(database, read_only=True, media_root=media_root)
    try:
        assert read_only.resolve_canonical_person_id("person_001") == "person_001"
        assert read_only.get_identity_lineage("person_001").member_person_ids == (
            "person_001",
        )
        files_before_merge = _file_hashes(tmp_path)
        with pytest.raises(ReadOnlyGlobalMemoryError):
            read_only.merge_persons("person_001", "person_002", reason="forbidden")
        assert read_only._conn.in_transaction is False
        assert _file_hashes(tmp_path) == files_before_merge
    finally:
        read_only.close()

    connection = sqlite3.connect(database)
    try:
        connection.row_factory = sqlite3.Row
        after = {
            table: [
                tuple(row)
                for row in connection.execute(
                    f'SELECT * FROM "{table}" ORDER BY rowid'
                ).fetchall()
            ]
            for table in TABLES
        }
        assert after == before
    finally:
        connection.close()


def test_begin_immediate_blocks_other_writes_until_merge_commits(tmp_path, monkeypatch):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    first = GlobalMemory(database, media_root=media_root)
    _seed_complete_merge(first, media_root)
    second = GlobalMemory(database, media_root=media_root)
    second._conn.execute("PRAGMA busy_timeout=150")
    validated = threading.Event()
    release = threading.Event()
    original = first._require_no_source_merge_audit
    errors: list[BaseException] = []

    def pause_after_validation(source_id):
        original(source_id)
        validated.set()
        assert release.wait(5)

    monkeypatch.setattr(first, "_require_no_source_merge_audit", pause_after_validation)

    def run_merge():
        try:
            first.merge_persons("person_001", "person_002", reason="concurrent")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_merge)
    thread.start()
    try:
        assert validated.wait(5)
        blocked_operations = (
            (
                "UPDATE persons SET name='blocked-source' WHERE person_id='person_001'",
                (),
            ),
            (
                "UPDATE persons SET name='blocked-target' WHERE person_id='person_002'",
                (),
            ),
            (
                "INSERT INTO appearances(person_id,date,clothing_status) "
                "VALUES('person_001','2099-01-01','ok')",
                (),
            ),
        )
        for sql, parameters in blocked_operations:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                second._conn.execute(sql, parameters)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            second.merge_persons("person_001", "person_002", reason="blocked")
    finally:
        release.set()
        thread.join(5)
        first.close()
        second.close()
    assert not thread.is_alive()
    assert errors == []

    reopened = GlobalMemory(database, media_root=media_root)
    try:
        assert reopened.resolve_canonical_person_id("person_001") == "person_002"
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert reopened._conn.execute(
            "SELECT name FROM persons WHERE person_id='person_001'"
        ).fetchone()[0] == "Reviewed Source"
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM appearances WHERE date='2099-01-01'"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


def test_concurrent_conflicting_targets_allow_at_most_one_merge(tmp_path):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    seed = GlobalMemory(database, media_root=media_root)
    try:
        _insert_person(seed, "person_001", (0.0, 1.0, 0.0), count=7)
        _insert_person(seed, "person_002", (1.0, 0.0, 0.0), count=2)
        _insert_person(seed, "person_003", (0.0, 0.0, 1.0), count=3)
    finally:
        seed.close()

    memories = [
        GlobalMemory(database, media_root=media_root),
        GlobalMemory(database, media_root=media_root),
    ]
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []

    def attempt(memory: GlobalMemory, target_id: str):
        barrier.wait()
        try:
            result = memory.merge_persons("person_001", target_id, reason=target_id)
            outcomes.append(("success", result))
        except BaseException as exc:
            outcomes.append(("error", exc))

    threads = [
        threading.Thread(target=attempt, args=(memories[0], "person_002")),
        threading.Thread(target=attempt, args=(memories[1], "person_003")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    for memory in memories:
        memory.close()

    assert all(not thread.is_alive() for thread in threads)
    successes = [value for kind, value in outcomes if kind == "success"]
    failures = [value for kind, value in outcomes if kind == "error"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AlreadyMergedConflictError)

    verify = GlobalMemory(database, media_root=media_root)
    try:
        source = verify._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_001'"
        ).fetchone()
        winner = successes[0].target_person_id
        loser = "person_003" if winner == "person_002" else "person_002"
        assert source["is_active"] == 0
        assert source["merged_into_person_id"] == winner
        assert verify._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert verify._conn.execute(
            "SELECT embedding_count FROM persons WHERE person_id=?",
            (winner,),
        ).fetchone()[0] == (9 if winner == "person_002" else 10)
        assert verify._conn.execute(
            "SELECT embedding_count FROM persons WHERE person_id=?",
            (loser,),
        ).fetchone()[0] == (2 if loser == "person_002" else 3)
        assert [tuple(row) for row in verify._conn.execute("PRAGMA foreign_key_check")] == []
    finally:
        verify.close()


REDIRECT_HOSTILE_STATES = {
    "three_person_cycle": (
        ("person_001", False, "person_002"),
        ("person_002", False, "person_003"),
        ("person_003", False, "person_001"),
    ),
    "four_person_cycle": (
        ("person_001", False, "person_002"),
        ("person_002", False, "person_003"),
        ("person_003", False, "person_004"),
        ("person_004", False, "person_001"),
    ),
    "long_chain_to_active": (
        ("person_001", False, "person_002"),
        ("person_002", False, "person_003"),
        ("person_003", False, "person_004"),
        ("person_004", False, "person_005"),
        ("person_005", True, None),
    ),
    "long_chain_to_missing": (
        ("person_001", False, "person_002"),
        ("person_002", False, "person_003"),
        ("person_003", False, "missing-person"),
    ),
    "chain_with_inactive_intermediate": (
        ("person_001", False, "person_002"),
        ("person_002", False, "person_003"),
        ("person_003", False, None),
    ),
}


@pytest.mark.parametrize(
    "persisted_state",
    REDIRECT_HOSTILE_STATES.values(),
    ids=REDIRECT_HOSTILE_STATES,
)
def test_long_redirect_cycles_and_chains_are_rejected_without_mutation(
    tmp_path,
    persisted_state,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        for index, (person_id, active, redirect) in enumerate(persisted_state, start=1):
            _insert_person(
                memory,
                person_id,
                (1.0, float(index)),
                count=index,
                active=active,
                redirect=redirect,
            )
        _insert_person(
            memory,
            "person_target",
            (0.0, 1.0),
            count=2,
        )
        before = _snapshot(memory)

        with pytest.raises(RedirectChainError):
            memory.resolve_canonical_person_id("person_001")
        assert _snapshot(memory) == before
        with pytest.raises(RedirectChainError):
            memory.get_identity_lineage("person_001")
        assert _snapshot(memory) == before
        with pytest.raises(AlreadyMergedConflictError):
            memory.merge_persons("person_001", "person_target", reason="corrupt chain")

        _assert_clean_failure(memory, database, before)
    finally:
        memory.close()


def test_concurrent_same_target_merge_blends_once_and_replays(tmp_path):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "media"
    seed = GlobalMemory(database, media_root=media_root)
    try:
        _seed_complete_merge(seed, media_root)
        source_before = dict(seed._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_001'"
        ).fetchone())
        target_before = dict(seed._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_002'"
        ).fetchone())
        weighted = (
            np.frombuffer(target_before["embedding"], dtype=np.float32).astype(np.float64)
            * target_before["embedding_count"]
            + np.frombuffer(source_before["embedding"], dtype=np.float32).astype(np.float64)
            * source_before["embedding_count"]
        )
        expected_embedding = (weighted / np.linalg.norm(weighted)).astype(np.float32)
    finally:
        seed.close()

    memories = [
        GlobalMemory(database, media_root=media_root),
        GlobalMemory(database, media_root=media_root),
    ]
    for memory in memories:
        memory._conn.execute("PRAGMA busy_timeout=2000")
    barrier = threading.Barrier(2)
    outcomes: list[tuple[int, str, object]] = []
    outcome_lock = threading.Lock()

    def attempt(index: int):
        try:
            barrier.wait(timeout=5)
            result = memories[index].merge_persons(
                "person_001",
                "person_002",
                reason="same target concurrency",
            )
            outcome = (index, "success", result)
        except BaseException as exc:
            outcome = (index, "error", exc)
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    try:
        assert all(not thread.is_alive() for thread in threads)
        lock_failures = [
            item
            for item in outcomes
            if item[1] == "error"
            and isinstance(item[2], sqlite3.OperationalError)
            and "locked" in str(item[2]).lower()
        ]
        unexpected = [
            item for item in outcomes if item[1] == "error" and item not in lock_failures
        ]
        assert unexpected == []
        for index, _kind, _error in lock_failures:
            replay = memories[index].merge_persons(
                "person_001",
                "person_002",
                reason="same target retry",
            )
            outcomes.append((index, "success", replay))

        results = [item[2] for item in outcomes if item[1] == "success"]
        first_time = [result for result in results if not result.idempotent_replay]
        replays = [result for result in results if result.idempotent_replay]
        assert len(first_time) == 1
        assert len(replays) == 1
        assert first_time[0].staled_suggestion_count == 2
        assert replays[0].staled_suggestion_count == 0
    finally:
        for memory in memories:
            memory.close()

    verify = GlobalMemory(database, media_root=media_root)
    try:
        source_after = verify._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_001'"
        ).fetchone()
        target_after = verify._conn.execute(
            "SELECT * FROM persons WHERE person_id='person_002'"
        ).fetchone()
        assert source_after["is_active"] == 0
        assert source_after["merged_into_person_id"] == "person_002"
        assert target_after["embedding_count"] == 9
        assert bytes(target_after["embedding"]) == expected_embedding.tobytes()
        assert verify._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert verify._conn.execute(
            "SELECT COUNT(*) FROM persons "
            "WHERE person_id='person_001' AND merged_into_person_id='person_002'"
        ).fetchone()[0] == 1
        assert [
            tuple(row)
            for row in verify._conn.execute(
                "SELECT source_person_id,candidate_person_id,status "
                "FROM identity_match_suggestions ORDER BY id"
            ).fetchall()
        ] == [
            ("person_001", "person_002", "stale"),
            ("person_003", "person_001", "stale"),
            ("person_002", "person_003", "pending"),
            ("person_001", "person_003", "accepted"),
            ("person_004", "person_001", "rejected"),
        ]
        assert [tuple(row) for row in verify._conn.execute("PRAGMA foreign_key_check")] == []
        assert verify._conn.in_transaction is False
    finally:
        verify.close()

    lock_probe = sqlite3.connect(database, timeout=0.25, isolation_level=None)
    try:
        lock_probe.execute("PRAGMA busy_timeout=250")
        lock_probe.execute("BEGIN IMMEDIATE")
        lock_probe.execute("ROLLBACK")
    finally:
        lock_probe.close()
