from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from forensics.global_memory import (
    GlobalMemory,
    InactiveSourceError,
    InvalidMergeRequestError,
    PersonNotFoundError,
    RedirectChainError,
)


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _insert_person(
    memory: GlobalMemory,
    person_id: str,
    embedding=(1.0, 0.0, 0.0),
    *,
    active: bool = True,
    redirect: str | None = None,
) -> None:
    vector = _unit(embedding)
    memory._conn.execute(
        """
        INSERT INTO persons(
            person_id, name, embedding, embedding_count,
            enrolled_at, updated_at, cameras, is_active,
            merged_into_person_id
        ) VALUES (?, ?, ?, 2, '2026-07-01', '2026-07-01', '[]', ?, ?)
        """,
        (
            person_id,
            person_id.replace("_", " ").title(),
            sqlite3.Binary(vector.astype(np.float32).tobytes()),
            int(active),
            redirect,
        ),
    )


def _insert_colliding_evidence(memory: GlobalMemory) -> None:
    memory._conn.execute(
        """
        INSERT INTO appearances(
            person_id, date, top, full_description, clothing_status,
            best_body_crops, video_sources
        ) VALUES(
            'person_001', '2026-07-20', 'source top', 'source description',
            'ok', '["source-body.jpg"]', '["source-video"]'
        )
        """
    )
    memory._conn.execute(
        """
        INSERT INTO appearances(
            person_id, date, top, full_description, clothing_status,
            best_body_crops, video_sources
        ) VALUES(
            'person_003', '2026-07-20', 'target top', 'target description',
            'ok', '["target-body.jpg"]', '["target-video"]'
        )
        """
    )
    memory._conn.execute(
        """
        INSERT INTO person_gallery(
            person_id, crop_type, path, sharpness, session_date,
            video_source, width, height
        ) VALUES('person_001', 'face', 'same.jpg', 10.0, '2026-07-20',
                 'source-video', 100, 200)
        """
    )
    memory._conn.execute(
        """
        INSERT INTO person_gallery(
            person_id, crop_type, path, sharpness, session_date,
            video_source, width, height
        ) VALUES('person_003', 'face', 'same.jpg', 20.0, '2026-07-21',
                 'target-video', 300, 400)
        """
    )
    memory._conn.execute(
        """
        INSERT INTO recognition_log(
            person_id, event_type, similarity, embedding_count_before,
            embedding_count_after, video_sources, best_face_crop, ts
        ) VALUES('person_001', 'recognized', 0.81, 1, 2,
                 '["source-video"]', 'source-face.jpg', '2026-07-20T12:00:00')
        """
    )
    memory._conn.execute(
        """
        INSERT INTO recognition_log(
            person_id, event_type, similarity, embedding_count_before,
            embedding_count_after, video_sources, best_face_crop, ts
        ) VALUES('person_003', 'recognized', 0.91, 1, 2,
                 '["target-video"]', 'target-face.jpg', '2026-07-20T12:00:00')
        """
    )


def test_canonical_resolution_and_lineage_are_direct_deterministic_and_persisted(
    tmp_path,
):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_003")
        _insert_person(
            memory,
            "person_002",
            active=False,
            redirect="person_003",
        )
        _insert_person(
            memory,
            "person_001",
            active=False,
            redirect="person_003",
        )
        _insert_person(memory, "person_004", active=False)
        _insert_person(memory, "person_005")
        _insert_person(
            memory,
            "person_006",
            active=False,
            redirect="person_005",
        )

        assert memory.resolve_canonical_person_id("person_003") == "person_003"
        assert memory.resolve_canonical_person_id("person_001") == "person_003"
        lineage = memory.get_identity_lineage("person_002")
        assert lineage.canonical_person_id == "person_003"
        assert lineage.member_person_ids == (
            "person_003",
            "person_001",
            "person_002",
        )
        assert [member.is_active for member in lineage.members] == [True, False, False]
        assert [member.merged_into_person_id for member in lineage.members] == [
            None,
            "person_003",
            "person_003",
        ]
    finally:
        memory.close()

    read_only = GlobalMemory(database, read_only=True, media_root=tmp_path / "media")
    try:
        assert read_only.resolve_canonical_person_id("person_001") == "person_003"
        assert read_only.get_identity_lineage("person_003").member_person_ids == (
            "person_003",
            "person_001",
            "person_002",
        )
    finally:
        read_only.close()


def test_lineage_reads_preserve_collisions_provenance_and_legacy_contracts(tmp_path):
    database = tmp_path / "memory.db"
    memory = GlobalMemory(database, media_root=tmp_path / "media")
    try:
        _insert_person(
            memory,
            "person_001",
            active=False,
            redirect="person_003",
        )
        _insert_person(memory, "person_003")
        _insert_colliding_evidence(memory)

        appearances = memory.get_lineage_appearances("person_001")
        gallery = memory.get_lineage_gallery("person_003")
        history = memory.get_lineage_recognition_history("person_001")

        assert [row["original_person_id"] for row in appearances] == [
            "person_003",
            "person_001",
        ]
        assert [row["top"] for row in appearances] == ["target top", "source top"]
        assert [row["date"] for row in appearances] == [
            "2026-07-20",
            "2026-07-20",
        ]
        assert len({row["id"] for row in appearances}) == 2
        assert [row["original_person_id"] for row in gallery] == [
            "person_003",
            "person_001",
        ]
        assert [row["path"] for row in gallery] == ["same.jpg", "same.jpg"]
        assert [row["sharpness"] for row in gallery] == [20.0, 10.0]
        assert [row["original_person_id"] for row in history] == [
            "person_003",
            "person_001",
        ]
        assert [row["ts"] for row in history] == [
            "2026-07-20T12:00:00",
            "2026-07-20T12:00:00",
        ]
        assert all(row["canonical_person_id"] == "person_003" for row in [
            *appearances,
            *gallery,
            *history,
        ])

        assert memory.get_lineage_appearances("person_003") == appearances
        assert memory.get_lineage_gallery("person_001") == gallery
        assert memory.get_lineage_recognition_history("person_003") == history
        assert memory.query_by_date("2026-07-20")[0]["person_id"] == "person_003"
        assert [row["person_id"] for row in memory.get_recognition_history("person_003")] == [
            "person_003"
        ]
    finally:
        memory.close()

    read_only = GlobalMemory(database, read_only=True, media_root=tmp_path / "media")
    try:
        assert read_only.get_lineage_appearances("person_001") == appearances
        assert read_only.get_lineage_gallery("person_001") == gallery
        assert read_only.get_lineage_recognition_history("person_001") == history
    finally:
        read_only.close()


@pytest.mark.parametrize(
    ("setup", "person_id", "error"),
    [
        ("missing", "missing", PersonNotFoundError),
        ("inactive", "person_001", InactiveSourceError),
        ("self", "person_001", RedirectChainError),
        ("inactive_target", "person_001", RedirectChainError),
        ("chain", "person_001", RedirectChainError),
        ("cycle", "person_001", RedirectChainError),
    ],
)
def test_canonical_resolution_rejects_corrupt_states(tmp_path, setup, person_id, error):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        if setup == "inactive":
            _insert_person(memory, "person_001", active=False)
        elif setup == "self":
            _insert_person(
                memory,
                "person_001",
                active=False,
                redirect="person_001",
            )
        elif setup == "inactive_target":
            _insert_person(
                memory,
                "person_001",
                active=False,
                redirect="person_002",
            )
            _insert_person(memory, "person_002", active=False)
        elif setup == "chain":
            _insert_person(
                memory,
                "person_001",
                active=False,
                redirect="person_002",
            )
            _insert_person(
                memory,
                "person_002",
                active=False,
                redirect="person_003",
            )
            _insert_person(memory, "person_003")
        elif setup == "cycle":
            _insert_person(
                memory,
                "person_001",
                active=False,
                redirect="person_002",
            )
            _insert_person(
                memory,
                "person_002",
                active=False,
                redirect="person_001",
            )

        with pytest.raises(error):
            memory.resolve_canonical_person_id(person_id)
    finally:
        memory.close()


def test_lineage_rejects_a_descendant_redirect_chain(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_003")
        _insert_person(
            memory,
            "person_002",
            active=False,
            redirect="person_003",
        )
        _insert_person(
            memory,
            "person_001",
            active=False,
            redirect="person_002",
        )

        with pytest.raises(RedirectChainError):
            memory.get_identity_lineage("person_003")
    finally:
        memory.close()


def test_active_only_candidate_ranking_excludes_merged_source(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001", (0.0, 1.0, 0.0))
        _insert_person(memory, "person_002", (1.0, 0.0, 0.0))
        query = _unit((0.0, 1.0, 0.0))
        assert {candidate.person_id for candidate in memory._rank_active_identity_candidates(query)} == {
            "person_001",
            "person_002",
        }

        memory.merge_persons("person_001", "person_002", reason="same person")

        assert [candidate.person_id for candidate in memory._rank_active_identity_candidates(query)] == [
            "person_002"
        ]
    finally:
        memory.close()


@pytest.mark.parametrize("person_id", ["", "   ", None, 7])
def test_lineage_reads_reject_invalid_person_ids_without_query(tmp_path, person_id):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        with pytest.raises(InvalidMergeRequestError):
            memory.get_identity_lineage(person_id)
    finally:
        memory.close()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_lineage_history_rejects_invalid_limits(tmp_path, limit):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "media")
    try:
        _insert_person(memory, "person_001")
        with pytest.raises(InvalidMergeRequestError):
            memory.get_lineage_recognition_history("person_001", limit=limit)
    finally:
        memory.close()
