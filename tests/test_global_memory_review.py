from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import numpy as np
import pytest

from forensics.global_memory import (
    GlobalMemory,
    IdentityReviewDecision,
    InvalidReviewRequestError,
    ReviewSuggestionConflictError,
    ReviewSuggestionIntegrityError,
    ReviewSuggestionNotFoundError,
    ReviewSuggestionStaleError,
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


def _unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _person(memory, person_id, values, *, count=1, name=None):
    embedding = _unit(values)
    memory._conn.execute(
        """
        INSERT INTO persons(
            person_id, name, embedding, embedding_count, enrolled_at,
            updated_at, cameras, profile_image, profile_image_source,
            is_active, merged_into_person_id
        ) VALUES(?, ?, ?, ?, '2026-07-01', '2026-07-02T03:04:05', ?, ?,
                 'auto', 1, NULL)
        """,
        (
            person_id,
            name or person_id,
            sqlite3.Binary(embedding.tobytes()),
            count,
            json.dumps(["rtsp://supervisor:secret@camera.local/live", "lobby"]),
            f"{person_id}.jpg",
        ),
    )


def _suggestion(
    memory,
    source="person_001",
    candidate="person_002",
    *,
    status="pending",
    created_at="2026-07-20T10:00:00",
):
    cursor = memory._conn.execute(
        """
        INSERT INTO identity_match_suggestions(
            source_person_id, candidate_person_id, similarity,
            second_similarity, margin, reason, status, created_at
        ) VALUES(?, ?, 0.87, 0.71, 0.16, 'ambiguous_match', ?, ?)
        """,
        (source, candidate, status, created_at),
    )
    return int(cursor.lastrowid)


def _seed(memory, media_root):
    media_root.mkdir(parents=True, exist_ok=True)
    for person_id in ("person_001", "person_002", "person_003", "person_004"):
        (media_root / f"{person_id}.jpg").write_bytes(person_id.encode())
    _person(memory, "person_001", (0, 1, 0), count=3, name="New Source")
    _person(memory, "person_002", (1, 0, 0), count=5, name="Known Candidate")
    _person(memory, "person_003", (0, 0, 1), count=2)
    _person(memory, "person_004", (1, 1, 0), count=4)
    memory._conn.execute("UPDATE counters SET value=4 WHERE key='person_count'")
    for person_id, label, sharpness in (
        ("person_001", "source", 11.0),
        ("person_002", "candidate", 22.0),
    ):
        memory._conn.execute(
            """
            INSERT INTO appearances(
                person_id, date, top, bottom, full_description,
                best_body_crops, video_sources
            ) VALUES(?, '2026-07-20', ?, 'dark trousers', ?, '[]', ?)
            """,
            (
                person_id,
                f"{label} top",
                f"{label} appearance",
                json.dumps([f"C:\\private\\{label}.mp4"]),
            ),
        )
        memory._conn.execute(
            """
            INSERT INTO person_gallery(
                person_id, crop_type, path, sharpness, session_date,
                video_source, width, height
            ) VALUES(?, 'face', ?, ?, '2026-07-20', ?, 100, 100)
            """,
            (person_id, f"{person_id}.jpg", sharpness, f"{label}-camera"),
        )
        memory._conn.execute(
            """
            INSERT INTO recognition_log(
                person_id, event_type, similarity, embedding_count_before,
                embedding_count_after, video_sources, best_face_crop, ts
            ) VALUES(?, 'new_enrollment', NULL, NULL, 1, ?, ?, '2026-07-20T12:00:00')
            """,
            (
                person_id,
                json.dumps(["rtsp://supervisor:secret@camera.local/live"]),
                f"{person_id}.jpg",
            ),
        )


def _memory(tmp_path):
    database = tmp_path / "memory.db"
    media = tmp_path / "media"
    memory = GlobalMemory(database, media_root=media)
    _seed(memory, media)
    return memory, database, media


def _snapshot(memory):
    return {
        table: [tuple(row) for row in memory._conn.execute(
            f'SELECT * FROM "{table}" ORDER BY rowid'
        ).fetchall()]
        for table in TABLES
    }


def _media_hashes(media_root):
    return {
        str(path.relative_to(media_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(media_root.rglob("*"))
        if path.is_file()
    }


def _assert_clean(memory, database, expected):
    assert _snapshot(memory) == expected
    assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
    assert memory._conn.in_transaction is False
    contender = sqlite3.connect(database, isolation_level=None, timeout=0.25)
    try:
        contender.execute("BEGIN IMMEDIATE")
        contender.execute("ROLLBACK")
    finally:
        contender.close()


def test_pending_queue_is_pending_only_oldest_first_and_paginated(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        later = _suggestion(memory, created_at="2026-07-20T11:00:00")
        earlier = _suggestion(memory, "person_003", "person_004", created_at="2026-07-19")
        rejected = _suggestion(memory, "person_004", "person_002", status="rejected")

        queue = memory.list_pending_identity_reviews()
        assert [item.suggestion_id for item in queue] == [str(earlier), str(later)]
        assert all(item.status == "pending" for item in queue)
        assert str(rejected) not in {item.suggestion_id for item in queue}
        assert memory.count_pending_identity_reviews() == 2
        assert memory.list_pending_identity_reviews(limit=1, offset=1)[0].suggestion_id == str(later)
        assert "secret" not in json.dumps([item.as_dict() for item in queue])
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("limit", "offset"),
    [
        (0, 0),
        (101, 0),
        (True, 0),
        ("5", 0),
        (1, -1),
        (1, True),
        (1, 2**63),
    ],
)
def test_pending_queue_validates_pagination(tmp_path, limit, offset):
    memory, _, _ = _memory(tmp_path)
    try:
        with pytest.raises(InvalidReviewRequestError):
            memory.list_pending_identity_reviews(limit=limit, offset=offset)
    finally:
        memory.close()


def test_detail_separates_source_and_candidate_evidence_and_is_safe(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory)
        detail = memory.get_identity_review(str(review_id)).as_dict()
        assert detail["source_profile"]["person_id"] == "person_001"
        assert detail["candidate_profile"]["person_id"] == "person_002"
        assert {row["original_person_id"] for row in detail["source_gallery"]} == {"person_001"}
        assert {row["original_person_id"] for row in detail["candidate_gallery"]} == {"person_002"}
        assert {row["original_person_id"] for row in detail["source_appearances"]} == {"person_001"}
        assert {row["original_person_id"] for row in detail["candidate_appearances"]} == {"person_002"}
        serialized = json.dumps(detail)
        assert '"embedding":' not in serialized.lower()
        assert "secret" not in serialized
        assert "supervisor" not in serialized
        assert "C:\\private" not in serialized
        assert "camera.local/live" in serialized
    finally:
        memory.close()


def test_missing_detail_is_typed(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        with pytest.raises(ReviewSuggestionNotFoundError):
            memory.get_identity_review("999")
    finally:
        memory.close()


@pytest.mark.parametrize(
    "suggestion_id",
    [
        "",
        "9" * 4301,
        "9" * 6000,
        "१२३",
        "١٢٣",
        "１２３",
        "𝟙𝟚𝟛",
        "+1",
        "-1",
        "1.0",
        "1e3",
        "0x1",
        " ",
        str(2**63),
    ],
)
def test_review_suggestion_id_rejects_non_ascii_and_out_of_range_values(
    tmp_path,
    suggestion_id,
):
    memory, _, _ = _memory(tmp_path)
    try:
        with pytest.raises(InvalidReviewRequestError):
            memory.get_identity_review(suggestion_id)
        memory._conn.execute("BEGIN IMMEDIATE")
        memory._conn.execute("ROLLBACK")
    finally:
        memory.close()


def test_accept_merges_source_into_candidate_accepts_selected_and_stales_only_related(tmp_path):
    memory, _, media = _memory(tmp_path)
    try:
        selected = _suggestion(memory)
        related_source = _suggestion(memory, "person_001", "person_003")
        related_candidate = _suggestion(memory, "person_004", "person_001")
        unrelated = _suggestion(memory, "person_003", "person_004")
        media_before = _media_hashes(media)

        result = memory.resolve_identity_review(
            str(selected),
            IdentityReviewDecision.ACCEPT,
            reason="visual supervisor confirmation",
            decision_source="shift-lead",
        )

        assert result.source_person_id == "person_001"
        assert result.target_person_id == "person_002"
        assert result.status == "accepted"
        assert result.idempotent_replay is False
        assert result.target_embedding_count_before == 5
        assert result.target_embedding_count_after == 8
        assert result.staled_suggestion_count == 2
        statuses = dict(memory._conn.execute(
            "SELECT id, status FROM identity_match_suggestions"
        ).fetchall())
        assert statuses == {
            selected: "accepted",
            related_source: "stale",
            related_candidate: "stale",
            unrelated: "pending",
        }
        source = memory._conn.execute(
            "SELECT is_active, merged_into_person_id, embedding_count FROM persons WHERE person_id='person_001'"
        ).fetchone()
        assert tuple(source) == (0, "person_002", 3)
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0] == 1
        assert memory._conn.execute(
            "SELECT reviewed_by FROM identity_match_suggestions WHERE id=?",
            (selected,),
        ).fetchone()[0] == "shift-lead"
        assert _media_hashes(media) == media_before
        assert memory.query_by_face((0, 1, 0), threshold=-1)[0]["person_id"] != "person_001"
        assert [tuple(row) for row in memory._conn.execute("PRAGMA foreign_key_check")] == []
    finally:
        memory.close()


def test_reject_changes_only_selected_suggestion(tmp_path):
    memory, _, media = _memory(tmp_path)
    try:
        selected = _suggestion(memory)
        other = _suggestion(memory, "person_001", "person_003")
        before_people = [tuple(row) for row in memory._conn.execute("SELECT * FROM persons ORDER BY person_id")]
        media_before = _media_hashes(media)
        result = memory.resolve_identity_review(selected, "reject", decision_source="lead")
        assert result.status == "rejected"
        assert result.audit_id is None
        assert memory._conn.execute(
            "SELECT status FROM identity_match_suggestions WHERE id=?", (other,)
        ).fetchone()[0] == "pending"
        assert [tuple(row) for row in memory._conn.execute("SELECT * FROM persons ORDER BY person_id")] == before_people
        assert memory._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 0
        assert _media_hashes(media) == media_before
    finally:
        memory.close()


def test_accept_and_reject_replay_survive_restart_without_churn(tmp_path):
    memory, database, media = _memory(tmp_path)
    accepted = _suggestion(memory)
    rejected = _suggestion(memory, "person_003", "person_004")
    memory.resolve_identity_review(accepted, "accept")
    memory.resolve_identity_review(rejected, "reject")
    before = _snapshot(memory)
    memory.close()

    reopened = GlobalMemory(database, media_root=media)
    try:
        accept_replay = reopened.resolve_identity_review(accepted, "accept")
        reject_replay = reopened.resolve_identity_review(rejected, "reject")
        assert accept_replay.idempotent_replay is True
        assert reject_replay.idempotent_replay is True
        assert _snapshot(reopened) == before
    finally:
        reopened.close()


def test_conflicting_and_stale_decisions_are_typed(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        accepted = _suggestion(memory)
        rejected = _suggestion(memory, "person_003", "person_004")
        stale = _suggestion(memory, "person_004", "person_002", status="stale")
        memory.resolve_identity_review(accepted, "accept")
        memory.resolve_identity_review(rejected, "reject")
        with pytest.raises(ReviewSuggestionConflictError):
            memory.resolve_identity_review(accepted, "reject")
        with pytest.raises(ReviewSuggestionConflictError):
            memory.resolve_identity_review(rejected, "accept")
        with pytest.raises(ReviewSuggestionStaleError):
            memory.resolve_identity_review(stale, "reject")
    finally:
        memory.close()


@pytest.mark.parametrize("corruption", ["redirect", "audit", "target"])
def test_accepted_corruption_is_rejected(tmp_path, corruption):
    memory, _, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory)
        memory.resolve_identity_review(review_id, "accept")
        if corruption == "redirect":
            memory._conn.execute(
                "UPDATE persons SET merged_into_person_id='person_003' WHERE person_id='person_001'"
            )
        elif corruption == "audit":
            memory._conn.execute("DROP TRIGGER trg_identity_merge_audit_no_update")
            memory._conn.execute(
                "UPDATE identity_merge_audit SET decision_source='other'"
            )
        else:
            memory._conn.execute(
                "UPDATE persons SET is_active=0 WHERE person_id='person_002'"
            )
        with pytest.raises(ReviewSuggestionIntegrityError):
            memory.resolve_identity_review(review_id, "accept")
    finally:
        memory.close()


def test_rejected_review_with_review_audit_is_integrity_error(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory, status="rejected")
        source = memory._conn.execute("SELECT * FROM persons WHERE person_id='person_001'").fetchone()
        target = memory._conn.execute("SELECT * FROM persons WHERE person_id='person_002'").fetchone()
        memory._insert_person_merge_audit(
            source=source,
            target=target,
            reason="corrupt",
            decision_source=f"identity_review:{review_id}",
            source_embedding_count=3,
            created_at="2026-07-20",
        )
        with pytest.raises(ReviewSuggestionIntegrityError):
            memory.resolve_identity_review(review_id, "reject")
    finally:
        memory.close()


def _insert_rejected_corruption_audit(
    memory,
    *,
    source_id,
    target_id,
    decision_source,
):
    source = memory._conn.execute(
        "SELECT * FROM persons WHERE person_id=?", (source_id,)
    ).fetchone()
    target = memory._conn.execute(
        "SELECT * FROM persons WHERE person_id=?", (target_id,)
    ).fetchone()
    memory._insert_person_merge_audit(
        source=source,
        target=target,
        reason="persisted contradiction",
        decision_source=decision_source,
        source_embedding_count=int(source["embedding_count"]),
        created_at="2026-07-20T13:00:00",
    )


def _corrupt_rejected_review(memory, review_id, corruption):
    if corruption == "merged_original_candidate":
        memory.merge_persons(
            "person_001",
            "person_002",
            reason="later merge into rejected candidate",
            decision_source="manual-supervisor",
        )
        return (0, "person_002")
    if corruption == "merged_other_target":
        memory.merge_persons(
            "person_001",
            "person_003",
            reason="later merge into another candidate",
            decision_source="manual-supervisor",
        )
        return (0, "person_003")
    if corruption == "inactive_without_redirect":
        memory._conn.execute(
            "UPDATE persons SET is_active=0 WHERE person_id='person_001'"
        )
        return (0, None)
    if corruption == "redirect_without_audit":
        memory._conn.execute(
            "UPDATE persons SET is_active=0, merged_into_person_id='person_002' "
            "WHERE person_id='person_001'"
        )
        return (0, "person_002")
    if corruption == "candidate_inactive":
        memory._conn.execute(
            "UPDATE persons SET is_active=0 WHERE person_id='person_002'"
        )
        return (1, None)
    if corruption == "review_linked_audit":
        _insert_rejected_corruption_audit(
            memory,
            source_id="person_003",
            target_id="person_004",
            decision_source=f"identity_review:{review_id}",
        )
        return (1, None)
    if corruption == "unrelated_source_audit":
        _insert_rejected_corruption_audit(
            memory,
            source_id="person_001",
            target_id="person_003",
            decision_source="unrelated-manual-merge",
        )
        return (1, None)
    raise AssertionError(f"unknown corruption {corruption}")


@pytest.mark.parametrize(
    "corruption",
    [
        "merged_original_candidate",
        "merged_other_target",
        "inactive_without_redirect",
        "redirect_without_audit",
        "candidate_inactive",
        "review_linked_audit",
        "unrelated_source_audit",
    ],
)
def test_corrupt_rejected_review_fails_closed_after_reopen_without_churn(
    tmp_path,
    corruption,
):
    memory, database, media = _memory(tmp_path)
    review_id = _suggestion(memory)
    memory.resolve_identity_review(review_id, "reject")
    media_before = _media_hashes(media)
    expected_source = _corrupt_rejected_review(memory, review_id, corruption)
    before = _snapshot(memory)
    assert memory._conn.execute(
        "SELECT status FROM identity_match_suggestions WHERE id=?", (review_id,)
    ).fetchone()[0] == "rejected"
    assert tuple(memory._conn.execute(
        "SELECT is_active, merged_into_person_id FROM persons "
        "WHERE person_id='person_001'"
    ).fetchone()) == expected_source
    memory.close()

    for operation in ("get", "replay"):
        reopened = GlobalMemory(database, media_root=media)
        try:
            before_operation = _snapshot(reopened)
            with pytest.raises(ReviewSuggestionIntegrityError):
                if operation == "get":
                    reopened.get_identity_review(review_id)
                else:
                    reopened.resolve_identity_review(review_id, "reject")
            assert before_operation == before
            _assert_clean(reopened, database, before)
            assert _media_hashes(media) == media_before
        finally:
            reopened.close()


def test_read_only_queries_work_and_decisions_fail_before_transaction(tmp_path):
    memory, database, media = _memory(tmp_path)
    review_id = _suggestion(memory)
    memory.close()
    read_only = GlobalMemory(database, read_only=True, media_root=media)
    try:
        assert read_only.list_pending_identity_reviews()[0].suggestion_id == str(review_id)
        assert read_only.get_identity_review(review_id).suggestion.status == "pending"
        with pytest.raises(ReadOnlyGlobalMemoryError):
            read_only.resolve_identity_review(review_id, "accept")
        assert read_only._conn.in_transaction is False
    finally:
        read_only.close()


@pytest.mark.parametrize(
    "seam",
    [
        "_insert_person_merge_audit",
        "_update_merge_target_embedding",
        "_stale_merge_suggestions",
        "_deactivate_merge_source",
        "_redirect_merge_source",
        "_update_identity_review_status",
        "_before_review_commit",
    ],
)
def test_accept_failure_injection_rolls_back_every_table_and_releases_lock(
    tmp_path,
    monkeypatch,
    seam,
):
    memory, database, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory)
        _suggestion(memory, "person_001", "person_003")
        before = _snapshot(memory)

        def fail(*_args, **_kwargs):
            raise RuntimeError(f"injected {seam}")

        monkeypatch.setattr(memory, seam, fail)
        with pytest.raises(RuntimeError, match="injected"):
            memory.resolve_identity_review(review_id, "accept")
        _assert_clean(memory, database, before)
    finally:
        memory.close()


def test_reject_status_failure_rolls_back_and_releases_lock(tmp_path, monkeypatch):
    memory, database, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory)
        before = _snapshot(memory)

        def fail(*_args, **_kwargs):
            raise RuntimeError("injected rejection status")

        monkeypatch.setattr(memory, "_update_identity_review_status", fail)
        with pytest.raises(RuntimeError, match="rejection status"):
            memory.resolve_identity_review(review_id, "reject")
        _assert_clean(memory, database, before)
    finally:
        memory.close()


def _concurrent_decisions(database, media, review_ids_and_decisions):
    barrier = threading.Barrier(len(review_ids_and_decisions))
    outcomes = []
    lock = threading.Lock()

    def decide(review_id, decision):
        memory = GlobalMemory(database, media_root=media)
        memory._conn.execute("PRAGMA busy_timeout=3000")
        try:
            barrier.wait(timeout=3)
            try:
                value = memory.resolve_identity_review(review_id, decision)
                outcome = ("success", value)
            except BaseException as exc:
                outcome = ("error", exc)
            with lock:
                outcomes.append(outcome)
        finally:
            memory.close()

    threads = [threading.Thread(target=decide, args=item) for item in review_ids_and_decisions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
        assert not thread.is_alive()
    return outcomes


def test_concurrent_same_accept_blends_once_and_replays(tmp_path):
    seed, database, media = _memory(tmp_path)
    review_id = _suggestion(seed)
    seed.close()
    outcomes = _concurrent_decisions(database, media, [(review_id, "accept")] * 2)
    assert [kind for kind, _ in outcomes].count("success") == 2
    assert sorted(value.idempotent_replay for _, value in outcomes) == [False, True]
    verify = GlobalMemory(database, media_root=media)
    try:
        assert verify._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 1
        assert verify._conn.execute("SELECT embedding_count FROM persons WHERE person_id='person_002'").fetchone()[0] == 8
    finally:
        verify.close()


def test_concurrent_accept_reject_has_one_winner_and_one_conflict(tmp_path):
    seed, database, media = _memory(tmp_path)
    review_id = _suggestion(seed)
    seed.close()
    outcomes = _concurrent_decisions(
        database,
        media,
        [(review_id, "accept"), (review_id, "reject")],
    )
    assert [kind for kind, _ in outcomes].count("success") == 1
    errors = [value for kind, value in outcomes if kind == "error"]
    assert len(errors) == 1 and isinstance(errors[0], ReviewSuggestionConflictError)


def test_concurrent_competing_candidates_merges_at_most_once(tmp_path):
    seed, database, media = _memory(tmp_path)
    first = _suggestion(seed, "person_001", "person_002")
    second = _suggestion(seed, "person_001", "person_003")
    seed.close()
    outcomes = _concurrent_decisions(
        database,
        media,
        [(first, "accept"), (second, "accept")],
    )
    assert [kind for kind, _ in outcomes].count("success") == 1
    errors = [value for kind, value in outcomes if kind == "error"]
    assert len(errors) == 1 and isinstance(
        errors[0], (ReviewSuggestionStaleError, ReviewSuggestionConflictError)
    )
    verify = GlobalMemory(database, media_root=media)
    try:
        assert verify._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 1
        assert verify._conn.execute("SELECT COUNT(*) FROM persons WHERE person_id='person_001' AND is_active=0").fetchone()[0] == 1
    finally:
        verify.close()


def test_concurrent_reject_replays_without_person_mutation(tmp_path):
    seed, database, media = _memory(tmp_path)
    review_id = _suggestion(seed)
    people_before = [tuple(row) for row in seed._conn.execute("SELECT * FROM persons ORDER BY person_id")]
    seed.close()
    outcomes = _concurrent_decisions(database, media, [(review_id, "reject")] * 2)
    assert [kind for kind, _ in outcomes].count("success") == 2
    assert sorted(value.idempotent_replay for _, value in outcomes) == [False, True]
    verify = GlobalMemory(database, media_root=media)
    try:
        assert [tuple(row) for row in verify._conn.execute("SELECT * FROM persons ORDER BY person_id")] == people_before
        assert verify._conn.execute("SELECT COUNT(*) FROM identity_merge_audit").fetchone()[0] == 0
    finally:
        verify.close()


def test_merge_audit_is_immutable_and_source_evidence_remains_owned(tmp_path):
    memory, _, _ = _memory(tmp_path)
    try:
        review_id = _suggestion(memory)
        memory.resolve_identity_review(review_id, "accept")
        with pytest.raises(sqlite3.DatabaseError):
            memory._conn.execute("UPDATE identity_merge_audit SET reason='changed'")
        assert memory._conn.execute(
            "SELECT COUNT(*) FROM appearances WHERE person_id='person_001'"
        ).fetchone()[0] == 1
        detail = memory.get_identity_review(review_id)
        assert {row["original_person_id"] for row in detail.source_appearances} == {"person_001"}
        assert {row["original_person_id"] for row in detail.candidate_appearances} == {
            "person_001",
            "person_002",
        }
    finally:
        memory.close()
