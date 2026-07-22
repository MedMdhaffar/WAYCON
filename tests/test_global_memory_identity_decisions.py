from __future__ import annotations

import inspect
from pathlib import Path
import sqlite3

import numpy as np
import pytest

import forensics.global_memory as global_memory_package
from forensics.global_memory import GlobalMemory
from forensics.global_memory import config as global_memory_config
from forensics.global_memory.identity_policy import (
    IdentityCandidate,
    IdentityDecisionReason,
    IdentityDecisionType,
    IdentityPolicyConfig,
    IdentityPolicyInputError,
    IdentityRegistrationResult,
)
import forensics.person_creation.live_analysis as live_analysis_module
import forensics.person_creation.nodes.finalize as finalize_module
import forensics.person_creation.nodes.process_live_stream as live_stream_module
import forensics.person_creation.service as service_module
import forensics.person_creation.tools.add_face_photos as phone_photo_module


POLICY = IdentityPolicyConfig()


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _candidate_vector(similarity: float, *, negative: bool = False) -> np.ndarray:
    side = float(np.sqrt(1.0 - similarity * similarity))
    return _unit([similarity, -side if negative else side, 0.0])


def _insert_candidate(
    memory: GlobalMemory,
    person_id: str,
    embedding,
    *,
    embedding_count: int = 4,
    active: bool = True,
) -> None:
    vector = memory._normalize_embedding(embedding)
    memory._conn.execute(
        """INSERT INTO persons(
            person_id, name, embedding, embedding_count,
            enrolled_at, updated_at, cameras, is_active
        ) VALUES (?, ?, ?, ?, '2026-07-01', '2026-07-01', '[]', ?)""",
        (
            person_id,
            person_id.replace("_", " ").title(),
            vector.astype(np.float32).tobytes(),
            embedding_count,
            int(active),
        ),
    )
    number = int(person_id.rsplit("_", 1)[-1])
    memory._conn.execute(
        "UPDATE counters SET value=MAX(value, ?) WHERE key='person_count'",
        (number,),
    )


def _profile(media_root: Path, embedding, name: str, *, face_count: int = 2) -> dict:
    face_paths = []
    sharpness = {}
    for index in range(face_count):
        path = media_root / "session" / name / "face_crops" / f"face-{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"face-{name}-{index}".encode())
        face_paths.append(str(path))
        sharpness[str(path)] = float(100 - index)
    body = media_root / "session" / name / "body_crops" / "body.jpg"
    body.parent.mkdir(parents=True, exist_ok=True)
    body.write_bytes(f"body-{name}".encode())
    return {
        "face_embedding": np.asarray(embedding, dtype=np.float32).tolist(),
        "face_crops": face_paths,
        "face_crop_sharpness": sharpness,
        "best_body_crops": [str(body)],
        "body_crop_sharpness": {str(body): 55.0},
        "appearance": {
            "date": "2026-07-20",
            "clothing_status": "ok",
            "top": "black jacket",
            "bottom": "blue jeans",
            "shoes": "boots",
            "full": "black jacket and blue jeans",
        },
        "appearance_signals": {
            "color": {"top": "black", "bottom": "blue"}
        },
        "video_sources": ["masked-camera-1"],
        "cameras": ["camera-1"],
    }


def _count(memory: GlobalMemory, table: str) -> int:
    return int(memory._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _logical_snapshot(memory: GlobalMemory) -> dict[str, list[tuple]]:
    tables = (
        "persons",
        "appearances",
        "person_gallery",
        "recognition_log",
        "identity_match_suggestions",
        "identity_merge_audit",
        "counters",
        "sqlite_sequence",
    )
    return {
        table: [tuple(row) for row in memory._conn.execute(
            f'SELECT * FROM "{table}" ORDER BY rowid'
        ).fetchall()]
        for table in tables
    }


def _install_failing_recognition_trigger(memory: GlobalMemory) -> None:
    memory._conn.execute(
        """
        CREATE TRIGGER fail_recognition_insert
        BEFORE INSERT ON recognition_log
        BEGIN
            SELECT RAISE(ABORT, 'injected recognition failure');
        END
        """
    )


def _assert_failed_transaction_is_clean(
    memory: GlobalMemory,
    before: dict[str, list[tuple]],
) -> None:
    assert _logical_snapshot(memory) == before
    assert list(memory._conn.execute("PRAGMA foreign_key_check")) == []
    assert memory._conn.in_transaction is False

    second = sqlite3.connect(memory.db_path, timeout=0.25, isolation_level=None)
    try:
        second.execute("PRAGMA busy_timeout=250")
        second.execute("BEGIN IMMEDIATE")
        second.execute("ROLLBACK")
    finally:
        second.close()


def test_new_person_below_minimum_persists_evidence_without_suggestion(tmp_path):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", _candidate_vector(0.67))
        profile = _profile(root, [1.0, 0.0, 0.0], "new-person")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=3,
            configuration=POLICY,
        )

        assert result.decision is IdentityDecisionType.NEW_PERSON
        assert result.reason is IdentityDecisionReason.BELOW_MINIMUM_SIMILARITY
        assert result.person_id == "person_002"
        assert result.suggestion_id is None
        assert _count(memory, "persons") == 2
        assert _count(memory, "appearances") == 1
        assert _count(memory, "person_gallery") == 3
        assert _count(memory, "recognition_log") == 1
        assert _count(memory, "identity_match_suggestions") == 0
        assert _count(memory, "identity_merge_audit") == 0
    finally:
        memory.close()


def test_review_range_creates_separate_person_and_exact_pending_suggestion(tmp_path):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", _candidate_vector(0.73))
        profile = _profile(root, [1.0, 0.0, 0.0], "review-range")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=3,
            configuration=POLICY,
        )
        suggestion = memory._conn.execute(
            "SELECT * FROM identity_match_suggestions"
        ).fetchone()

        assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
        assert result.reason is IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS
        assert result.person_id == "person_002"
        assert result.suggestion_id == suggestion["id"]
        assert suggestion["source_person_id"] == "person_002"
        assert suggestion["candidate_person_id"] == "person_001"
        assert suggestion["similarity"] == pytest.approx(result.top_similarity)
        assert suggestion["second_similarity"] is None
        assert suggestion["margin"] is None
        assert suggestion["reason"] == "similarity_between_thresholds"
        assert suggestion["status"] == "pending"
        assert suggestion["reviewed_at"] is None
        assert suggestion["reviewed_by"] is None
        assert _count(memory, "persons") == 2
        assert _count(memory, "identity_match_suggestions") == 1
        assert _count(memory, "identity_merge_audit") == 0
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("low_confidence", "observations", "second_similarity", "reason"),
    [
        (True, 3, None, IdentityDecisionReason.LOW_CONFIDENCE_CLUSTER),
        (False, 2, None, IdentityDecisionReason.INSUFFICIENT_FACE_OBSERVATIONS),
        (False, 3, 0.88, IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL),
    ],
)
def test_high_similarity_safety_gates_create_nonblocking_review(
    tmp_path,
    low_confidence,
    observations,
    second_similarity,
    reason,
):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", _candidate_vector(0.90))
        if second_similarity is not None:
            _insert_candidate(
                memory,
                "person_002",
                _candidate_vector(second_similarity, negative=True),
            )
        before_people = _count(memory, "persons")
        profile = _profile(root, [1.0, 0.0, 0.0], f"review-{reason.value}")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=observations,
            low_confidence=low_confidence,
            configuration=POLICY,
        )

        assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
        assert result.reason is reason
        assert result.suggestion_id is not None
        assert _count(memory, "persons") == before_people + 1
        assert _count(memory, "identity_match_suggestions") == 1
        assert _count(memory, "identity_merge_audit") == 0
    finally:
        memory.close()


def test_exact_margin_boundary_attaches_existing_without_new_rows(tmp_path, monkeypatch):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", [1.0, 0.0, 0.0])
        _insert_candidate(memory, "person_002", [0.0, 1.0, 0.0])
        monkeypatch.setattr(
            memory,
            "_rank_active_identity_candidates",
            lambda _embedding: (
                IdentityCandidate("person_001", 0.90),
                IdentityCandidate("person_002", 0.87),
            ),
        )
        profile = _profile(root, [1.0, 0.0, 0.0], "margin-boundary")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=3,
            configuration=POLICY,
        )

        assert result.margin >= 0.03
        assert result.decision is IdentityDecisionType.ATTACH_EXISTING
        assert result.person_id == "person_001"
        assert _count(memory, "persons") == 2
        assert _count(memory, "identity_match_suggestions") == 0
    finally:
        memory.close()


def test_inactive_candidate_is_neither_ranked_attached_nor_suggested(tmp_path):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", [1.0, 0.0, 0.0], active=False)
        _insert_candidate(memory, "person_002", _candidate_vector(0.73), active=True)
        profile = _profile(root, [1.0, 0.0, 0.0], "inactive-filter")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=3,
            configuration=POLICY,
        )

        assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
        assert result.top_candidate_person_id == "person_002"
        assert result.second_candidate_person_id is None
        suggestion = memory._conn.execute(
            "SELECT candidate_person_id FROM identity_match_suggestions"
        ).fetchone()
        assert suggestion[0] == "person_002"
    finally:
        memory.close()


def test_equal_similarity_tie_is_ordered_by_person_id(tmp_path):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        vector = _candidate_vector(0.90)
        _insert_candidate(memory, "person_002", vector)
        _insert_candidate(memory, "person_001", vector)
        profile = _profile(root, [1.0, 0.0, 0.0], "tie")

        result = memory.register_with_identity_policy(
            profile,
            observation_count=3,
            configuration=POLICY,
        )

        assert result.top_candidate_person_id == "person_001"
        assert result.second_candidate_person_id == "person_002"
        assert result.reason is IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL
    finally:
        memory.close()


def test_suggestion_insertion_failure_rolls_back_all_identity_evidence(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        _insert_candidate(memory, "person_001", _candidate_vector(0.73))
        before = _logical_snapshot(memory)
        original = memory._insert_identity_suggestion

        def insert_then_fail(**kwargs):
            original(**kwargs)
            raise RuntimeError("injected suggestion failure")

        monkeypatch.setattr(memory, "_insert_identity_suggestion", insert_then_fail)
        profile = _profile(root, [1.0, 0.0, 0.0], "rollback")

        with pytest.raises(RuntimeError, match="injected suggestion failure"):
            memory.register_with_identity_policy(
                profile,
                observation_count=3,
                configuration=POLICY,
            )

        assert _logical_snapshot(memory) == before
        assert memory._conn.in_transaction is False
    finally:
        memory.close()


@pytest.mark.parametrize(
    "outcome",
    ["new_person", "review_required", "attach_existing"],
)
def test_real_recognition_failure_rolls_back_complete_phase3e_outcome(
    tmp_path,
    outcome,
):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        if outcome == "review_required":
            _insert_candidate(memory, "person_001", _candidate_vector(0.73))
        elif outcome == "attach_existing":
            _insert_candidate(
                memory,
                "person_001",
                [1.0, 0.0, 0.0],
                embedding_count=4,
            )
        _install_failing_recognition_trigger(memory)
        before = _logical_snapshot(memory)
        profile = _profile(root, [1.0, 0.0, 0.0], f"strict-{outcome}")

        with pytest.raises(sqlite3.IntegrityError, match="injected recognition failure"):
            memory.register_with_identity_policy(
                profile,
                observation_count=3,
                configuration=POLICY,
            )

        _assert_failed_transaction_is_clean(memory, before)
    finally:
        memory.close()


def test_legacy_register_keeps_best_effort_recognition_logging(tmp_path):
    root = tmp_path / "person_db"
    memory = GlobalMemory(tmp_path / "memory.db", media_root=root)
    try:
        assert global_memory_config.SIMILARITY_THRESHOLD == 0.60
        _insert_candidate(
            memory,
            "person_001",
            _candidate_vector(0.65),
            embedding_count=4,
        )
        _install_failing_recognition_trigger(memory)
        profile = _profile(root, [1.0, 0.0, 0.0], "legacy-best-effort")

        person_id = memory.register(profile)

        assert person_id == "person_001"
        assert isinstance(person_id, str)
        assert _count(memory, "persons") == 1
        assert memory._conn.execute(
            "SELECT embedding_count FROM persons WHERE person_id='person_001'"
        ).fetchone()[0] == 6
        assert _count(memory, "recognition_log") == 0
        assert _count(memory, "identity_match_suggestions") == 0
    finally:
        memory.close()


def test_strong_attachment_matches_legacy_embedding_and_evidence_behavior(tmp_path):
    root = tmp_path / "person_db"
    face_profile = _profile(root, [0.98, 0.20, 0.0], "compatibility")
    legacy = GlobalMemory(tmp_path / "legacy.db", media_root=root)
    policy = GlobalMemory(tmp_path / "policy.db", media_root=root)
    try:
        for memory in (legacy, policy):
            _insert_candidate(
                memory,
                "person_001",
                [1.0, 0.0, 0.0],
                embedding_count=4,
            )

        legacy_person_id = legacy.register(face_profile)
        result = policy.register_with_identity_policy(
            face_profile,
            observation_count=3,
            configuration=POLICY,
        )
        legacy_person = legacy._conn.execute(
            "SELECT embedding, embedding_count FROM persons WHERE person_id='person_001'"
        ).fetchone()
        policy_person = policy._conn.execute(
            "SELECT embedding, embedding_count FROM persons WHERE person_id='person_001'"
        ).fetchone()

        assert legacy_person_id == result.person_id == "person_001"
        assert result.decision is IdentityDecisionType.ATTACH_EXISTING
        assert bytes(legacy_person["embedding"]) == bytes(policy_person["embedding"])
        assert legacy_person["embedding_count"] == policy_person["embedding_count"]
        assert [tuple(r) for r in legacy._conn.execute("SELECT * FROM appearances")] == [
            tuple(r) for r in policy._conn.execute("SELECT * FROM appearances")
        ]
        assert [tuple(r)[1:] for r in legacy._conn.execute("SELECT * FROM person_gallery ORDER BY id")] == [
            tuple(r)[1:] for r in policy._conn.execute("SELECT * FROM person_gallery ORDER BY id")
        ]
        legacy_log = legacy._conn.execute(
            "SELECT person_id,event_type,similarity,embedding_count_before,"
            "embedding_count_after,video_sources,best_face_crop FROM recognition_log"
        ).fetchall()
        policy_log = policy._conn.execute(
            "SELECT person_id,event_type,similarity,embedding_count_before,"
            "embedding_count_after,video_sources,best_face_crop FROM recognition_log"
        ).fetchall()
        assert [tuple(r) for r in legacy_log] == [tuple(r) for r in policy_log]
        assert _count(policy, "identity_match_suggestions") == 0
        assert _count(policy, "identity_merge_audit") == 0
    finally:
        legacy.close()
        policy.close()


@pytest.mark.parametrize(
    "embedding",
    [[], [[1.0, 0.0]], [float("nan"), 0.0], [0.0, 0.0]],
)
def test_invalid_incoming_embeddings_are_rejected_before_identity_write(
    tmp_path,
    embedding,
):
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "person_db")
    try:
        with pytest.raises(IdentityPolicyInputError):
            memory.register_with_identity_policy(
                {"face_embedding": embedding},
                observation_count=3,
                configuration=POLICY,
            )
        assert _count(memory, "persons") == 0
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("label", "stored_embeddings"),
    [
        ("nan_top", [[float("nan"), 0.0, 0.0]]),
        ("infinity_top", [[float("inf"), 0.0, 0.0]]),
        (
            "nan_second_and_margin",
            [[1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
        ),
        (
            "infinity_second_and_margin",
            [[1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]],
        ),
    ],
)
def test_operational_policy_rejects_nonfinite_stored_similarities(
    tmp_path,
    label,
    stored_embeddings,
):
    del label
    memory = GlobalMemory(tmp_path / "memory.db", media_root=tmp_path / "person_db")
    try:
        for index, embedding in enumerate(stored_embeddings, start=1):
            memory._conn.execute(
                """
                INSERT INTO persons(
                    person_id, name, embedding, embedding_count,
                    enrolled_at, updated_at, cameras, is_active
                ) VALUES (?, ?, ?, 1, '2026-07-01', '2026-07-01', '[]', 1)
                """,
                (
                    f"person_{index:03d}",
                    f"Person {index:03d}",
                    np.asarray(embedding, dtype=np.float32).tobytes(),
                ),
            )
        before = _logical_snapshot(memory)

        with pytest.raises(IdentityPolicyInputError, match="finite"):
            memory.register_with_identity_policy(
                {"face_embedding": [1.0, 0.0, 0.0]},
                observation_count=3,
                configuration=POLICY,
            )

        assert _logical_snapshot(memory) == before
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("profile", "expected_observations", "expected_low_confidence"),
    [
        (
            {
                "face_embedding": [1.0, 0.0, 0.0],
                "cluster_face_count": 4,
                "low_confidence": True,
                "face_crops": [],
            },
            4,
            True,
        ),
        (
            {
                "face_embedding": [1.0, 0.0, 0.0],
                "face_crops": ["accepted-a.jpg", "accepted-b.jpg"],
            },
            2,
            False,
        ),
    ],
)
def test_canonical_finalize_uses_policy_and_returns_without_review_blocking(
    tmp_path,
    monkeypatch,
    profile,
    expected_observations,
    expected_low_confidence,
):
    calls = []
    review_result = IdentityRegistrationResult(
        person_id="person_777",
        decision=IdentityDecisionType.REVIEW_REQUIRED,
        reason=IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS,
        suggestion_id=42,
        top_candidate_person_id="person_001",
        top_similarity=0.73,
        second_candidate_person_id=None,
        second_similarity=None,
        margin=None,
        observation_count=expected_observations,
        low_confidence=expected_low_confidence,
        configuration=POLICY,
    )

    class FakeMemory:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def register_with_identity_policy(self, received, **kwargs):
            prepare = kwargs.pop("prepare_profile_for_person")
            prepare(review_result.person_id, received)
            calls.append((received, kwargs))
            return review_result

        def get_person(self, _person_id):
            return {"name": "Person 777"}

        def update_crop_paths(self, _person_id, _profile):
            return None

        def referenced_media_paths(self, _person_id):
            return set()

        def close(self):
            return None

    root = tmp_path / "person_db"
    output = root / "session"
    root.mkdir()
    for path in profile.get("face_crops", []):
        (root / path).write_bytes(b"accepted-face")
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(root))
    monkeypatch.setattr(global_memory_package, "GlobalMemory", FakeMemory)
    state = {
        "output_dir": str(output),
        "per_cluster_profiles": {0: dict(profile)},
        "identity_clusters": [{"cluster_id": 0}],
        "unresolved_faces": [],
        "unattached_bodies": [],
    }

    result = finalize_module.finalize(state)

    assert result["profile"]["id"] == "person_777"
    assert len(calls) == 1
    assert calls[0][1] == {
        "observation_count": expected_observations,
        "low_confidence": expected_low_confidence,
    }
    assert review_result.decision is IdentityDecisionType.REVIEW_REQUIRED
    assert review_result.reason is IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS
    assert review_result.suggestion_id == 42


def test_phone_stop_and_service_paths_remain_unintegrated():
    """Only the rolling analysis lane may reach the identity policy.

    Phase 4 step 1 moved Phase 3E decisions into LiveRollingAnalysisSession so
    identities are persisted while capture is active. The phone, stream and
    service paths must still never register an identity themselves.
    """
    live_source = inspect.getsource(live_analysis_module)
    stream_source = inspect.getsource(live_stream_module)
    phone_source = inspect.getsource(phone_photo_module)
    service_source = inspect.getsource(service_module)

    # The rolling lane keeps its read-only preview handle *and* owns the
    # authoritative Phase 3E decision.
    assert "read_only=True" in live_source
    assert "register_with_identity_policy" in live_source
    # Live decisions must relocate evidence into the canonical person directory
    # inside the identity transaction, so no _staging path is ever persisted.
    assert "prepare_profile_for_person" in live_source
    assert "relocate_profile_media" in live_source

    assert "register_with_identity_policy" not in stream_source
    assert "register_with_identity_policy" not in phone_source
    assert "register_with_identity_policy" not in service_source
    assert "_stop_event" in stream_source
    assert "reconnect" in stream_source.lower()
