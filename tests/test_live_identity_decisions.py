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
from forensics.global_memory.identity_policy import (
    IdentityCandidate,
    IdentityPolicyConfig,
)
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


def _snapshot(
    version: int,
    paths: tuple[str, ...],
    *,
    chunk: int = 0,
    latency_timing: dict | None = None,
):
    embeddings = tuple(
        _record({
            "crop_path": path,
            "embedding": [1.0, float(index) / 1000.0],
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0 + index,
            "face_quality": {
                "quality_class": "high",
                "accepted_for_embedding": True,
                "immediate_confirmation_eligible": True,
            },
            **({
                "_latency_timing": dict(latency_timing),
            } if latency_timing is not None else {}),
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
        cluster=kwargs.pop("cluster", _cluster_all),
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


class _RankingMemory:
    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.rank_calls = 0

    def rank_identity_candidates(self, _embedding):
        self.rank_calls += 1
        return list(self.candidates)

    def close(self):
        return None


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


def test_first_valid_embedded_face_publishes_provisional_comparison(
    media_root,
    database,
):
    seeded = _seed_person(
        database,
        [1.0, 0.0],
        _face_paths(media_root, 3, prefix="first-face-seed"),
    )
    paths = _face_paths(media_root, 1, prefix="first-face")
    state = {
        "snapshot": _snapshot(
            1,
            paths,
            latency_timing={
                "source_frame_timestamp": "2026-07-23T10:00:00+00:00",
                "capture_monotonic": time.monotonic() - 0.05,
                "face_detected_monotonic": time.monotonic() - 0.04,
                "quality_accepted_monotonic": time.monotonic() - 0.03,
                "embedding_completed_monotonic": time.monotonic() - 0.02,
            },
        ),
    }
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        associate=_association,
        job_id="job-real-singleton-cluster",
        identity_decisions=True,
    )

    snapshot = _run(session, state, [(1, state["snapshot"])])
    identity = _identity(snapshot)

    assert identity["decision"] == "attach_existing"
    assert identity["reason"] == "strong_clear_match"
    assert identity["clustering_state"] == "unresolved"
    assert identity["state"] == "provisional"
    assert identity["provisional"] is True
    assert identity["persisted"] is False
    assert identity["observation_count"] == 1
    assert identity["candidate_person_id"] == seeded
    assert identity["candidate_similarity"] == 1.0
    assert identity["decision_version"] == 1
    assert identity["evidence_version"] == 1
    assert identity["evidence_signature"]
    assert identity["canonical_person_id"] is None
    assert identity["comparison_timestamp"]
    assert identity["latency_metrics"]["capture_to_face_ms"] >= 0
    assert identity["latency_metrics"]["capture_to_embedding_ms"] >= 0
    assert identity["latency_metrics"]["embedding_to_comparison_ms"] >= 0
    assert identity["latency_metrics"]["comparison_to_status_ms"] >= 0
    assert identity["latency_metrics"]["capture_to_status_ms"] >= 0
    assert _person_count(database) == 1
    assert len(session.identity_decisions()) == 0


def test_dbscan_noise_face_publishes_once_and_reconciles_to_same_card(
    media_root,
    database,
):
    first_paths = _face_paths(media_root, 1, prefix="noise-first")
    resolved_paths = first_paths + _face_paths(
        media_root,
        2,
        prefix="noise-resolved",
    )

    def unresolved_then_resolved(state):
        records = list(state["all_face_embeddings"])
        if len(records) == 1:
            return {
                "identity_clusters": [],
                "unresolved_faces": records,
            }
        return _cluster_all(state)

    class ChangingRankingMemory(_RankingMemory):
        def rank_identity_candidates(self, _embedding):
            self.rank_calls += 1
            if self.rank_calls == 1:
                return [IdentityCandidate("person_900", 0.9)]
            return []

    ranked = ChangingRankingMemory([])
    state = {"snapshot": _snapshot(1, first_paths)}
    session = _session(
        database,
        lambda: state["snapshot"],
        cluster=unresolved_then_resolved,
        decision_memory_factory=lambda _path: ranked,
    )
    session.start()
    try:
        session.request_version(1)
        first_snapshot = _wait_for(session, 1)
        first = _identity(first_snapshot)
        first_id = first["live_identity_id"]

        assert len(first_snapshot["live_identities"]) == 1
        assert first["clustering_state"] == "unresolved"
        assert first["observation_count"] == 1
        assert first["provisional"] is True
        assert first["persisted"] is False
        assert first["best_face_path"] == first_paths[0]
        assert first["comparison_timestamp"]
        assert first["evidence_version"] == 1
        assert ranked.rank_calls == 1
        assert _person_count(database) == 0

        state["snapshot"] = _snapshot(2, first_paths)
        session.request_version(2)
        duplicate_snapshot = _wait_for(session, 2)
        assert len(duplicate_snapshot["live_identities"]) == 1
        assert duplicate_snapshot["live_identities"][0]["live_identity_id"] == first_id
        assert ranked.rank_calls == 1

        state["snapshot"] = _snapshot(3, resolved_paths)
        session.request_version(3)
        resolved_snapshot = _wait_for(session, 3)
        resolved = _identity(resolved_snapshot)

        assert len(resolved_snapshot["live_identities"]) == 1
        assert resolved["live_identity_id"] == first_id
        assert resolved["clustering_state"] == "resolved"
        assert resolved["observation_count"] == 3
        assert resolved["evidence_version"] == 2
        assert resolved["provisional"] is True
        assert resolved["persisted"] is False
        assert ranked.rank_calls == 2
        assert _person_count(database) == 0
    finally:
        session.finish(3)

    assert not session.worker_alive


def test_two_independent_weak_noise_embeddings_create_no_cards(
    media_root,
    database,
):
    paths = _face_paths(media_root, 2, prefix="separate-noise")
    embeddings = tuple(
        _record({
            "crop_path": path,
            "embedding": vector,
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0,
        })
        for index, (path, vector) in enumerate(zip(
            paths,
            ([1.0, 0.0], [0.0, 1.0]),
        ))
    )
    faces = tuple(
        _record({
            "path": path,
            "frame_idx": index,
            "video": "camera-source",
            "bbox": [0, 0, 80, 80],
            "sharpness": 100.0,
        })
        for index, path in enumerate(paths)
    )
    actual_noise_snapshot = FrozenAnalysisSnapshot(
        version=1,
        last_completed_preprocessing_chunk=0,
        person_name="Live Subject",
        video_paths=("camera-source",),
        identity_clustering_config=_record({"eps": 0.4, "min_samples": 3}),
        quality_body_crops=(),
        quality_face_crops=faces,
        face_embeddings=embeddings,
        face_chunk_membership=tuple((path, 0) for path in paths),
    )
    ranked = _RankingMemory([])
    state = {"snapshot": actual_noise_snapshot}
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        associate=_association,
        job_id="job-actual-dbscan-noise",
        identity_decisions=True,
        decision_memory_factory=lambda _path: ranked,
    )

    snapshot = _run(session, state, [(1, state["snapshot"])])

    assert snapshot["live_identities"] == []
    assert snapshot["unresolved_embedding_count"] == 2
    assert snapshot["resolved_cluster_count"] == 0
    assert ranked.rank_calls == 2
    assert _person_count(database) == 0


def test_noise_memberships_merge_without_duplicate_live_card(media_root, database):
    initial_paths = _face_paths(media_root, 2, prefix="merge-noise")
    resolved_paths = initial_paths + _face_paths(
        media_root,
        1,
        prefix="merge-resolved",
    )

    def noise_then_cluster(state):
        records = list(state["all_face_embeddings"])
        if len(records) == 2:
            return {
                "identity_clusters": [],
                "unresolved_faces": records,
            }
        return _cluster_all(state)

    state = {"snapshot": _snapshot(1, initial_paths)}
    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        cluster=noise_then_cluster,
        associate=_association,
        job_id="job-noise-merge",
        identity_decisions=False,
    )
    session.start()
    try:
        session.request_version(1)
        unresolved = _wait_for(session, 1)
        assert unresolved["live_identities"] == []

        state["snapshot"] = _snapshot(2, resolved_paths)
        session.request_version(2)
        resolved = _wait_for(session, 2)

        assert len(resolved["live_identities"]) == 1
        assert resolved["live_identities"][0]["live_identity_id"] == "live_0001"
        assert resolved["live_identities"][0]["clustering_state"] == "resolved"
        assert resolved["retired_live_identity_ids"] == ["live_0002"]
        assert any(
            event["type"] == "identity_merged"
            and event["retained_live_id"] == "live_0001"
            and event["absorbed_live_ids"] == ["live_0002"]
            for event in resolved["live_recognition_events"]
        )
    finally:
        session.finish(2)


def test_retired_identity_reappears_without_duplicate_card_or_persistence(
    media_root,
    database,
):
    stable_paths = _face_paths(media_root, 6, prefix="reappearing")
    extra_paths = _face_paths(media_root, 2, prefix="temporary")
    state = {"snapshot": _snapshot(1, stable_paths)}

    def temporarily_missing_cluster(node_state):
        records = list(node_state["all_face_embeddings"])
        if len(records) == 7:
            return {"identity_clusters": [], "unresolved_faces": []}
        stable_records = records[:6]
        return {
            "identity_clusters": [{
                "cluster_id": 0,
                "face_records": stable_records,
                "representative_embedding": [1.0, 0.0],
                "face_count": len(stable_records),
                "confidence": 1.0,
                "low_confidence": False,
            }],
            "unresolved_faces": [],
        }

    session = _session(
        database,
        lambda: state["snapshot"],
        cluster=temporarily_missing_cluster,
        job_id="job-retired-recovery",
    )
    session.start()
    try:
        session.request_version(1)
        initial = _wait_for(session, 1)
        assert [
            item["live_identity_id"] for item in initial["live_identities"]
        ] == ["live_0001"]
        persisted_before = {
            table: _count(database, table)
            for table in (
                "persons",
                "person_gallery",
                "appearances",
                "recognition_log",
                "identity_evidence",
            )
        }

        state["snapshot"] = _snapshot(
            2,
            stable_paths + extra_paths[:1],
        )
        session.request_version(2)
        missing = _wait_for(session, 2)
        assert missing["live_identities"] == []
        assert missing["retired_live_identity_ids"] == ["live_0001"]

        state["snapshot"] = _snapshot(
            3,
            stable_paths + extra_paths,
        )
        session.request_version(3)
        reappeared = _wait_for(session, 3)

        assert [
            item["live_identity_id"] for item in reappeared["live_identities"]
        ] == ["live_0001"]
        assert reappeared["retired_live_identity_ids"] == []
        assert len(session.identity_decisions()) == 1
        assert {
            table: _count(database, table)
            for table in persisted_before
        } == persisted_before
    finally:
        session.finish(3)


class _CountingMemory:
    """Record every Global Memory call the rolling analysis worker makes."""

    def __init__(self, delegate):
        self._delegate = delegate
        self.calls: list[str] = []

    def __getattr__(self, name):
        attribute = getattr(self._delegate, name)
        if not callable(attribute):
            return attribute

        def recorded(*args, **kwargs):
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return recorded


def test_analysis_pass_issues_no_representative_diagnostic_memory_queries(
    media_root,
    database,
):
    """The medoid/mean comparison must not run inside the live analysis pass.

    It previously cost one ``get_person`` per enrolled person per identity on
    every rolling pass, so the read-only handle must now stay untouched.
    """
    for index in range(3):
        _seed_person(
            database,
            [1.0, float(index) / 10.0],
            _face_paths(media_root, 4, prefix=f"enrolled-{index}"),
        )

    paths = _face_paths(media_root, 4, prefix="diagnostic-free")
    state = {"snapshot": _snapshot(1, paths)}
    counters: list[_CountingMemory] = []

    def memory_factory(path):
        counter = _CountingMemory(GlobalMemory(str(path), read_only=True))
        counters.append(counter)
        return counter

    session = _session(
        database,
        lambda: state["snapshot"],
        memory_factory=memory_factory,
    )
    session.start()
    try:
        session.request_version(1)
        snapshot = _wait_for(session, 1)
    finally:
        session.finish(1)

    assert snapshot["live_identities"], "the pass must still produce an identity"
    assert counters, "the read-only Global Memory handle should still be opened"
    # ``close`` is worker shutdown, not a query.
    queries = [name for name in counters[0].calls if name != "close"]
    assert queries == [], (
        f"live analysis issued unexpected Global Memory queries: {queries}"
    )
    assert "list_all" not in counters[0].calls
    assert "get_person" not in counters[0].calls

    for identity in snapshot["live_identities"]:
        assert "representative_similarity_diagnostic" not in identity
        assert "comparisons" not in identity
        assert "medoid_similarity" not in identity
        assert "normalized_mean_similarity" not in identity


def test_representative_comparison_is_absent_from_live_analysis_module():
    """The comparison must live only in the offline tool."""
    from forensics.person_creation import live_analysis

    assert not hasattr(live_analysis, "representative_similarity_diagnostic")
    source = Path(live_analysis.__file__).read_text(encoding="utf-8")
    assert "compare_cluster_representatives" not in source.replace(
        "tools/compare_cluster_representatives.py", ""
    )


def test_one_face_publishes_second_candidate_and_margin(media_root, database):
    paths = _face_paths(media_root, 1, prefix="ranked-first-face")
    state = {"snapshot": _snapshot(1, paths)}
    ranked = _RankingMemory([
        IdentityCandidate("person_010", 0.91),
        IdentityCandidate("person_011", 0.72),
    ])
    session = _session(
        database,
        lambda: state["snapshot"],
        decision_memory_factory=lambda _path: ranked,
    )

    identity = _identity(_run(session, state, [(1, state["snapshot"])]))

    assert ranked.rank_calls == 1
    assert identity["observation_count"] == 1
    assert identity["candidate_person_id"] == "person_010"
    assert identity["candidate_similarity"] == 0.91
    assert identity["second_candidate_person_id"] == "person_011"
    assert identity["second_candidate_similarity"] == 0.72
    assert identity["margin"] == 0.19
    assert identity["provisional"] is True
    assert identity["persisted"] is False


def test_zero_embedded_faces_performs_no_comparison(media_root, database):
    state = {"snapshot": _snapshot(1, ())}
    opened = []
    session = _session(
        database,
        lambda: state["snapshot"],
        decision_memory_factory=lambda _path: opened.append(True),
    )

    snapshot = _run(session, state, [(1, state["snapshot"])])

    assert snapshot["live_identities"] == []
    assert opened == []
    assert _person_count(database) == 0


def test_distinct_face_recompares_but_replay_and_body_only_update_do_not(
    media_root,
    database,
):
    first = _face_paths(media_root, 1, prefix="comparison-version")
    second = first + _face_paths(media_root, 1, prefix="comparison-version-next")
    body_path = media_root / "session" / "_staging" / "body_crops" / "body.jpg"
    body_path.parent.mkdir(parents=True)
    body_path.write_bytes(b"body-evidence")
    state = {"snapshot": _snapshot(1, first)}
    ranked = _RankingMemory([
        IdentityCandidate("person_010", 0.91),
        IdentityCandidate("person_011", 0.72),
    ])
    association_calls = 0

    def association(_state):
        nonlocal association_calls
        association_calls += 1
        assignments = [] if association_calls < 4 else [{
            "body_crop_path": str(body_path),
            "body_sharpness": 100.0,
        }]
        return SimpleNamespace(cluster_assignments={0: assignments})

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        cluster=_cluster_all,
        associate=association,
        job_id="job-comparison-signature",
        identity_decisions=True,
        decision_memory_factory=lambda _path: ranked,
        policy_config=IdentityPolicyConfig(minimum_face_observations=2),
    )
    session.start()
    try:
        session.request_version(1)
        first_result = _identity(_wait_for(session, 1))
        assert ranked.rank_calls == 1
        assert first_result["evidence_version"] == 1

        state["snapshot"] = _snapshot(2, first)
        session.request_version(2)
        replay = _identity(_wait_for(session, 2))
        assert ranked.rank_calls == 1
        assert replay["evidence_version"] == 1

        state["snapshot"] = _snapshot(3, second)
        session.request_version(3)
        distinct = _identity(_wait_for(session, 3))
        assert ranked.rank_calls == 2
        assert distinct["evidence_version"] == 2
        assert distinct["observation_count"] == 2

        state["snapshot"] = _snapshot(4, second)
        session.request_version(4)
        body_only = _identity(_wait_for(session, 4))
        assert ranked.rank_calls == 2
        assert body_only["evidence_version"] == 2
        assert body_only["observation_count"] == 2
    finally:
        session.finish(4)


def test_weak_first_face_does_not_create_new_person(media_root, database):
    paths = _face_paths(media_root, 1, prefix="weak-first")
    state = {"snapshot": _snapshot(1, paths)}
    session = _session(database, lambda: state["snapshot"])

    identity = _identity(_run(session, state, [(1, state["snapshot"])]))

    assert identity["decision"] == "new_person"
    assert identity["provisional"] is True
    assert identity["persisted"] is False
    assert identity["observation_count"] == 1
    assert _person_count(database) == 0


def test_non_finite_embedding_performs_no_comparison(media_root, database):
    paths = _face_paths(media_root, 1, prefix="non-finite")
    state = {"snapshot": _snapshot(1, paths)}
    ranked = _RankingMemory([IdentityCandidate("person_010", 0.91)])

    def invalid_cluster(node_state):
        result = _cluster_all(node_state)
        result["identity_clusters"][0]["representative_embedding"] = [
            float("nan"),
            0.0,
        ]
        return result

    session = LiveRollingAnalysisSession(
        snapshot_provider=lambda: state["snapshot"],
        join_timeout_seconds=5.0,
        database_path=database,
        cluster=invalid_cluster,
        associate=_association,
        job_id="job-invalid-embedding",
        identity_decisions=True,
        decision_memory_factory=lambda _path: ranked,
    )

    identity = _identity(_run(session, state, [(1, state["snapshot"])]))

    assert ranked.rank_calls == 0
    assert identity["decision"] is None
    assert identity["state"] == "observing"
    assert identity["candidate_person_id"] is None
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
