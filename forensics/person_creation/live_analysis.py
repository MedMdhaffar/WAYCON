"""Latest-only rolling identity analysis for an active live session."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from forensics.identity_evidence import identity_evidence_key
from forensics.person_creation.live_identity import (
    CurrentMembership,
    IdentityAssignmentResult,
    assign_live_identities,
)
from forensics.person_creation.live_session import FrozenAnalysisSnapshot


_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)
_EVENT_LIMIT = 100


class LiveAnalysisLifecycleError(RuntimeError):
    """The rolling worker may still be accessing session evidence."""


@dataclass(frozen=True)
class _CommittedInputs:
    active_memberships: tuple[tuple[str, frozenset[str]], ...]
    membership_history: tuple[tuple[str, frozenset[str]], ...]
    retired_live_ids: frozenset[str]
    next_live_number: int
    memory_matches: tuple[tuple[str, dict | None], ...]
    events: tuple[dict, ...]
    event_keys: tuple[str, ...]
    next_event_number: int


@dataclass(frozen=True)
class _AnalysisResult:
    version: int
    embedding_count: int
    body_count: int
    unresolved_embedding_count: int
    resolved_cluster_count: int
    last_completed_chunk: int | None
    live_identities: tuple[dict, ...]
    active_memberships: tuple[tuple[str, frozenset[str]], ...]
    membership_history: tuple[tuple[str, frozenset[str]], ...]
    retired_live_ids: frozenset[str]
    next_live_number: int
    memory_matches: tuple[tuple[str, dict | None], ...]
    events: tuple[dict, ...]
    event_keys: tuple[str, ...]
    next_event_number: int
    canonical_state: dict


def _safe_warning(stage: str, version: int, error: BaseException) -> str:
    error_name = type(error).__name__
    return f"Rolling {stage} failed for version {version} ({error_name})."


def _safe_path(value: Any) -> str:
    return _CAMERA_URI_RE.sub("<camera-source>", str(value or ""))


_STABLE_CONSECUTIVE_OUTCOMES = 2
_STABLE_OBSERVATION_COUNT = 6
_MINIMUM_FACE_OBSERVATIONS_FOR_COMPARISON = 1


def _rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 4)


def _live_number(live_id: str) -> int:
    try:
        return int(str(live_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 2**31 - 1


def _event_key(event: Mapping[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in event.items()
        if key not in {"event_id", "analysis_version"}
    }
    return json.dumps(semantic, sort_keys=True, separators=(",", ":"))


class LiveRollingAnalysisSession:
    """Own one latest-only analysis worker and session-memory preview state."""

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[], FrozenAnalysisSnapshot],
        notify: Callable[[dict], Any] | None = None,
        join_timeout_seconds: float = 120.0,
        database_path: str | Path | None = None,
        cluster: Callable[[dict], dict] | None = None,
        associate: Callable[[Mapping[str, Any]], Any] | None = None,
        memory_factory: Callable[[Path], Any] | None = None,
        job_id: str | None = None,
        identity_decisions: bool = False,
        decision_memory_factory: Callable[[Path], Any] | None = None,
        policy_config: Any | None = None,
        debounce_seconds: float = 0.5,
        minimum_new_versions: int = 3,
    ) -> None:
        self.join_timeout_seconds = max(0.01, float(join_timeout_seconds))
        self._snapshot_provider = snapshot_provider
        self._notify_callback = notify
        self._database_path = Path(database_path) if database_path is not None else None
        self._cluster = cluster
        self._associate = associate
        self._memory_factory = memory_factory
        self._job_id = str(job_id or "live-job")
        self._identity_decisions_enabled = bool(identity_decisions)
        self._decision_memory_factory = decision_memory_factory
        self._policy_config = policy_config
        self._debounce_seconds = max(0.0, min(float(debounce_seconds), 0.5))
        self._minimum_new_versions = max(1, int(minimum_new_versions))
        self._decision_memory: Any | None = None
        # Decision state is committed here the moment a Global Memory write
        # succeeds, never through _commit_result: a stale pass is discarded by
        # _commit_result, and re-deciding it would create a duplicate person.
        self._identity_decision_records: dict[str, dict] = {}
        self._identity_evidence_versions: dict[str, int] = {}
        self._identity_evidence_signatures: dict[str, str] = {}
        self._identity_all_evidence_signatures: dict[str, str] = {}
        self._identity_decision_keys: set[str] = set()
        self._identity_states: dict[str, tuple[str, int]] = {}
        self._identity_provisional: dict[str, dict] = {}
        self._identity_outcome_streak: dict[str, tuple[str, int]] = {}

        self._condition = threading.Condition(threading.Lock())
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepting = True
        self._shutdown_requested = False
        self._requested_version = 0
        self._pending_since: float | None = None
        self._completed_version = 0
        self._analysis_version = 0
        self._publication_sequence = 0
        self._generated_at = datetime.now(timezone.utc).isoformat()
        self._analysis_in_progress = False
        self._analysis_state = "idle"
        self._analyzed_embedding_count = 0
        self._unresolved_embedding_count = 0
        self._resolved_cluster_count = 0
        self._last_attempted_embedding_count: int | None = None
        self._last_successful_embedding_count = 0
        self._last_successful_body_count = 0
        self._has_valid_result = True
        self._last_completed_chunk: int | None = None
        self._warning: str | None = None
        self._worker_error: str | None = None
        self._live_identities: tuple[dict, ...] = ()
        self._active_memberships: dict[str, frozenset[str]] = {}
        self._membership_history: dict[str, frozenset[str]] = {}
        self._retired_live_ids: frozenset[str] = frozenset()
        self._next_live_number = 1
        self._memory_matches: dict[str, dict | None] = {}
        self._events: tuple[dict, ...] = ()
        self._event_keys: tuple[str, ...] = ()
        self._next_event_number = 1
        self._canonical_state: dict = {}
        self._discard_late_results = False
        self._analysis_runs_started = 0
        self._analysis_versions_conflated = 0
        self._analysis_versions_skipped_unchanged = 0

    def start(self) -> None:
        with self._condition:
            if self._started:
                raise LiveAnalysisLifecycleError(
                    "Rolling analysis session has already been started."
                )
            self._started = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="person-creation-live-analysis",
                daemon=True,
            )
            thread = self._thread
        thread.start()
        self._publish()

    def request_version(self, version: int) -> bool:
        version = max(0, int(version))
        with self._condition:
            if not self._accepting or self._worker_error is not None:
                return False
            if version > self._requested_version:
                if self._requested_version > self._completed_version:
                    self._analysis_versions_conflated += (
                        version - self._requested_version
                    )
                self._requested_version = version
                if self._pending_since is None:
                    self._pending_since = time.monotonic()
                if not self._analysis_in_progress:
                    self._analysis_state = "scheduled"
            self._wake.set()
            self._condition.notify_all()
        return True

    @property
    def worker_alive(self) -> bool:
        with self._condition:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def finish(self, final_version: int) -> None:
        """Complete or record the final pass, then prove the worker is dead."""
        if not self._started:
            return
        self.request_version(final_version)
        deadline = time.monotonic() + self.join_timeout_seconds

        with self._condition:
            while self._completed_version < int(final_version):
                if self._worker_error is not None:
                    break
                thread = self._thread
                if thread is None or not thread.is_alive():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            self._accepting = False
            self._shutdown_requested = True
            self._wake.set()
            self._condition.notify_all()
            thread = self._thread

        remaining = max(0.0, deadline - time.monotonic())
        if thread is not None:
            thread.join(timeout=remaining)
            if thread.is_alive():
                with self._condition:
                    self._discard_late_results = True
                    self._analysis_in_progress = False
                    self._analysis_state = "drain_timed_out"
                    self._warning = (
                        "Rolling analysis drain exceeded its bounded deadline."
                    )
                raise LiveAnalysisLifecycleError(
                    "Rolling analysis worker did not stop within "
                    f"{self.join_timeout_seconds:g} seconds; staging must be preserved."
                )
        self._publish()

    def abort(self) -> None:
        """Request shutdown after another fail-closed lane has failed."""
        with self._condition:
            self._accepting = False
            self._shutdown_requested = True
            self._wake.set()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self.join_timeout_seconds)
            if thread.is_alive():
                raise LiveAnalysisLifecycleError(
                    "Rolling analysis worker remained alive during abort; "
                    "staging must be preserved."
                )
        self._publish()

    def public_snapshot(self) -> dict:
        with self._condition:
            return self._public_snapshot_locked()

    def canonical_state(self) -> dict:
        with self._condition:
            return deepcopy(self._canonical_state)

    def _public_snapshot_locked(self) -> dict:
        return {
            "enabled": True,
            "publication_sequence": self._publication_sequence,
            "evidence_version": self._analysis_version,
            "generated_at": self._generated_at,
            "requested_version": self._requested_version,
            "analysis_version": self._analysis_version,
            "analysis_state": self._analysis_state,
            "analysis_in_progress": self._analysis_in_progress,
            "analysis_runs_started": self._analysis_runs_started,
            "analysis_versions_conflated": self._analysis_versions_conflated,
            "analysis_versions_skipped_unchanged": (
                self._analysis_versions_skipped_unchanged
            ),
            "analyzed_embedding_count": self._analyzed_embedding_count,
            "unresolved_embedding_count": self._unresolved_embedding_count,
            "resolved_cluster_count": self._resolved_cluster_count,
            "last_completed_preprocessing_chunk": self._last_completed_chunk,
            "analysis_warning": self._worker_error or self._warning,
            "live_identities": deepcopy(list(self._live_identities)),
            "retired_live_identity_ids": sorted(
                self._retired_live_ids,
                key=lambda item: (_live_number(item), item),
            ),
            "live_recognition_events": deepcopy(list(self._events)),
        }

    def _worker_loop(self) -> None:
        memory = None
        cleanup_fatal: BaseException | None = None
        try:
            while True:
                self._wake.wait()
                while True:
                    with self._condition:
                        if (
                            self._shutdown_requested
                            and self._discard_late_results
                        ):
                            return
                        if self._requested_version <= self._completed_version:
                            if self._shutdown_requested:
                                return
                            self._wake.clear()
                            break
                        pending_elapsed = (
                            0.0
                            if self._pending_since is None
                            else time.monotonic() - self._pending_since
                        )
                        pending_versions = (
                            self._requested_version - self._completed_version
                        )
                        if (
                            not self._shutdown_requested
                            and pending_versions < self._minimum_new_versions
                            and pending_elapsed < self._debounce_seconds
                        ):
                            self._condition.wait(
                                timeout=self._debounce_seconds - pending_elapsed
                            )
                            continue
                        target_version = self._requested_version
                        self._pending_since = None
                        self._analysis_in_progress = True
                        self._analysis_state = "running"
                        self._analysis_runs_started += 1

                    snapshot = self._snapshot_provider()
                    embedding_count = len(snapshot.face_embeddings)
                    body_count = len(snapshot.quality_body_crops)
                    with self._condition:
                        can_skip = (
                            self._has_valid_result
                            and embedding_count == self._last_successful_embedding_count
                            and body_count == self._last_successful_body_count
                        )
                        reuse_clustering = (
                            self._has_valid_result
                            and embedding_count == self._last_successful_embedding_count
                            and body_count != self._last_successful_body_count
                            and bool(self._canonical_state)
                        )
                        clustered_override = (
                            {
                                "identity_clusters": deepcopy(
                                    self._canonical_state.get(
                                        "identity_clusters",
                                        [],
                                    )
                                ),
                                "unresolved_faces": deepcopy(
                                    self._canonical_state.get(
                                        "unresolved_faces",
                                        [],
                                    )
                                ),
                            }
                            if reuse_clustering else None
                        )
                        self._last_attempted_embedding_count = embedding_count
                    if can_skip:
                        with self._condition:
                            self._analysis_versions_skipped_unchanged += 1
                        self._complete_without_analysis(snapshot, target_version)
                        continue

                    try:
                        if memory is None:
                            memory = self._open_memory()
                        committed = self._committed_inputs()
                        result = self._analyze(
                            snapshot,
                            memory,
                            committed,
                            clustered_override=clustered_override,
                        )
                        self._commit_result(result, target_version)
                    except Exception as exc:
                        self._record_pass_failure(snapshot, target_version, exc)
        except BaseException as exc:
            self._mark_worker_failure(exc)
            self._publish()
        finally:
            cleanup_error: Exception | None = None
            if memory is not None:
                try:
                    memory.close()
                except Exception as exc:
                    cleanup_error = exc
                except BaseException as exc:
                    cleanup_fatal = exc
            decision_memory = self._decision_memory
            self._decision_memory = None
            if decision_memory is not None:
                try:
                    decision_memory.close()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
                except BaseException as exc:
                    cleanup_fatal = cleanup_fatal or exc
            with self._condition:
                if cleanup_error is not None:
                    self._warning = _safe_warning(
                        "Global Memory cleanup",
                        self._requested_version,
                        cleanup_error,
                    )
                    if self._analysis_state not in {"worker_failed", "warning"}:
                        self._analysis_state = "shutdown_warning"
                if cleanup_fatal is not None:
                    self._worker_error = _safe_warning(
                        "Global Memory cleanup",
                        self._requested_version,
                        cleanup_fatal,
                    )
                    self._warning = self._worker_error
                    self._analysis_state = "worker_failed"
                self._analysis_in_progress = False
                self._accepting = False
                self._condition.notify_all()
            self._publish()
            if cleanup_fatal is not None:
                raise cleanup_fatal

    def _mark_worker_failure(self, error: BaseException) -> None:
        with self._condition:
            self._worker_error = _safe_warning(
                "analysis worker",
                self._requested_version,
                error,
            )
            self._warning = self._worker_error
            self._analysis_state = "worker_failed"
            self._analysis_in_progress = False
            self._accepting = False
            self._condition.notify_all()

    def _open_memory(self) -> Any | None:
        if self._database_path is None:
            from forensics.global_memory.config import DB_PATH

            database_path = Path(DB_PATH)
        else:
            database_path = self._database_path
        try:
            mode = database_path.stat().st_mode
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise OSError("Global Memory path could not be inspected.") from exc
        if not stat.S_ISREG(mode):
            raise OSError("Global Memory path is not a readable database file.")
        if self._memory_factory is not None:
            return self._memory_factory(database_path)
        from forensics.global_memory.store import GlobalMemory

        return GlobalMemory(str(database_path), read_only=True)

    def _cluster_stage(self) -> Callable[[dict], dict]:
        if self._cluster is not None:
            return self._cluster
        from forensics.person_creation.nodes.cluster_identities import cluster_identities

        return cluster_identities

    def _association_stage(self) -> Callable[[Mapping[str, Any]], Any]:
        if self._associate is not None:
            return self._associate
        from forensics.person_creation.nodes.assign_bodies_to_clusters import (
            compute_body_cluster_assignments,
        )

        return compute_body_cluster_assignments

    def _committed_inputs(self) -> _CommittedInputs:
        with self._condition:
            return _CommittedInputs(
                active_memberships=tuple(self._active_memberships.items()),
                membership_history=tuple(self._membership_history.items()),
                retired_live_ids=self._retired_live_ids,
                next_live_number=self._next_live_number,
                memory_matches=tuple(
                    (live_id, deepcopy(match))
                    for live_id, match in self._memory_matches.items()
                ),
                events=tuple(deepcopy(event) for event in self._events),
                event_keys=self._event_keys,
                next_event_number=self._next_event_number,
            )

    def _analyze(
        self,
        snapshot: FrozenAnalysisSnapshot,
        memory: Any | None,
        committed: _CommittedInputs,
        clustered_override: Mapping[str, Any] | None = None,
    ) -> _AnalysisResult:
        state = snapshot.mutable_node_state()
        clustered = (
            deepcopy(dict(clustered_override))
            if clustered_override is not None
            else self._cluster_stage()(state)
        )
        if not isinstance(clustered, dict):
            raise TypeError("cluster_identities returned a non-dictionary result")
        state.update(clustered)
        association = self._association_stage()(state)
        state.update({
            "associations": list(getattr(association, "associations", []) or []),
            "cluster_assignments": dict(
                getattr(association, "cluster_assignments", {}) or {}
            ),
            "unattached_bodies": list(
                getattr(association, "unattached_bodies", []) or []
            ),
            "frame_groups": list(getattr(association, "frame_groups", []) or []),
            "rejected_pairs": list(
                getattr(association, "rejected_pairs", []) or []
            ),
        })
        from forensics.person_creation.nodes.build_profile import build_profile
        from forensics.person_creation.nodes.compute_reid import compute_reid
        from forensics.person_creation.nodes.select_best import select_best

        state.update(select_best(state))
        state.update(compute_reid(state))
        state.update(build_profile(state))
        canonical_node_state = state

        clusters = list(state.get("identity_clusters") or [])
        unresolved_records = list(state.get("unresolved_faces") or [])
        analysis_groups = list(clusters)
        for index, record in enumerate(unresolved_records):
            path = _safe_path(record.get("crop_path"))
            vector = np.asarray(record.get("embedding"), dtype=np.float64)
            if (
                not path
                or vector.ndim != 1
                or vector.size == 0
                or not np.isfinite(vector).all()
            ):
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            analysis_groups.append({
                "cluster_id": -(index + 1),
                "face_records": [record],
                "representative_embedding": (vector / norm).astype(float).tolist(),
                "face_count": 1,
                "confidence": 0.0,
                "low_confidence": True,
                "clustering_state": "unresolved",
            })
        current_memberships = []
        cluster_by_signature: dict[tuple[str, ...], dict] = {}
        for cluster in analysis_groups:
            paths = frozenset(
                _safe_path(record.get("crop_path"))
                for record in cluster.get("face_records", [])
                if record.get("crop_path")
            )
            if not paths:
                continue
            current = CurrentMembership(
                cluster_label=int(cluster["cluster_id"]),
                crop_paths=paths,
            )
            current_memberships.append(current)
            cluster_by_signature[current.signature] = cluster

        previous_active_memberships = dict(committed.active_memberships)
        previous_memberships = dict(committed.membership_history)
        assigned = assign_live_identities(
            current_memberships=current_memberships,
            previous_memberships=previous_memberships,
            next_live_number=committed.next_live_number,
        )
        chunk_by_path = dict(snapshot.face_chunk_membership)
        previous_matches = dict(committed.memory_matches)
        next_matches: dict[str, dict | None] = {}
        identities: list[dict] = []

        for identity in sorted(
            assigned.identities,
            key=lambda item: (_live_number(item.session_person_id), item.session_person_id),
        ):
            cluster = cluster_by_signature[tuple(sorted(identity.crop_paths))]
            # Representative-strategy comparison is deliberately absent here.
            # It costs one Global Memory query per enrolled person per identity
            # per pass and publishes the enrolled roster, so it lives only in
            # forensics/person_creation/tools/compare_cluster_representatives.py.
            representative_path = self._representative_path(cluster)
            chunks = sorted(
                chunk_by_path[path]
                for path in identity.crop_paths
                if path in chunk_by_path
            )
            assignments = list(
                getattr(association, "cluster_assignments", {}).get(
                    identity.cluster_label,
                    [],
                )
            )
            body_count = len(assignments)
            face_count = len(identity.crop_paths)
            try:
                record = self._maybe_decide_identity(
                    live_id=identity.session_person_id,
                    cluster=cluster,
                    snapshot=snapshot,
                    crop_paths=identity.crop_paths,
                    assignments=assignments,
                    face_count=face_count,
                )
            except Exception as exc:
                record = None
                with self._condition:
                    self._warning = _safe_warning(
                        "identity decision",
                        snapshot.version,
                        exc,
                    )
            if record is not None and record.get("candidate_person_id"):
                match = {
                    "person_id": str(record["candidate_person_id"]),
                    "name": str(record["candidate_person_id"]),
                    "similarity": record.get("candidate_similarity"),
                }
            elif self._identity_decisions_enabled:
                match = None
            else:
                match = self._memory_match(
                    memory,
                    cluster.get("representative_embedding"),
                )
            next_matches[identity.session_person_id] = match
            if (
                str(cluster.get("clustering_state") or "resolved")
                == "unresolved"
                and not bool((record or {}).get("publishable"))
            ):
                # Noise remains in the canonical evidence/membership ledger so
                # a later DBSCAN cluster keeps the same live ID. It is not a
                # frontend identity, review item, VLM input, or durable write.
                continue
            state, state_version = self._identity_state_and_version(
                identity.session_person_id,
                record,
            )
            identities.append({
                "session_person_id": identity.session_person_id,
                "live_identity_id": identity.session_person_id,
                "cluster_label": identity.cluster_label,
                "status": "provisional",
                "clustering_state": str(
                    cluster.get("clustering_state") or "resolved"
                ),
                "state": state,
                "version": state_version,
                "face_count": face_count,
                "body_count": body_count,
                "associated_body_count": body_count,
                "first_seen_chunk": chunks[0] if chunks else None,
                "last_seen_chunk": chunks[-1] if chunks else None,
                "first_seen": chunks[0] if chunks else None,
                "last_seen": chunks[-1] if chunks else None,
                "representative_face_path": representative_path,
                "best_face_path": representative_path,
                "best_body_path": self._best_body_path(assignments),
                "memory_match": dict(match) if match is not None else None,
                "candidate_person_id": (record or {}).get("candidate_person_id"),
                "candidate_similarity": (record or {}).get("candidate_similarity"),
                "second_candidate_person_id": (
                    (record or {}).get("second_candidate_person_id")
                ),
                "second_candidate_similarity": (
                    (record or {}).get("second_candidate_similarity")
                ),
                "margin": (record or {}).get("margin"),
                "decision": (record or {}).get("decision"),
                "reason": (record or {}).get("reason"),
                "provisional": bool((record or {}).get("provisional")),
                "persisted": bool((record or {}).get("persisted")),
                "canonical_person_id": (record or {}).get("canonical_person_id"),
                "suggestion_id": (record or {}).get("suggestion_id"),
                "decision_version": (record or {}).get("decision_version"),
                "evidence_version": (record or {}).get("evidence_version"),
                "evidence_signature": (record or {}).get("evidence_signature"),
                "observation_count": (
                    (record or {}).get("observation_count", face_count)
                ),
                "comparison_timestamp": (
                    (record or {}).get("comparison_timestamp")
                ),
                "latency_metrics": deepcopy(
                    (record or {}).get("latency_metrics") or {}
                ),
                "quality_classification": (
                    (record or {}).get("quality_classification")
                ),
            })

        raw_events = self._build_events(
            assigned=assigned,
            previous_matches=previous_matches,
            next_matches=next_matches,
            analysis_version=snapshot.version,
        )
        events, event_keys, next_event_number = self._prepare_events(
            raw_events=tuple(raw_events),
            previous_events=committed.events,
            previous_keys=committed.event_keys,
            next_event_number=committed.next_event_number,
        )
        active_ids = {live_id for live_id, _paths in assigned.active_memberships}
        membership_history = dict(previous_memberships)
        for live_id, paths in assigned.active_memberships:
            membership_history[live_id] = frozenset(
                set(membership_history.get(live_id, frozenset())) | set(paths)
            )
        retired = frozenset(
            (
                set(committed.retired_live_ids)
                | (set(previous_active_memberships) - active_ids)
            )
            - active_ids
        )
        return _AnalysisResult(
            version=snapshot.version,
            embedding_count=len(snapshot.face_embeddings),
            body_count=len(snapshot.quality_body_crops),
            unresolved_embedding_count=len(unresolved_records),
            resolved_cluster_count=len(clusters),
            last_completed_chunk=snapshot.last_completed_preprocessing_chunk,
            live_identities=tuple(identities),
            active_memberships=assigned.active_memberships,
            membership_history=tuple(membership_history.items()),
            retired_live_ids=retired,
            next_live_number=assigned.next_live_number,
            memory_matches=tuple(sorted(next_matches.items())),
            events=events,
            event_keys=event_keys,
            next_event_number=next_event_number,
            canonical_state={
                key: deepcopy(canonical_node_state.get(key))
                for key in (
                    "identity_clusters",
                    "unresolved_faces",
                    "associations",
                    "cluster_assignments",
                    "unattached_bodies",
                    "frame_groups",
                    "rejected_pairs",
                    "per_cluster_best_body_crops",
                    "best_body_crops",
                    "reid_embeddings",
                    "reid_crop_counts",
                    "reid_reasons",
                    "per_cluster_profiles",
                    "profile",
                )
            },
        )

    @staticmethod
    def _representative_path(cluster: Mapping[str, Any]) -> str:
        records = list(cluster.get("face_records") or [])
        if not records:
            return ""
        selected = min(
            records,
            key=lambda record: (
                -float(record.get("sharpness") or 0.0),
                _safe_path(record.get("crop_path")),
            ),
        )
        return _safe_path(selected.get("crop_path"))

    @staticmethod
    def _memory_match(memory: Any | None, embedding: Any) -> dict | None:
        if memory is None or embedding is None:
            return None
        matches = memory.query_by_face(embedding, top_k=1)
        if not matches:
            return None
        match = matches[0]
        return {
            "person_id": str(match["person_id"]),
            "name": str(match["name"]),
            "similarity": round(float(match["similarity"]), 4),
        }

    def identity_decisions(self) -> tuple[dict, ...]:
        """Return every identity decision already persisted during capture."""
        with self._condition:
            return tuple(
                deepcopy(record)
                for _live_id, record in sorted(
                    self._identity_decision_records.items(),
                    key=lambda item: (_live_number(item[0]), item[0]),
                )
            )

    def _policy_configuration(self) -> Any:
        if self._policy_config is not None:
            return self._policy_config
        from forensics.global_memory.identity_policy import IdentityPolicyConfig

        self._policy_config = IdentityPolicyConfig.from_environment()
        return self._policy_config

    def _open_decision_memory(self) -> Any | None:
        """Open one writable Global Memory handle reused for every decision."""
        if self._decision_memory is not None:
            return self._decision_memory
        if self._decision_memory_factory is not None:
            from forensics.global_memory.config import DB_PATH

            self._decision_memory = self._decision_memory_factory(
                self._database_path or Path(DB_PATH)
            )
            return self._decision_memory
        from forensics.global_memory.store import GlobalMemory

        # With no explicit path, defer to GlobalMemory so FORENSICS_MEMORY_DB
        # keeps tests and isolated deployments off the default database.
        self._decision_memory = (
            GlobalMemory(str(self._database_path))
            if self._database_path is not None else GlobalMemory()
        )
        return self._decision_memory

    @staticmethod
    def _evidence_signature(keys) -> str:
        digest = hashlib.sha256()
        for key in sorted(keys):
            digest.update(str(key).encode("utf-8", "replace"))
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def _identity_evidence(
        cluster: Mapping[str, Any],
        assignments: list[dict],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        faces: list[tuple[str, str]] = []
        seen: set[str] = set()
        for record in cluster.get("face_records") or []:
            path = _safe_path(record.get("crop_path"))
            if not path:
                continue
            key = identity_evidence_key(path, "face")
            if key not in seen:
                seen.add(key)
                faces.append((key, path))
        bodies: list[tuple[str, str]] = []
        for item in assignments:
            path = _safe_path(item.get("body_crop_path"))
            if not path:
                continue
            key = identity_evidence_key(path, "body")
            if key not in seen:
                seen.add(key)
                bodies.append((key, path))
        return faces, bodies

    @staticmethod
    def _face_embedding_for_evidence(
        cluster: Mapping[str, Any],
        evidence_keys: set[str],
    ) -> list[float]:
        """Build the representative from exactly the selected face crops."""
        vectors: list[np.ndarray] = []
        matched: set[str] = set()
        for record in cluster.get("face_records") or []:
            path = _safe_path(record.get("crop_path"))
            if not path:
                continue
            key = identity_evidence_key(path, "face")
            if key not in evidence_keys or key in matched:
                continue
            vector = np.asarray(record.get("embedding"), dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("face evidence embedding is invalid")
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("face evidence embedding has zero norm")
            vectors.append(vector / norm)
            matched.add(key)
        if matched != evidence_keys:
            raise ValueError("face evidence embedding is missing")
        mean = np.mean(np.asarray(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            raise ValueError("face evidence mean embedding has zero norm")
        return (mean / norm).astype(float).tolist()

    def _provisional_decision(
        self,
        memory: Any,
        cluster: Mapping[str, Any],
        face_count: int,
        policy: Any,
    ) -> Any:
        from forensics.global_memory.identity_policy import evaluate_identity_decision

        embedding = np.asarray(
            cluster.get("representative_embedding"),
            dtype=np.float64,
        )
        if (
            embedding.ndim != 1
            or embedding.size == 0
            or not np.isfinite(embedding).all()
            or float(np.linalg.norm(embedding)) <= 0
        ):
            raise ValueError("representative face embedding is invalid")
        candidates = memory.rank_identity_candidates(embedding.tolist())
        return evaluate_identity_decision(
            top_candidate=candidates[0] if candidates else None,
            second_candidate=candidates[1] if len(candidates) > 1 else None,
            observation_count=face_count,
            low_confidence=bool(cluster.get("low_confidence")),
            configuration=policy,
        )

    @staticmethod
    def _latency_metrics(
        cluster: Mapping[str, Any],
        comparison_completed: float,
    ) -> dict:
        timed_records = []
        for record in cluster.get("face_records") or []:
            timing = dict(record.get("_latency_timing") or {})
            try:
                capture = float(timing["capture_monotonic"])
                face = float(timing["face_detected_monotonic"])
                embedding = float(timing["embedding_completed_monotonic"])
            except (KeyError, TypeError, ValueError):
                continue
            timed_records.append((capture, face, embedding, timing))
        if not timed_records:
            return {}
        capture, face, embedding, timing = min(
            timed_records,
            key=lambda item: item[0],
        )
        return {
            "source_frame_timestamp": timing.get("source_frame_timestamp"),
            "capture_monotonic": capture,
            "face_detected_monotonic": face,
            "quality_accepted_monotonic": timing.get(
                "quality_accepted_monotonic"
            ),
            "embedding_completed_monotonic": embedding,
            "comparison_completed_monotonic": float(comparison_completed),
            "capture_to_face_ms": round(max(0.0, face - capture) * 1000.0, 3),
            "capture_to_embedding_ms": round(
                max(0.0, embedding - capture) * 1000.0,
                3,
            ),
            "embedding_to_comparison_ms": round(
                max(0.0, comparison_completed - embedding) * 1000.0,
                3,
            ),
        }

    @staticmethod
    def _best_body_path(assignments: list[dict]) -> str:
        if not assignments:
            return ""
        selected = min(
            assignments,
            key=lambda item: (
                -float(item.get("body_sharpness") or 0.0),
                _safe_path(item.get("body_crop_path")),
            ),
        )
        return _safe_path(selected.get("body_crop_path"))

    def _live_profile(
        self,
        *,
        cluster: Mapping[str, Any],
        snapshot: FrozenAnalysisSnapshot,
        crop_paths: frozenset[str],
        assignments: list[dict],
    ) -> dict:
        face_crops = sorted(crop_paths)
        body_crops = sorted({
            str(item.get("body_crop_path"))
            for item in assignments
            if item.get("body_crop_path")
        })
        face_sharpness = {
            str(record.get("crop_path")): float(record.get("sharpness") or 0.0)
            for record in cluster.get("face_records") or []
            if record.get("crop_path")
        }
        return {
            "name": snapshot.person_name,
            "face_embedding": cluster.get("representative_embedding"),
            "face_crops": face_crops,
            "body_crops": body_crops,
            "best_body_crops": body_crops[:5],
            "face_crop_sharpness": face_sharpness,
            "video_sources": list(snapshot.video_paths),
            "appearance": {"date": date.today().isoformat()},
        }

    def _maybe_decide_identity(
        self,
        *,
        live_id: str,
        cluster: Mapping[str, Any],
        snapshot: FrozenAnalysisSnapshot,
        crop_paths: frozenset[str],
        assignments: list[dict],
        face_count: int,
    ) -> dict | None:
        """Evaluate once per distinct face signature and persist only clusters."""
        if not self._identity_decisions_enabled:
            return None
        with self._condition:
            if self._discard_late_results:
                return deepcopy(
                    self._identity_decision_records.get(live_id)
                    or self._identity_provisional.get(live_id)
                )

        faces, bodies = self._identity_evidence(cluster, assignments)
        face_count = len(faces)
        face_signature = self._evidence_signature(
            key for key, _path in faces
        )
        all_signature = self._evidence_signature(
            key for key, _path in faces + bodies
        )

        with self._condition:
            record = deepcopy(self._identity_decision_records.get(live_id))
            face_unchanged = (
                self._identity_evidence_signatures.get(live_id) == face_signature
            )
            all_unchanged = (
                self._identity_all_evidence_signatures.get(live_id)
                == all_signature
            )
            version = self._identity_evidence_versions.get(live_id, 0)
            if not face_unchanged:
                version += 1
                self._identity_evidence_signatures[live_id] = face_signature
                self._identity_evidence_versions[live_id] = version
            self._identity_all_evidence_signatures[live_id] = all_signature
            provisional = deepcopy(self._identity_provisional.get(live_id))
            streak = self._identity_outcome_streak.get(live_id)

        if record is not None:
            if all_unchanged:
                return record
            return self._append_new_evidence(
                live_id=live_id,
                record=record,
                faces=faces,
                bodies=bodies,
                cluster=cluster,
                snapshot=snapshot,
                version=version,
                signature=all_signature,
            )

        if face_unchanged and provisional is not None:
            return provisional

        policy = self._policy_configuration()
        if face_count < _MINIMUM_FACE_OBSERVATIONS_FOR_COMPARISON:
            return None
        memory = self._open_decision_memory()
        if memory is None:
            return None

        unresolved = (
            str(cluster.get("clustering_state") or "resolved") == "unresolved"
        )
        decision = self._provisional_decision(memory, cluster, face_count, policy)
        comparison_completed = time.monotonic()
        comparison_timestamp = datetime.now(timezone.utc).isoformat()
        latency_metrics = self._latency_metrics(cluster, comparison_completed)
        margin_sufficient = (
            decision.margin is None
            or decision.margin >= policy.minimum_margin
        )
        quality_results = [
            record.get("face_quality") or {}
            for record in cluster.get("face_records") or []
        ]
        confirmation_eligible = bool(quality_results) and all(
            result.get("immediate_confirmation_eligible") is True
            for result in quality_results
        )
        # Backward-compatible test fixtures without quality metadata are
        # standard evidence, never a production bypass.
        strong_single_known = bool(
            unresolved
            and face_count == 1
            and confirmation_eligible
            and decision.top_candidate_person_id
            and decision.top_similarity is not None
            and decision.top_similarity >= policy.maximum_similarity
            and margin_sufficient
        )
        outcome = (
            "attach_existing" if strong_single_known else decision.decision.value
        )
        reason = (
            "strong_clear_match" if strong_single_known else decision.reason.value
        )
        consecutive = streak[1] + 1 if streak and streak[0] == outcome else 1
        published = {
            "live_identity_id": live_id,
            "job_id": self._job_id,
            "decision": outcome,
            "reason": reason,
            "publishable": (not unresolved) or strong_single_known,
            "provisional": True,
            "persisted": False,
            "canonical_person_id": None,
            "suggestion_id": None,
            "candidate_person_id": decision.top_candidate_person_id,
            "candidate_similarity": _rounded(decision.top_similarity),
            "second_candidate_person_id": decision.second_candidate_person_id,
            "second_candidate_similarity": _rounded(decision.second_similarity),
            "margin": _rounded(decision.margin),
            "decision_version": version,
            "evidence_version": version,
            "evidence_signature": face_signature,
            "observation_count": face_count,
            "last_evidence_signature": all_signature,
            "comparison_timestamp": comparison_timestamp,
            "latency_metrics": latency_metrics,
            "quality_classification": (
                quality_results[0].get("quality_class")
                if quality_results else None
            ),
        }
        previous_candidate = (provisional or {}).get("candidate_person_id")
        next_candidate = published.get("candidate_person_id")
        candidate_changed = bool(
            previous_candidate and previous_candidate != next_candidate
        )
        if candidate_changed:
            previous_similarity = float(
                (provisional or {}).get("candidate_similarity") or 0.0
            )
            next_similarity = float(published.get("candidate_similarity") or 0.0)
            previous_margin = float((provisional or {}).get("margin") or 0.0)
            next_margin = float(published.get("margin") or 0.0)
            materially_stronger = (
                next_similarity >= previous_similarity + policy.minimum_margin
                and next_margin >= previous_margin
                and confirmation_eligible
            )
            if not materially_stronger:
                for field in (
                    "decision",
                    "candidate_person_id",
                    "candidate_similarity",
                    "second_candidate_person_id",
                    "second_candidate_similarity",
                    "margin",
                    "publishable",
                ):
                    published[field] = (provisional or {}).get(field)
                published["reason"] = "candidate_evidence_conflict"
                published["conflict_count"] = int(
                    (provisional or {}).get("conflict_count") or 0
                ) + 1
                with self._condition:
                    self._identity_outcome_streak.pop(live_id, None)
                    self._identity_provisional[live_id] = deepcopy(published)
                return published
        if unresolved:
            # A one-face bypass is visual only. It never advances the durable
            # streak and can never invoke register_with_identity_policy.
            with self._condition:
                self._identity_provisional[live_id] = deepcopy(published)
            return published
        with self._condition:
            self._identity_outcome_streak[live_id] = (outcome, consecutive)
            self._identity_provisional[live_id] = deepcopy(published)

        stable = (
            consecutive >= _STABLE_CONSECUTIVE_OUTCOMES
            or face_count >= _STABLE_OBSERVATION_COUNT
        )
        if not stable:
            return published
        return self._persist_identity(
            live_id=live_id,
            cluster=cluster,
            snapshot=snapshot,
            faces=faces,
            bodies=bodies,
            face_count=face_count,
            version=version,
            signature=all_signature,
            policy=policy,
            memory=memory,
            comparison_timestamp=comparison_timestamp,
            latency_metrics=latency_metrics,
        )

    def _persist_identity(
        self,
        *,
        live_id: str,
        cluster: Mapping[str, Any],
        snapshot: FrozenAnalysisSnapshot,
        faces: list[tuple[str, str]],
        bodies: list[tuple[str, str]],
        face_count: int,
        version: int,
        signature: str,
        policy: Any,
        memory: Any,
        comparison_timestamp: str,
        latency_metrics: Mapping[str, Any],
    ) -> dict | None:
        key = f"{self._job_id}|{live_id}|{version}"
        with self._condition:
            if key in self._identity_decision_keys:
                return deepcopy(self._identity_decision_records.get(live_id))

        from forensics.person_creation.media_lifecycle import relocate_profile_media

        profile = self._live_profile(
            cluster=cluster,
            snapshot=snapshot,
            crop_paths=frozenset(path for _key, path in faces),
            assignments=[],
        )
        profile["body_crops"] = sorted(path for _key, path in bodies)
        profile["best_body_crops"] = profile["body_crops"][:5]
        relocated: dict[str, Any] = {}

        def prepare_for_person(person_id: str, raw_profile: dict) -> dict:
            # Relocate retained evidence into the canonical person directory
            # *inside* the identity transaction, so Global Memory only ever
            # persists person_NNN relative paths - never _staging, never
            # cluster_N, never absolute. Staging sources are deliberately left
            # in place: the batch tail still reads them after Stop.
            relocation = relocate_profile_media(
                raw_profile,
                person_id,
                media_root=memory.media_root,
            )
            relocated["profile"] = relocation.profile
            return relocation.profile

        result = memory.register_with_identity_policy(
            profile,
            observation_count=face_count,
            low_confidence=bool(cluster.get("low_confidence")),
            configuration=policy,
            prepare_profile_for_person=prepare_for_person,
            evidence_keys=[key for key, _path in faces + bodies],
        )
        persisted = relocated.get("profile") or {}
        persisted_faces = sorted(str(path) for path in persisted.get("face_crops") or [])
        persisted_bodies = sorted(str(path) for path in persisted.get("body_crops") or [])
        evidence_keys = sorted(key for key, _path in faces + bodies)
        record = {
            "live_identity_id": live_id,
            "job_id": self._job_id,
            "decision": result.decision.value,
            "reason": result.reason.value,
            "provisional": False,
            "persisted": True,
            "canonical_person_id": result.person_id,
            "suggestion_id": result.suggestion_id,
            "candidate_person_id": result.top_candidate_person_id,
            "candidate_similarity": _rounded(result.top_similarity),
            "second_candidate_person_id": result.second_candidate_person_id,
            "second_candidate_similarity": _rounded(result.second_similarity),
            "margin": _rounded(result.margin),
            "decision_version": version,
            "evidence_version": version,
            "evidence_signature": self._evidence_signature(
                key for key, _path in faces
            ),
            "observation_count": face_count,
            "evidence_keys": evidence_keys,
            "persisted_evidence_keys": evidence_keys,
            "persisted_observation_count": face_count,
            "last_appended_analysis_version": snapshot.version,
            "last_evidence_signature": signature,
            "source_face_crops": sorted(path for _key, path in faces),
            "source_body_crops": sorted(path for _key, path in bodies),
            "persisted_face_crops": persisted_faces,
            "persisted_body_crops": persisted_bodies,
            "canonical_face_paths": persisted_faces,
            "canonical_body_paths": persisted_bodies,
            "persisted_face_count": len(persisted_faces),
            "persisted_body_count": len(persisted_bodies),
            "comparison_timestamp": comparison_timestamp,
            "latency_metrics": deepcopy(dict(latency_metrics)),
        }
        with self._condition:
            self._identity_decision_keys.add(key)
            self._identity_decision_records[live_id] = record
            self._identity_provisional.pop(live_id, None)
        return deepcopy(record)

    def _append_new_evidence(
        self,
        *,
        live_id: str,
        record: dict,
        faces: list[tuple[str, str]],
        bodies: list[tuple[str, str]],
        cluster: Mapping[str, Any],
        snapshot: FrozenAnalysisSnapshot,
        version: int,
        signature: str,
    ) -> dict:
        """Route evidence observed after persistence through the append path."""
        persisted_keys = set(record.get("persisted_evidence_keys") or [])
        new_faces = [(k, p) for k, p in faces if k not in persisted_keys]
        new_bodies = [(k, p) for k, p in bodies if k not in persisted_keys]
        canonical = str(record.get("canonical_person_id") or "")

        if (not new_faces and not new_bodies) or not canonical:
            with self._condition:
                updated = dict(record)
                updated["last_evidence_signature"] = signature
                self._identity_decision_records[live_id] = updated
            return deepcopy(updated)

        memory = self._open_decision_memory()
        if memory is None:
            return deepcopy(record)

        from forensics.person_creation.media_lifecycle import relocate_profile_media

        relocation = relocate_profile_media(
            {
                "face_crops": [path for _key, path in new_faces],
                "body_crops": [path for _key, path in new_bodies],
                "best_body_crops": [path for _key, path in new_bodies][:5],
            },
            canonical,
            media_root=memory.media_root,
        )
        relocated = relocation.profile
        appended = memory.append_identity_evidence(
            canonical,
            embedding=(
                self._face_embedding_for_evidence(
                    cluster,
                    {key for key, _path in new_faces},
                )
                if new_faces else None
            ),
            observation_count=len(new_faces),
            face_crops=list(relocated.get("face_crops") or []),
            body_crops=list(relocated.get("body_crops") or []),
            appearance={
                "date": date.today().isoformat(),
                "video_sources": list(snapshot.video_paths),
            },
            evidence_keys=[key for key, _path in new_faces + new_bodies],
        )

        with self._condition:
            updated = dict(record)
            updated["persisted_evidence_keys"] = sorted(
                persisted_keys
                | {key for key, _path in new_faces}
                | {key for key, _path in new_bodies}
            )
            updated["evidence_keys"] = sorted(
                set(updated.get("evidence_keys") or [])
                | {key for key, _path in faces + bodies}
            )
            updated["persisted_face_crops"] = sorted(
                set(updated.get("persisted_face_crops") or [])
                | {str(path) for path in relocated.get("face_crops") or []}
            )
            updated["persisted_body_crops"] = sorted(
                set(updated.get("persisted_body_crops") or [])
                | {str(path) for path in relocated.get("body_crops") or []}
            )
            updated["persisted_face_count"] = len(updated["persisted_face_crops"])
            updated["persisted_body_count"] = len(updated["persisted_body_crops"])
            updated["canonical_face_paths"] = list(
                updated["persisted_face_crops"]
            )
            updated["canonical_body_paths"] = list(
                updated["persisted_body_crops"]
            )
            updated["persisted_observation_count"] = int(
                updated.get("persisted_observation_count") or 0
            ) + int(
                appended.embedding_count_after - appended.embedding_count_before
            )
            updated["last_appended_analysis_version"] = snapshot.version
            updated["last_evidence_signature"] = signature
            self._identity_decision_records[live_id] = updated
        return deepcopy(updated)

    def _identity_state_and_version(
        self,
        live_id: str,
        record: Mapping[str, Any] | None,
    ) -> tuple[str, int]:
        """Version increments only when the identity state itself changes."""
        if record is None:
            state = "observing"
        elif record.get("provisional"):
            state = "provisional"
        else:
            state = str(record.get("decision"))
        with self._condition:
            previous = self._identity_states.get(live_id)
            if previous is None:
                resolved = (state, 1)
            elif previous[0] != state:
                resolved = (state, previous[1] + 1)
            else:
                resolved = previous
            self._identity_states[live_id] = resolved
        return resolved

    def _build_events(
        self,
        *,
        assigned: IdentityAssignmentResult,
        previous_matches: Mapping[str, dict | None],
        next_matches: Mapping[str, dict | None],
        analysis_version: int,
    ) -> list[dict]:
        events: list[dict] = []
        for transition in assigned.transitions:
            event = {
                "type": transition.event_type,
                "session_person_id": transition.session_person_id,
                "analysis_version": analysis_version,
            }
            if transition.previous_live_id is not None:
                event["previous_live_id"] = transition.previous_live_id
            if transition.retained_live_id is not None:
                event["retained_live_id"] = transition.retained_live_id
            if transition.child_live_ids:
                event["child_live_ids"] = list(transition.child_live_ids)
            if transition.absorbed_live_ids:
                event["absorbed_live_ids"] = list(transition.absorbed_live_ids)
            if transition.overlap_counts:
                event["overlap_counts"] = {
                    live_id: int(count)
                    for live_id, count in transition.overlap_counts
                }
            events.append(event)

        created = {
            transition.session_person_id
            for transition in assigned.transitions
            if transition.event_type == "identity_created"
        }
        for live_id in sorted(next_matches, key=lambda item: (_live_number(item), item)):
            old = previous_matches.get(live_id)
            new = next_matches[live_id]
            if old is None and new is None:
                if live_id in created:
                    events.append({
                        "type": "memory_no_match",
                        "session_person_id": live_id,
                        "analysis_version": analysis_version,
                    })
                continue
            if old is None and new is not None:
                event_type = "memory_match_found"
            elif old is not None and new is None:
                event_type = "memory_match_lost"
            elif old["person_id"] != new["person_id"]:
                event_type = "memory_match_changed"
            else:
                continue
            event = {
                "type": event_type,
                "session_person_id": live_id,
                "analysis_version": analysis_version,
            }
            if old is not None:
                event["previous_person_id"] = old["person_id"]
            if new is not None:
                event["memory_match"] = dict(new)
            events.append(event)
        return events

    def _prepare_events(
        self,
        *,
        raw_events: tuple[dict, ...],
        previous_events: tuple[dict, ...],
        previous_keys: tuple[str, ...],
        next_event_number: int,
    ) -> tuple[tuple[dict, ...], tuple[str, ...], int]:
        events = [deepcopy(event) for event in previous_events]
        keys = list(previous_keys)
        known = set(keys)
        next_number = int(next_event_number)
        for raw_event in raw_events:
            event = json.loads(json.dumps(raw_event, sort_keys=True))
            key = _event_key(event)
            if key in known:
                continue
            if len(events) == _EVENT_LIMIT:
                known.discard(keys.pop(0))
                events.pop(0)
            event["event_id"] = f"event_{next_number:04d}"
            next_number += 1
            events.append(event)
            keys.append(key)
            known.add(key)
        return tuple(events), tuple(keys), next_number

    def _commit_result(self, result: _AnalysisResult, target_version: int) -> None:
        active_memberships = dict(result.active_memberships)
        membership_history = dict(result.membership_history)
        memory_matches = {
            live_id: deepcopy(match) for live_id, match in result.memory_matches
        }
        status_published = time.monotonic()
        live_identity_records = []
        for raw_identity in result.live_identities:
            identity = deepcopy(raw_identity)
            metrics = dict(identity.get("latency_metrics") or {})
            try:
                capture = float(metrics["capture_monotonic"])
                comparison = float(metrics["comparison_completed_monotonic"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                metrics["status_published_monotonic"] = status_published
                metrics["comparison_to_status_ms"] = round(
                    max(0.0, status_published - comparison) * 1000.0,
                    3,
                )
                metrics["capture_to_status_ms"] = round(
                    max(0.0, status_published - capture) * 1000.0,
                    3,
                )
            identity["latency_metrics"] = metrics
            live_identity_records.append(identity)
        live_identities = tuple(live_identity_records)
        events = tuple(deepcopy(event) for event in result.events)
        event_keys = tuple(result.event_keys)
        with self._condition:
            if self._discard_late_results:
                self._analysis_in_progress = False
                self._condition.notify_all()
                return
            self._active_memberships = active_memberships
            self._membership_history = membership_history
            self._retired_live_ids = result.retired_live_ids
            self._next_live_number = result.next_live_number
            self._memory_matches = memory_matches
            self._live_identities = live_identities
            self._events = events
            self._event_keys = event_keys
            self._next_event_number = result.next_event_number
            self._canonical_state = deepcopy(result.canonical_state)
            self._completed_version = max(
                self._completed_version,
                target_version,
                result.version,
            )
            self._analysis_version = max(
                self._analysis_version,
                target_version,
                result.version,
            )
            self._last_attempted_embedding_count = result.embedding_count
            self._last_successful_embedding_count = result.embedding_count
            self._last_successful_body_count = result.body_count
            self._has_valid_result = True
            self._analyzed_embedding_count = result.embedding_count
            self._unresolved_embedding_count = result.unresolved_embedding_count
            self._resolved_cluster_count = result.resolved_cluster_count
            self._last_completed_chunk = result.last_completed_chunk
            self._warning = None
            self._analysis_in_progress = False
            self._analysis_state = (
                "scheduled"
                if self._requested_version > self._completed_version
                else "ready"
            )
            self._condition.notify_all()
        self._publish()

    def _complete_without_analysis(
        self,
        snapshot: FrozenAnalysisSnapshot,
        target_version: int,
    ) -> None:
        with self._condition:
            completed = max(target_version, snapshot.version)
            self._completed_version = max(self._completed_version, completed)
            self._analysis_version = max(self._analysis_version, completed)
            self._last_completed_chunk = snapshot.last_completed_preprocessing_chunk
            self._warning = None
            self._analysis_in_progress = False
            self._analysis_state = "ready"
            self._condition.notify_all()
        self._publish()

    def _record_pass_failure(
        self,
        snapshot: FrozenAnalysisSnapshot,
        target_version: int,
        error: Exception,
    ) -> None:
        warning = _safe_warning("analysis", snapshot.version, error)
        with self._condition:
            completed = min(
                max(target_version, snapshot.version),
                self._requested_version,
            )
            self._completed_version = max(self._completed_version, completed)
            self._last_completed_chunk = snapshot.last_completed_preprocessing_chunk
            self._warning = warning
            self._analysis_in_progress = False
            self._analysis_state = "warning"
            self._condition.notify_all()
        self._publish()

    def _publish(self) -> None:
        if self._notify_callback is None:
            return
        with self._condition:
            self._publication_sequence += 1
            self._generated_at = datetime.now(timezone.utc).isoformat()
            snapshot = self._public_snapshot_locked()
        try:
            self._notify_callback(snapshot)
        except Exception as exc:
            with self._condition:
                self._warning = _safe_warning(
                    "status publication",
                    self._completed_version,
                    exc,
                )
                if self._analysis_state == "ready":
                    self._analysis_state = "warning"
