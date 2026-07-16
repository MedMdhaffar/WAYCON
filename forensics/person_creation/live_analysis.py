"""Latest-only rolling identity analysis for an active live session."""

from __future__ import annotations

import json
import re
import stat
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

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
    last_completed_chunk: int | None
    live_identities: tuple[dict, ...]
    active_memberships: tuple[tuple[str, frozenset[str]], ...]
    retired_live_ids: frozenset[str]
    next_live_number: int
    memory_matches: tuple[tuple[str, dict | None], ...]
    events: tuple[dict, ...]
    event_keys: tuple[str, ...]
    next_event_number: int


def _safe_warning(stage: str, version: int, error: BaseException) -> str:
    error_name = type(error).__name__
    return f"Rolling {stage} failed for version {version} ({error_name})."


def _safe_path(value: Any) -> str:
    return _CAMERA_URI_RE.sub("<camera-source>", str(value or ""))


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
    ) -> None:
        self.join_timeout_seconds = max(0.01, float(join_timeout_seconds))
        self._snapshot_provider = snapshot_provider
        self._notify_callback = notify
        self._database_path = Path(database_path) if database_path is not None else None
        self._cluster = cluster
        self._associate = associate
        self._memory_factory = memory_factory

        self._condition = threading.Condition(threading.Lock())
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepting = True
        self._shutdown_requested = False
        self._requested_version = 0
        self._completed_version = 0
        self._analysis_version = 0
        self._publication_sequence = 0
        self._analysis_in_progress = False
        self._analysis_state = "idle"
        self._analyzed_embedding_count = 0
        self._last_attempted_embedding_count: int | None = None
        self._last_successful_embedding_count = 0
        self._has_valid_result = True
        self._last_completed_chunk: int | None = None
        self._warning: str | None = None
        self._worker_error: str | None = None
        self._live_identities: tuple[dict, ...] = ()
        self._active_memberships: dict[str, frozenset[str]] = {}
        self._retired_live_ids: frozenset[str] = frozenset()
        self._next_live_number = 1
        self._memory_matches: dict[str, dict | None] = {}
        self._events: tuple[dict, ...] = ()
        self._event_keys: tuple[str, ...] = ()
        self._next_event_number = 1

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
                self._requested_version = version
                if not self._analysis_in_progress:
                    self._analysis_state = "scheduled"
            self._wake.set()
            self._condition.notify_all()
        self._publish()
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

    def _public_snapshot_locked(self) -> dict:
        return {
            "enabled": True,
            "publication_sequence": self._publication_sequence,
            "requested_version": self._requested_version,
            "analysis_version": self._analysis_version,
            "analysis_state": self._analysis_state,
            "analysis_in_progress": self._analysis_in_progress,
            "analyzed_embedding_count": self._analyzed_embedding_count,
            "last_completed_preprocessing_chunk": self._last_completed_chunk,
            "analysis_warning": self._worker_error or self._warning,
            "live_identities": deepcopy(list(self._live_identities)),
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
                        if self._requested_version <= self._completed_version:
                            if self._shutdown_requested:
                                return
                            self._wake.clear()
                            break
                        target_version = self._requested_version
                        self._analysis_in_progress = True
                        self._analysis_state = "running"

                    snapshot = self._snapshot_provider()
                    embedding_count = len(snapshot.face_embeddings)
                    with self._condition:
                        can_skip = (
                            self._has_valid_result
                            and embedding_count == self._last_successful_embedding_count
                        )
                        self._last_attempted_embedding_count = embedding_count
                    if can_skip:
                        self._complete_without_analysis(snapshot, target_version)
                        continue

                    try:
                        if memory is None:
                            memory = self._open_memory()
                        committed = self._committed_inputs()
                        result = self._analyze(snapshot, memory, committed)
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
    ) -> _AnalysisResult:
        state = snapshot.mutable_node_state()
        clustered = self._cluster_stage()(state)
        if not isinstance(clustered, dict):
            raise TypeError("cluster_identities returned a non-dictionary result")
        state.update(clustered)
        association = self._association_stage()(state)

        clusters = list(state.get("identity_clusters") or [])
        current_memberships = []
        cluster_by_signature: dict[tuple[str, ...], dict] = {}
        for cluster in clusters:
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

        previous_memberships = dict(committed.active_memberships)
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
            representative_path = self._representative_path(cluster)
            match = self._memory_match(memory, cluster.get("representative_embedding"))
            next_matches[identity.session_person_id] = match
            chunks = sorted(
                chunk_by_path[path]
                for path in identity.crop_paths
                if path in chunk_by_path
            )
            body_count = len(
                getattr(association, "cluster_assignments", {}).get(
                    identity.cluster_label,
                    [],
                )
            )
            identities.append({
                "session_person_id": identity.session_person_id,
                "cluster_label": identity.cluster_label,
                "status": "provisional",
                "face_count": len(identity.crop_paths),
                "associated_body_count": body_count,
                "first_seen_chunk": chunks[0] if chunks else None,
                "last_seen_chunk": chunks[-1] if chunks else None,
                "representative_face_path": representative_path,
                "memory_match": dict(match) if match is not None else None,
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
        retired = frozenset(
            set(committed.retired_live_ids)
            | (set(previous_memberships) - active_ids)
        )
        return _AnalysisResult(
            version=snapshot.version,
            embedding_count=len(snapshot.face_embeddings),
            last_completed_chunk=snapshot.last_completed_preprocessing_chunk,
            live_identities=tuple(identities),
            active_memberships=assigned.active_memberships,
            retired_live_ids=retired,
            next_live_number=assigned.next_live_number,
            memory_matches=tuple(sorted(next_matches.items())),
            events=events,
            event_keys=event_keys,
            next_event_number=next_event_number,
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
        memory_matches = {
            live_id: deepcopy(match) for live_id, match in result.memory_matches
        }
        live_identities = tuple(deepcopy(identity) for identity in result.live_identities)
        events = tuple(deepcopy(event) for event in result.events)
        event_keys = tuple(result.event_keys)
        with self._condition:
            if self._requested_version > result.version:
                self._completed_version = max(self._completed_version, result.version)
                self._analysis_in_progress = False
                self._analysis_state = "scheduled"
                self._condition.notify_all()
                return
            self._active_memberships = active_memberships
            self._retired_live_ids = result.retired_live_ids
            self._next_live_number = result.next_live_number
            self._memory_matches = memory_matches
            self._live_identities = live_identities
            self._events = events
            self._event_keys = event_keys
            self._next_event_number = result.next_event_number
            self._completed_version = max(target_version, result.version)
            self._analysis_version = max(target_version, result.version)
            self._last_attempted_embedding_count = result.embedding_count
            self._last_successful_embedding_count = result.embedding_count
            self._has_valid_result = True
            self._analyzed_embedding_count = result.embedding_count
            self._last_completed_chunk = result.last_completed_chunk
            self._warning = None
            self._analysis_in_progress = False
            self._analysis_state = "ready"
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
