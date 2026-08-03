"""Bounded asynchronous clothing inference for persisted live identities."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from forensics.identity_evidence import identity_evidence_key
from forensics.media_paths import (
    IMAGE_EXTENSIONS,
    MediaPathError,
    get_media_root,
    normalize_media_path,
)


_CANONICAL_PERSON_ID = re.compile(r"^person_[0-9]+$")
_CAMERA_URL = re.compile(r"rtsps?://\S+", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:[^\s/]+/)+)[^\s]*")
_VLM_FIELDS = {
    "vlm_status": "not_started",
    "vlm_state": "not_started",
    "clothing_description": "",
    "clothing": {},
    "clothing_diagnostics": [],
    "vlm_error": None,
    "vlm_version": None,
    "selected_body_crop": None,
}


@dataclass(frozen=True)
class LiveVLMJob:
    job_id: str
    live_identity_id: str
    live_identity_version: int
    canonical_person_id: str
    body_crop_path: str
    queued_at: str
    attempt: int = 1


def _safe_error(value: Any) -> str:
    """Return a bounded category-like error without secrets or local paths."""
    text = _CAMERA_URL.sub("<camera-source>", str(value or "inference_error"))
    text = _ABSOLUTE_PATH.sub("<path>", text)
    text = " ".join(text.split())[:160]
    allowed = {
        "image_decode_failed",
        "inference_error",
        "invalid_output",
        "empty_output",
        "no_valid_body_crop",
        "persistence_error",
        "queue_capacity",
        "timeout",
    }
    return text if text in allowed else "inference_error"


def _safe_text(value: Any) -> str:
    text = _CAMERA_URL.sub("<camera-source>", str(value or ""))
    return _ABSOLUTE_PATH.sub("<path>", text).strip()[:1000]


def _default_describe(job: LiveVLMJob) -> dict:
    from forensics.person_creation.nodes.describe_clothing import describe_clothing

    return describe_clothing({
        "per_cluster_best_body_crops": {0: [job.body_crop_path]},
        "best_body_crops": [job.body_crop_path],
    })


def _default_persist(job: LiveVLMJob, clothing: Mapping[str, Any]) -> None:
    from forensics.global_memory import GlobalMemory

    appearance = {
        "date": date.today().isoformat(),
        "clothing_status": "ok",
        "top": clothing.get("top"),
        "bottom": clothing.get("bottom"),
        "shoes": clothing.get("shoes"),
        "full": clothing.get("full"),
    }
    with GlobalMemory() as memory:
        canonical = memory.resolve_canonical_person_id(job.canonical_person_id)
        if canonical != job.canonical_person_id:
            raise RuntimeError("stale canonical identity")
        memory.update_crop_paths(canonical, {
            "face_crops": [],
            "body_crops": [job.body_crop_path],
            "best_body_crops": [job.body_crop_path],
            "appearance": appearance,
            "video_sources": [],
        })


class LiveIdentityVLMCoordinator:
    """One non-blocking bounded VLM lane owned by one live job."""

    def __init__(
        self,
        *,
        job_id: str,
        queue_capacity: int = 2,
        media_root: str | Path | None = None,
        describe: Callable[[LiveVLMJob], dict] | None = None,
        persist: Callable[[LiveVLMJob, Mapping[str, Any]], None] | None = None,
        notify: Callable[[dict], None] | None = None,
        core_work_pending: Callable[[], bool] | None = None,
        core_queue_depth: Callable[[], int] | None = None,
        maximum_core_deferral_seconds: float = 7.5,
        safe_core_queue_depth: int = 1,
    ) -> None:
        self.job_id = str(job_id or "live-job")
        self.queue_capacity = max(1, int(queue_capacity))
        self.media_root = get_media_root(media_root)
        self._describe = describe or _default_describe
        self._persist = persist or _default_persist
        self._notify = notify
        self._core_work_pending = core_work_pending
        self._core_queue_depth = core_queue_depth
        self._maximum_core_deferral_seconds = max(
            0.1,
            float(maximum_core_deferral_seconds),
        )
        self._safe_core_queue_depth = max(0, int(safe_core_queue_depth))

        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[LiveVLMJob] = deque()
        self._current: dict[str, tuple[int, str | None, str | None]] = {}
        self._states: dict[str, dict] = {}
        self._latest_snapshot: dict = {}
        self._active: LiveVLMJob | None = None
        self._accepting = True
        self._stopped = False
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._timed_out = 0
        self._queue_peak = 0
        self._deferred_for_core_work = 0
        self._deferral_started: dict[tuple[str, int, str], float] = {}
        self._maximum_deferral_ms = 0.0
        self._observations_received = 0
        self._jobs_eligible = 0
        self._jobs_submitted = 0
        self._jobs_replaced = 0
        self._jobs_started = 0
        self._results_merged = 0
        self._results_published = 0
        self._last_error: str | None = None
        self._persisted_versions: set[tuple[str, str, int]] = set()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"live-vlm-{self.job_id[:24]}",
            daemon=True,
        )
        self._thread.start()

    def observe(self, snapshot: Mapping[str, Any], receipts=()) -> dict:
        """Accept a rolling snapshot without waiting for VLM inference."""
        receipt_by_id = {
            str(item.get("live_identity_id")): item
            for item in receipts or ()
            if isinstance(item, Mapping) and item.get("live_identity_id")
        }
        submissions: list[LiveVLMJob] = []
        with self._condition:
            self._observations_received += 1
            self._latest_snapshot = deepcopy(dict(snapshot))
            identities = list(self._latest_snapshot.get("live_identities") or [])
            next_current: dict[str, tuple[int, str | None, str | None]] = {}
            for raw_identity in identities:
                if not isinstance(raw_identity, dict):
                    continue
                live_id = str(raw_identity.get("live_identity_id") or "")
                if not live_id or "version" not in raw_identity:
                    continue
                version = int(raw_identity.get("version") or 0)
                receipt = receipt_by_id.get(live_id)
                canonical_person_id = self._canonical_person(receipt)
                face_path = self._canonical_selected_path(
                    receipt,
                    "face",
                    raw_identity.get("best_face_path")
                    or raw_identity.get("representative_face_path"),
                    canonical_person_id,
                )
                body_path = self._canonical_selected_path(
                    receipt,
                    "body",
                    raw_identity.get("best_body_path"),
                    canonical_person_id,
                )
                next_current[live_id] = (
                    version,
                    canonical_person_id,
                    body_path,
                )
                state = self._states.setdefault(live_id, deepcopy(_VLM_FIELDS))
                raw_identity["representative_face_path"] = face_path
                raw_identity["best_face_path"] = face_path
                raw_identity["best_body_path"] = body_path
                raw_identity.update(deepcopy(state))
                eligible = (
                    self._accepting
                    and canonical_person_id is not None
                    and body_path is not None
                )
                if eligible:
                    self._jobs_eligible += 1
                if (
                    eligible
                    and self._should_submit_locked(live_id, version, body_path)
                ):
                    submissions.append(LiveVLMJob(
                        job_id=self.job_id,
                        live_identity_id=live_id,
                        live_identity_version=version,
                        canonical_person_id=canonical_person_id,
                        body_crop_path=body_path,
                        queued_at=datetime.now().isoformat(timespec="milliseconds"),
                        attempt=1,
                    ))
            self._current = next_current
            self._drop_retired_queued_locked(set(next_current))
            for job in submissions:
                self._enqueue_locked(job)
            decorated = self._decorated_locked()
        return decorated

    def public_snapshot(self) -> dict:
        with self._condition:
            return self._decorated_locked()

    def close(self, timeout_seconds: float) -> dict:
        """Stop submissions and drain for at most ``timeout_seconds``."""
        timeout = max(0.0, float(timeout_seconds))
        with self._condition:
            self._accepting = False
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        publish = False
        with self._condition:
            if self._thread.is_alive():
                self._stopped = True
                unfinished = list(self._queue)
                self._queue.clear()
                if self._active is not None:
                    unfinished.append(self._active)
                seen: set[tuple[str, int, str]] = set()
                for job in unfinished:
                    marker = (
                        job.live_identity_id,
                        job.live_identity_version,
                        job.body_crop_path,
                    )
                    if marker in seen:
                        continue
                    seen.add(marker)
                    if self._job_is_current_locked(job):
                        self._set_state_locked(
                            job,
                            status="timed_out",
                            error="timeout",
                        )
                        self._timed_out += 1
                        self._last_error = "timeout"
                        publish = True
                self._condition.notify_all()
            decorated = self._decorated_locked()
        if publish:
            self._publish()
        return decorated

    def _canonical_person(self, receipt: Mapping[str, Any] | None) -> str | None:
        if not receipt:
            return None
        person_id = str(receipt.get("canonical_person_id") or "")
        return person_id if _CANONICAL_PERSON_ID.fullmatch(person_id) else None

    def _canonical_selected_path(
        self,
        receipt: Mapping[str, Any] | None,
        crop_type: str,
        selected_source: Any,
        canonical_person_id: str | None,
    ) -> str | None:
        if receipt is None or canonical_person_id is None:
            return None
        field = (
            "canonical_face_paths" if crop_type == "face"
            else "canonical_body_paths"
        )
        fallback = (
            "persisted_face_crops" if crop_type == "face"
            else "persisted_body_crops"
        )
        candidates = list(receipt.get(field) or receipt.get(fallback) or [])
        valid_candidates: list[str] = []
        for raw in candidates:
            try:
                path = normalize_media_path(
                    str(raw),
                    media_root=self.media_root,
                    allow_legacy_absolute=False,
                    require_exists=True,
                )
            except (MediaPathError, FileNotFoundError, OSError):
                continue
            parts = path.split("/")
            if (
                len(parts) == 3
                and parts[0] == canonical_person_id
                and parts[1] == f"{crop_type}_crops"
                and Path(parts[2]).suffix.lower() in IMAGE_EXTENSIONS
                and "_staging" not in parts
                and "session" not in parts
                and not any(re.fullmatch(r"cluster_[0-9]+", part) for part in parts)
            ):
                valid_candidates.append(path)
        if not valid_candidates:
            return None

        try:
            selected_path = normalize_media_path(
                str(selected_source),
                media_root=self.media_root,
                allow_legacy_absolute=True,
                require_exists=True,
            )
            selected_key = identity_evidence_key(
                selected_path,
                crop_type,
                media_root=self.media_root,
                allow_legacy_absolute=False,
            )
        except (MediaPathError, FileNotFoundError, OSError, ValueError):
            selected_key = None

        if selected_key is not None:
            for path in valid_candidates:
                try:
                    candidate_key = identity_evidence_key(
                        path,
                        crop_type,
                        media_root=self.media_root,
                        allow_legacy_absolute=False,
                    )
                except (MediaPathError, FileNotFoundError, OSError, ValueError):
                    continue
                if candidate_key == selected_key:
                    return path
        return valid_candidates[-1]

    def _should_submit_locked(self, live_id: str, version: int, crop: str) -> bool:
        state = self._states.get(live_id) or {}
        if (
            state.get("vlm_version") == version
            and state.get("selected_body_crop") == crop
            and state.get("vlm_status")
            in {"queued", "processing", "completed", "failed", "timed_out"}
        ):
            return False
        if self._active is not None and (
            self._active.live_identity_id == live_id
            and self._active.live_identity_version == version
            and self._active.body_crop_path == crop
        ):
            return False
        return not any(
            job.live_identity_id == live_id
            and job.live_identity_version == version
            and job.body_crop_path == crop
            for job in self._queue
        )

    def _enqueue_locked(self, job: LiveVLMJob) -> None:
        replacement_index = next(
            (
                index
                for index, queued in enumerate(self._queue)
                if queued.live_identity_id == job.live_identity_id
            ),
            None,
        )
        if replacement_index is not None:
            replaced = self._queue[replacement_index]
            self._deferral_started.pop((
                replaced.live_identity_id,
                replaced.live_identity_version,
                replaced.body_crop_path,
            ), None)
            self._queue[replacement_index] = job
            self._dropped += 1
            self._jobs_replaced += 1
        else:
            if len(self._queue) >= self.queue_capacity:
                dropped = self._queue.popleft()
                self._mark_dropped_locked(dropped)
            self._queue.append(job)
        self._jobs_submitted += 1
        self._queue_peak = max(self._queue_peak, len(self._queue))
        self._set_state_locked(job, status="queued")
        self._condition.notify()

    def _mark_dropped_locked(self, job: LiveVLMJob) -> None:
        self._dropped += 1
        self._deferral_started.pop((
            job.live_identity_id,
            job.live_identity_version,
            job.body_crop_path,
        ), None)
        if self._job_is_current_locked(job):
            self._set_state_locked(job, status="not_started")

    def _drop_retired_queued_locked(self, active_ids: set[str]) -> None:
        retained: deque[LiveVLMJob] = deque()
        for job in self._queue:
            if job.live_identity_id in active_ids:
                retained.append(job)
            else:
                self._dropped += 1
                self._deferral_started.pop((
                    job.live_identity_id,
                    job.live_identity_version,
                    job.body_crop_path,
                ), None)
        self._queue = retained

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and self._accepting and not self._stopped:
                    self._condition.wait()
                if self._stopped or (not self._accepting and not self._queue):
                    return
                pending = self._queue[0]
                pending_key = (
                    pending.live_identity_id,
                    pending.live_identity_version,
                    pending.body_crop_path,
                )
                if (
                    self._accepting
                    and self._core_work_pending is not None
                    and self._core_is_busy()
                ):
                    now = time.monotonic()
                    started = self._deferral_started.setdefault(pending_key, now)
                    elapsed = max(0.0, now - started)
                    self._maximum_deferral_ms = max(
                        self._maximum_deferral_ms,
                        elapsed * 1000.0,
                    )
                    if (
                        elapsed < self._maximum_core_deferral_seconds
                        or (
                            self._current_core_queue_depth()
                            > self._safe_core_queue_depth
                            and elapsed
                            < self._maximum_core_deferral_seconds * 1.25
                        )
                    ):
                        self._deferred_for_core_work += 1
                        self._condition.wait(timeout=0.05)
                        continue
                job = self._queue.popleft()
                self._deferral_started.pop(pending_key, None)
                self._active = job
                self._jobs_started += 1
                if self._job_is_current_locked(job):
                    self._set_state_locked(job, status="processing")
                    should_publish = True
                else:
                    should_publish = False
            if should_publish:
                self._publish()
            else:
                with self._condition:
                    if self._active == job:
                        self._active = None
                    self._condition.notify_all()
                continue

            try:
                result = self._describe(job)
                clothing, diagnostics, failure = self._parse_result(job, result)
            except Exception:
                clothing, diagnostics, failure = None, [], "inference_error"
                with self._condition:
                    self._last_error = "inference_error"

            with self._condition:
                stale = self._stopped or not self._job_is_current_locked(job)
            if not stale and clothing is not None:
                persistence_key = (
                    job.canonical_person_id,
                    job.live_identity_id,
                    job.live_identity_version,
                )
                with self._condition:
                    already_persisted = persistence_key in self._persisted_versions
                if not already_persisted:
                    try:
                        self._persist(job, clothing)
                    except Exception:
                        clothing = None
                        failure = "persistence_error"
                    else:
                        with self._condition:
                            self._persisted_versions.add(persistence_key)

            publish = False
            with self._condition:
                if self._active == job:
                    self._active = None
                if not self._stopped and self._job_is_current_locked(job):
                    if clothing is not None:
                        self._set_state_locked(
                            job,
                            status="completed",
                            description=clothing.get("full"),
                            clothing=clothing,
                            diagnostics=diagnostics,
                        )
                        self._completed += 1
                        self._results_merged += 1
                        self._last_error = None
                    else:
                        status = "timed_out" if failure == "timeout" else "failed"
                        self._set_state_locked(
                            job,
                            status=status,
                            diagnostics=diagnostics,
                            error=failure,
                        )
                        if status == "timed_out":
                            self._timed_out += 1
                        else:
                            self._failed += 1
                        self._last_error = failure or "inference_error"
                    publish = True
                self._condition.notify_all()
            if publish:
                published = self._publish()
                if published and clothing is not None:
                    with self._condition:
                        self._results_published += 1

    def _core_is_busy(self) -> bool:
        try:
            return bool(
                self._core_work_pending is not None
                and self._core_work_pending()
            )
        except Exception:
            return False

    def _current_core_queue_depth(self) -> int:
        try:
            return max(
                0,
                int(self._core_queue_depth() if self._core_queue_depth else 0),
            )
        except Exception:
            return 0

    def _parse_result(
        self,
        job: LiveVLMJob,
        result: Mapping[str, Any],
    ) -> tuple[dict | None, list[dict], str | None]:
        per_cluster = result.get("per_cluster_clothing") or {}
        clothing = per_cluster.get(0, per_cluster.get("0", {}))
        diagnostics: list[dict] = []
        for raw in result.get("clothing_diagnostics") or []:
            item = dict(raw) if isinstance(raw, Mapping) else {}
            item["selected_body_crop"] = job.body_crop_path
            if item.get("failure_reason"):
                item["failure_reason"] = _safe_error(item["failure_reason"])
            diagnostics.append(item)
        if isinstance(clothing, Mapping) and clothing.get("status") == "ok":
            safe = {
                key: _safe_text(clothing.get(key))
                for key in ("top", "bottom", "shoes", "full")
            }
            return safe, diagnostics, None
        reason = (
            clothing.get("failure_reason")
            if isinstance(clothing, Mapping)
            else "inference_error"
        )
        return None, diagnostics, _safe_error(reason)

    def _job_is_current_locked(self, job: LiveVLMJob) -> bool:
        return (
            job.job_id == self.job_id
            and self._current.get(job.live_identity_id)
            == (
                job.live_identity_version,
                job.canonical_person_id,
                job.body_crop_path,
            )
        )

    def _set_state_locked(
        self,
        job: LiveVLMJob,
        *,
        status: str,
        description: Any = "",
        clothing: Mapping[str, Any] | None = None,
        diagnostics: list[dict] | None = None,
        error: Any = None,
    ) -> None:
        public_state = {
            "queued": "pending",
            "processing": "running",
        }.get(status, status)
        self._states[job.live_identity_id] = {
            "vlm_status": status,
            "vlm_state": public_state,
            "clothing_description": _safe_text(description),
            "clothing": deepcopy(dict(clothing or {})),
            "clothing_diagnostics": deepcopy(diagnostics or []),
            "vlm_error": _safe_error(error) if error else None,
            "vlm_version": job.live_identity_version,
            "selected_body_crop": job.body_crop_path,
        }

    def _decorated_locked(self) -> dict:
        snapshot = deepcopy(self._latest_snapshot)
        identities = list(snapshot.get("live_identities") or [])
        per_cluster_clothing: dict[Any, dict] = {}
        for identity in identities:
            if not isinstance(identity, dict):
                continue
            live_id = str(identity.get("live_identity_id") or "")
            state = self._states.get(live_id)
            if state is not None:
                identity.update(deepcopy(state))
                clothing = state.get("clothing")
                if (
                    state.get("vlm_status") == "completed"
                    and isinstance(clothing, dict)
                    and clothing
                ):
                    cluster_label = identity.get("cluster_label")
                    per_cluster_clothing[cluster_label] = {
                        **deepcopy(clothing),
                        "status": "ok",
                        "selected_body_crop": state.get("selected_body_crop"),
                        "live_identity_id": live_id,
                    }
        snapshot.update({
            "per_cluster_clothing": per_cluster_clothing,
            "vlm_queue_depth": len(self._queue),
            "vlm_queue_capacity": self.queue_capacity,
            "vlm_queue_peak": self._queue_peak,
            "vlm_jobs_deferred_for_core_work": self._deferred_for_core_work,
            "vlm_active_jobs": 1 if self._active is not None else 0,
            "vlm_active_identity": (
                self._active.live_identity_id if self._active is not None else None
            ),
            "vlm_completed": self._completed,
            "vlm_failed": self._failed,
            "vlm_dropped": self._dropped,
            "vlm_timed_out": self._timed_out,
            "vlm_observations_received": self._observations_received,
            "vlm_jobs_eligible": self._jobs_eligible,
            "vlm_jobs_submitted": self._jobs_submitted,
            "vlm_jobs_replaced": self._jobs_replaced,
            "vlm_max_deferral_ms": round(self._maximum_deferral_ms, 3),
            "vlm_jobs_started": self._jobs_started,
            "vlm_jobs_completed": self._completed,
            "vlm_jobs_failed": self._failed,
            "vlm_jobs_timed_out": self._timed_out,
            "vlm_results_merged": self._results_merged,
            "vlm_results_published": self._results_published,
            "vlm_last_error": self._last_error,
        })
        return snapshot

    def _publish(self) -> bool:
        if self._notify is None:
            return False
        snapshot = self.public_snapshot()
        try:
            self._notify(snapshot)
        except Exception:
            with self._condition:
                self._last_error = "inference_error"
            return False
        return True
