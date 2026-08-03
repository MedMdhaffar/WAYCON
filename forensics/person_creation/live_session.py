"""Bounded preview preprocessing for one live-camera capture session."""

from __future__ import annotations

import queue
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TYPE_CHECKING

import numpy as np

from forensics.person_creation.live_chunk_processing import (
    PreprocessedLiveChunk,
    preprocess_live_chunk,
)

if TYPE_CHECKING:
    from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


PreviewCallback = Callable[[dict], Any]
StopCallback = Callable[[], Any]
VersionCallback = Callable[[int], Any]
_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)
_SENTINEL = object()


class LivePreprocessingSessionError(RuntimeError):
    """The preview worker failed or could not be shut down safely."""


def _safe_message(value: Any) -> str:
    return _CAMERA_URI_RE.sub("<camera-source>", str(value))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenRecord.from_mapping(value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, str):
        return _safe_message(value)
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, FrozenRecord):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class FrozenRecord:
    """Recursively immutable analysis record frozen once at accumulation."""

    items: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FrozenRecord:
        return cls(tuple(
            (str(key), _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ))

    def to_dict(self) -> dict:
        return {key: _thaw_value(value) for key, value in self.items}

    def get(self, key: str, default: Any = None) -> Any:
        for item_key, value in self.items:
            if item_key == key:
                return value
        return default


@dataclass(frozen=True)
class FrozenAnalysisSnapshot:
    version: int
    last_completed_preprocessing_chunk: int | None
    person_name: str
    video_paths: tuple[str, ...]
    identity_clustering_config: FrozenRecord
    quality_body_crops: tuple[FrozenRecord, ...]
    quality_face_crops: tuple[FrozenRecord, ...]
    face_embeddings: tuple[FrozenRecord, ...]
    face_chunk_membership: tuple[tuple[str, int], ...]
    output_dir: str = ""
    source_type: str = "live_camera"
    camera_id: str | None = None
    reid_config: FrozenRecord = FrozenRecord(())
    reid_available: bool = False
    reid_unavailable_reason: str = ""

    def mutable_node_state(self) -> dict:
        return {
            "person_name": self.person_name,
            "video_paths": list(self.video_paths),
            "identity_clustering_config": self.identity_clustering_config.to_dict(),
            "quality_body_crops": [item.to_dict() for item in self.quality_body_crops],
            "quality_face_crops": [item.to_dict() for item in self.quality_face_crops],
            "all_face_embeddings": [item.to_dict() for item in self.face_embeddings],
            "output_dir": self.output_dir,
            "source_type": self.source_type,
            "camera_id": self.camera_id,
            "reid_config": self.reid_config.to_dict(),
            "reid_available": self.reid_available,
            "reid_unavailable_reason": self.reid_unavailable_reason,
        }


class LivePreprocessingSession:
    """Own one bounded queue, one preview worker, and one safe accumulator."""

    def __init__(
        self,
        *,
        base_state: Mapping[str, Any],
        queue_capacity: int = 2,
        join_timeout_seconds: float = 120.0,
        notify: PreviewCallback | None = None,
        request_stop: StopCallback | None = None,
        on_accumulator_advanced: VersionCallback | None = None,
        rolling_analysis: bool = False,
        preprocess: Callable[..., PreprocessedLiveChunk] | None = None,
        queue_retry_seconds: float = 0.05,
    ) -> None:
        self.queue_capacity = max(1, int(queue_capacity))
        self.join_timeout_seconds = max(0.01, float(join_timeout_seconds))
        self._base_state = {
            "person_name": str(base_state.get("person_name") or ""),
            "video_paths": tuple(
                _safe_message(item) for item in base_state.get("video_paths", [])
            ),
            "identity_clustering_config": dict(
                base_state.get("identity_clustering_config") or {}
            ),
            "output_dir": str(base_state.get("output_dir") or ""),
            "source_type": str(base_state.get("source_type") or "live_camera"),
            "camera_id": base_state.get("camera_id"),
            "reid_config": dict(base_state.get("reid_config") or {}),
            "reid_available": bool(base_state.get("reid_available")),
            "reid_unavailable_reason": str(
                base_state.get("reid_unavailable_reason") or ""
            ),
        }
        self._notify_callback = notify
        self._request_stop = request_stop
        self._on_accumulator_advanced = on_accumulator_advanced
        self._rolling_analysis = bool(rolling_analysis)
        self._preprocess = preprocess or preprocess_live_chunk
        self._queue_retry_seconds = max(0.01, float(queue_retry_seconds))
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_capacity)
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._failure_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._sentinel_enqueued = False
        self._closed = False

        self._raw_body_crops: list[dict] = []
        self._raw_face_crops: list[dict] = []
        self._quality_body_crops: list[dict] = []
        self._quality_face_crops: list[dict] = []
        self._face_embeddings: list[dict] = []
        self._failed_face_embeddings: list[dict] = []
        self._frozen_identity_config = (
            FrozenRecord.from_mapping(self._base_state["identity_clustering_config"])
            if self._rolling_analysis
            else None
        )
        self._analysis_quality_body_chunks: tuple[tuple[FrozenRecord, ...], ...] = ()
        self._analysis_quality_face_chunks: tuple[tuple[FrozenRecord, ...], ...] = ()
        self._analysis_embedding_chunks: tuple[tuple[FrozenRecord, ...], ...] = ()
        self._analysis_membership_chunks: tuple[tuple[tuple[str, int], ...], ...] = ()
        self._analysis_membership_paths: set[str] = set()
        self._accumulator_version = 0
        self._capture_completed_chunk_indices: list[int] = []
        self._preprocessing_completed_chunk_indices: list[int] = []
        self._active_preprocessing_chunk: int | None = None
        self._warnings: list[str] = []
        self._worker_error: dict | None = None
        self._maximum_queue_depth = 0
        self._face_rejection_counts: dict[str, int] = {}
        self._duplicate_face_evidence_skipped = 0

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise LivePreprocessingSessionError(
                    "Live preprocessing session has already been started."
                )
            self._started = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="person-creation-live-preprocessing",
                daemon=True,
            )
            thread = self._thread
        thread.start()
        self._publish()

    def submit_chunk(self, chunk: LiveChunkResult) -> None:
        """Append one completed chunk and enqueue it exactly once."""
        self._raise_if_failed()
        chunk_index = int(chunk.chunk_index)
        with self._lock:
            if self._closed or self._sentinel_enqueued:
                raise LivePreprocessingSessionError(
                    "Cannot submit a chunk after preprocessing shutdown began."
                )
            if chunk_index in self._capture_completed_chunk_indices:
                raise ValueError(f"Duplicate live chunk index: {chunk_index}")
            self._capture_completed_chunk_indices.append(chunk_index)
            self._raw_body_crops.extend(deepcopy(chunk.body_crops))
            self._raw_face_crops.extend(deepcopy(chunk.face_crops))

        self._put_with_health_checks(chunk)
        self._publish()

    def finish(self) -> None:
        """Deliver the sentinel, drain all work, and join the sole worker."""
        with self._lifecycle_lock:
            if not self._started:
                return
            if self._closed:
                self._raise_if_failed()
                return
            thread = self._thread

        if self._failure_event.is_set():
            if thread is not None:
                thread.join(timeout=self.join_timeout_seconds)
            self._raise_if_failed()

        with self._lock:
            sentinel_needed = not self._sentinel_enqueued
        if sentinel_needed:
            self._put_with_health_checks(
                _SENTINEL,
                deadline_seconds=self.join_timeout_seconds,
            )
            with self._lock:
                self._sentinel_enqueued = True

        if thread is not None:
            thread.join(timeout=self.join_timeout_seconds)
            if thread.is_alive():
                self._record_failure(
                    None,
                    "Live preprocessing worker did not stop within "
                    f"{self.join_timeout_seconds:g} seconds.",
                )
                raise LivePreprocessingSessionError(self._worker_error_message())

        with self._lifecycle_lock:
            self._closed = True
        self._raise_if_failed()
        self._publish()

    def commit_preprocessed(
        self,
        chunk: LiveChunkResult,
        result: PreprocessedLiveChunk,
        *,
        publish: bool = True,
    ) -> None:
        """Commit core-inference output without routing it through another worker.

        The live core worker already owns quality filtering and embedding.  This
        method enters that completed work into the existing evidence ledger so
        the canonical rolling coordinator remains the sole identity system.
        """
        self._raise_if_failed()
        chunk_index = int(chunk.chunk_index)
        with self._lock:
            if self._closed or self._sentinel_enqueued:
                raise LivePreprocessingSessionError(
                    "Cannot commit evidence after preprocessing shutdown began."
                )
            if chunk_index in self._capture_completed_chunk_indices:
                raise ValueError(f"Duplicate live chunk index: {chunk_index}")
            self._capture_completed_chunk_indices.append(chunk_index)
            self._raw_body_crops.extend(deepcopy(chunk.body_crops))
            self._raw_face_crops.extend(deepcopy(chunk.face_crops))
        self._accumulate_result(chunk_index, result)
        if publish:
            self._publish()

    @property
    def worker_alive(self) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def worker_failed(self) -> bool:
        return self._failure_event.is_set()

    def public_snapshot(self) -> dict:
        """Return JSON-safe counts only; embedding vectors never leave the lane."""
        with self._lock:
            captured = len(self._capture_completed_chunk_indices)
            completed = len(self._preprocessing_completed_chunk_indices)
            return {
                "enabled": True,
                "queue_capacity": self.queue_capacity,
                "queue_depth": self._queue.qsize(),
                "maximum_queue_depth": self._maximum_queue_depth,
                "capture_completed_chunks": captured,
                "preprocessing_completed_chunks": completed,
                "preprocessing_pending_chunks": max(0, captured - completed),
                "preprocessing_active_chunk": self._active_preprocessing_chunk,
                "quality_body_crops": len(self._quality_body_crops),
                "quality_face_crops": len(self._quality_face_crops),
                "embedded_faces": len(self._face_embeddings),
                "failed_face_embeddings": len(self._failed_face_embeddings),
                "face_rejection_counts": dict(self._face_rejection_counts),
                "duplicate_face_evidence_skipped": (
                    self._duplicate_face_evidence_skipped
                ),
                "accumulator_version": self._accumulator_version,
                "warnings": list(self._warnings),
                "worker_failed": self._failure_event.is_set(),
                "worker_error": deepcopy(self._worker_error),
            }

    def records_snapshot(self) -> dict:
        """Return copied crop records and indices without embedding vectors."""
        with self._lock:
            return {
                "raw_body_crops": deepcopy(self._raw_body_crops),
                "raw_face_crops": deepcopy(self._raw_face_crops),
                "quality_body_crops": deepcopy(self._quality_body_crops),
                "quality_face_crops": deepcopy(self._quality_face_crops),
                "failed_face_embeddings": deepcopy(self._failed_face_embeddings),
                "capture_completed_chunk_indices": list(
                    self._capture_completed_chunk_indices
                ),
                "preprocessing_completed_chunk_indices": list(
                    self._preprocessing_completed_chunk_indices
                ),
                "active_preprocessing_chunk": self._active_preprocessing_chunk,
                "warnings": list(self._warnings),
                "worker_error": deepcopy(self._worker_error),
                "embedded_face_count": len(self._face_embeddings),
                "accumulator_version": self._accumulator_version,
            }

    @property
    def accumulator_version(self) -> int:
        with self._lock:
            return self._accumulator_version

    @property
    def rolling_analysis_enabled(self) -> bool:
        return self._rolling_analysis

    def analysis_snapshot(self) -> FrozenAnalysisSnapshot:
        """Capture immutable chunk references, then flatten outside the lock."""
        if not self._rolling_analysis:
            raise LivePreprocessingSessionError(
                "Rolling analysis evidence is disabled for this preprocessing session."
            )
        with self._lock:
            last_chunk = (
                self._preprocessing_completed_chunk_indices[-1]
                if self._preprocessing_completed_chunk_indices
                else None
            )
            version = self._accumulator_version
            body_chunks = self._analysis_quality_body_chunks
            face_chunks = self._analysis_quality_face_chunks
            embedding_chunks = self._analysis_embedding_chunks
            membership_chunks = self._analysis_membership_chunks
        return FrozenAnalysisSnapshot(
            version=version,
            last_completed_preprocessing_chunk=last_chunk,
            person_name=self._base_state["person_name"],
            video_paths=self._base_state["video_paths"],
            identity_clustering_config=self._frozen_identity_config,
            quality_body_crops=tuple(
                record for chunk in body_chunks for record in chunk
            ),
            quality_face_crops=tuple(
                record for chunk in face_chunks for record in chunk
            ),
            face_embeddings=tuple(
                record for chunk in embedding_chunks for record in chunk
            ),
            face_chunk_membership=tuple(
                membership for chunk in membership_chunks for membership in chunk
            ),
            output_dir=self._base_state["output_dir"],
            source_type=self._base_state["source_type"],
            camera_id=self._base_state["camera_id"],
            reid_config=FrozenRecord.from_mapping(self._base_state["reid_config"]),
            reid_available=self._base_state["reid_available"],
            reid_unavailable_reason=self._base_state["reid_unavailable_reason"],
        )

    def _worker_loop(self) -> None:
        while True:
            if self._failure_event.is_set() and self._queue.empty():
                return
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                chunk_index = int(item.chunk_index)
                with self._lock:
                    self._active_preprocessing_chunk = chunk_index
                self._publish()

                result = self._preprocess(
                    chunk=item,
                    base_state=self._base_state,
                    notify=None,
                )
                self._accumulate_result(chunk_index, result)
                self._publish()
            except Exception as exc:
                chunk_index = (
                    None if item is _SENTINEL else getattr(item, "chunk_index", None)
                )
                self._record_failure(chunk_index, exc)
                return
            finally:
                self._queue.task_done()

    def _accumulate_result(
        self,
        chunk_index: int,
        result: PreprocessedLiveChunk,
    ) -> None:
        frozen_body: tuple[FrozenRecord, ...] = ()
        frozen_face: tuple[FrozenRecord, ...] = ()
        frozen_embeddings: tuple[FrozenRecord, ...] = ()
        if self._rolling_analysis:
            frozen_body = tuple(
                FrozenRecord.from_mapping(record)
                for record in result.quality_body_crops
            )
            frozen_face = tuple(
                FrozenRecord.from_mapping(record)
                for record in result.quality_face_crops
            )
            frozen_embeddings = tuple(
                FrozenRecord.from_mapping(record)
                for record in result.face_embeddings
            )
        with self._lock:
            if chunk_index in self._preprocessing_completed_chunk_indices:
                raise ValueError(
                    f"Duplicate preprocessed live chunk index: {chunk_index}"
                )
            self._quality_body_crops.extend(deepcopy(result.quality_body_crops))
            self._quality_face_crops.extend(deepcopy(result.quality_face_crops))
            self._face_embeddings.extend(result.face_embeddings)
            self._failed_face_embeddings.extend(
                deepcopy(result.failed_face_embeddings)
            )
            for reason, count in result.face_rejection_counts.items():
                self._face_rejection_counts[str(reason)] = (
                    self._face_rejection_counts.get(str(reason), 0) + int(count)
                )
            if self._rolling_analysis:
                self._analysis_quality_body_chunks += (frozen_body,)
                self._analysis_quality_face_chunks += (frozen_face,)
                self._analysis_embedding_chunks += (frozen_embeddings,)
                membership_chunk = []
                for record in frozen_embeddings:
                    crop_path = record.get("crop_path")
                    path = str(crop_path) if crop_path else ""
                    if path and path not in self._analysis_membership_paths:
                        self._analysis_membership_paths.add(path)
                        membership_chunk.append((path, chunk_index))
                    elif path:
                        self._duplicate_face_evidence_skipped += 1
                self._analysis_membership_chunks += (tuple(membership_chunk),)
            self._warnings.extend(
                _safe_message(warning) for warning in result.warnings
            )
            self._preprocessing_completed_chunk_indices.append(chunk_index)
            self._accumulator_version += 1
            accumulator_version = self._accumulator_version
            self._active_preprocessing_chunk = None
        if self._on_accumulator_advanced is not None:
            try:
                self._on_accumulator_advanced(accumulator_version)
            except Exception as exc:
                warning = _safe_message(
                    f"Rolling analysis request failed: {exc}"
                )
                with self._lock:
                    self._warnings.append(warning)

    def _put_with_health_checks(
        self,
        item: Any,
        *,
        deadline_seconds: float | None = None,
    ) -> None:
        deadline = (
            None
            if deadline_seconds is None
            else time.monotonic() + max(0.0, float(deadline_seconds))
        )
        while True:
            self._raise_if_failed()
            with self._lifecycle_lock:
                thread = self._thread
            if thread is None or not thread.is_alive():
                raise LivePreprocessingSessionError(
                    "Live preprocessing worker stopped before queued work was accepted."
                )
            timeout = self._queue_retry_seconds
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    message = (
                        "Preprocessing sentinel was not accepted within "
                        f"{deadline_seconds:g} seconds."
                    )
                    self._record_failure(None, message)
                    raise LivePreprocessingSessionError(
                        self._worker_error_message()
                    )
                timeout = min(timeout, remaining)
            try:
                self._queue.put(item, timeout=timeout)
                with self._lock:
                    self._maximum_queue_depth = max(
                        self._maximum_queue_depth,
                        self._queue.qsize(),
                    )
                return
            except queue.Full:
                continue

    def _record_failure(self, chunk_index: Any, error: Any) -> None:
        message = _safe_message(error)
        with self._lock:
            if self._worker_error is None:
                self._worker_error = {
                    "chunk_index": chunk_index,
                    "message": message,
                }
            self._active_preprocessing_chunk = None
        self._failure_event.set()
        if self._request_stop is not None:
            try:
                self._request_stop()
            except Exception:
                pass
        self._publish()

    def _worker_error_message(self) -> str:
        with self._lock:
            error = deepcopy(self._worker_error)
        if not error:
            return "Live preprocessing worker failed."
        index = error.get("chunk_index")
        prefix = "Live preprocessing worker failed"
        if index is not None:
            prefix += f" on chunk {index}"
        return f"{prefix}: {error.get('message', 'unknown error')}"

    def _raise_if_failed(self) -> None:
        if self._failure_event.is_set():
            raise LivePreprocessingSessionError(self._worker_error_message())

    def _publish(self) -> None:
        if self._notify_callback is None:
            return
        snapshot = self.public_snapshot()
        try:
            self._notify_callback(snapshot)
        except Exception as exc:
            warning = _safe_message(
                f"Live preprocessing progress callback failed: {exc}"
            )
            with self._lock:
                self._warnings.append(warning)
