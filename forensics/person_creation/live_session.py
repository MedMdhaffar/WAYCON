"""Bounded preview preprocessing for one live-camera capture session."""

from __future__ import annotations

import queue
import re
import threading
import time
from copy import deepcopy
from typing import Any, Callable, Mapping, TYPE_CHECKING

from forensics.person_creation.live_chunk_processing import (
    PreprocessedLiveChunk,
    preprocess_live_chunk,
)

if TYPE_CHECKING:
    from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


PreviewCallback = Callable[[dict], Any]
StopCallback = Callable[[], Any]
_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)
_SENTINEL = object()


class LivePreprocessingSessionError(RuntimeError):
    """The preview worker failed or could not be shut down safely."""


def _safe_message(value: Any) -> str:
    return _CAMERA_URI_RE.sub("<camera-source>", str(value))


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
        preprocess: Callable[..., PreprocessedLiveChunk] | None = None,
        queue_retry_seconds: float = 0.05,
    ) -> None:
        self.queue_capacity = max(1, int(queue_capacity))
        self.join_timeout_seconds = max(0.01, float(join_timeout_seconds))
        self._base_state = {
            "person_name": str(base_state.get("person_name") or ""),
        }
        self._notify_callback = notify
        self._request_stop = request_stop
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
        self._capture_completed_chunk_indices: list[int] = []
        self._preprocessing_completed_chunk_indices: list[int] = []
        self._active_preprocessing_chunk: int | None = None
        self._warnings: list[str] = []
        self._worker_error: dict | None = None

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
                "capture_completed_chunks": captured,
                "preprocessing_completed_chunks": completed,
                "preprocessing_pending_chunks": max(0, captured - completed),
                "preprocessing_active_chunk": self._active_preprocessing_chunk,
                "quality_body_crops": len(self._quality_body_crops),
                "quality_face_crops": len(self._quality_face_crops),
                "embedded_faces": len(self._face_embeddings),
                "failed_face_embeddings": len(self._failed_face_embeddings),
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
            }

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
                with self._lock:
                    if chunk_index in self._preprocessing_completed_chunk_indices:
                        raise ValueError(
                            f"Duplicate preprocessed live chunk index: {chunk_index}"
                        )
                    self._quality_body_crops.extend(
                        deepcopy(result.quality_body_crops)
                    )
                    self._quality_face_crops.extend(
                        deepcopy(result.quality_face_crops)
                    )
                    self._face_embeddings.extend(result.face_embeddings)
                    self._failed_face_embeddings.extend(
                        deepcopy(result.failed_face_embeddings)
                    )
                    self._warnings.extend(
                        _safe_message(warning) for warning in result.warnings
                    )
                    self._preprocessing_completed_chunk_indices.append(chunk_index)
                    self._active_preprocessing_chunk = None
                self._publish()
            except Exception as exc:
                chunk_index = (
                    None if item is _SENTINEL else getattr(item, "chunk_index", None)
                )
                self._record_failure(chunk_index, exc)
                return
            finally:
                self._queue.task_done()

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
