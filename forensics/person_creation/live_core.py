"""Non-starving bounded core-inference lane for live camera evidence."""

from __future__ import annotations

import queue
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from forensics.person_creation.live_chunk_processing import preprocess_live_chunk
from forensics.person_creation.nodes.process_live_stream import LiveChunkResult
from forensics.person_creation.nodes.process_video import detect_and_save_frame


_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)


def _safe_message(value: Any) -> str:
    return _CAMERA_URI_RE.sub("<camera-source>", str(value))


@dataclass(frozen=True)
class CoreFrame:
    frame: Any
    chunk_index: int
    source_stem: str


class LiveCoreInferenceSession:
    """Own detection, quality filtering, embedding, and ledger submission.

    Producers never block: the two-slot queue always retains the newest
    selected frames.  Canonical clustering is triggered only after completed
    evidence has entered ``LivePreprocessingSession``.
    """

    def __init__(
        self,
        *,
        base_state: Mapping[str, Any],
        preprocessing_session: Any,
        person_detector: Any,
        face_detector: Any,
        body_dir: Any,
        face_dir: Any,
        source_metadata: Mapping[str, Any],
        queue_capacity: int = 2,
        request_stop: Callable[[], Any] | None = None,
        detect: Callable[..., tuple[list[dict], list[dict]]] | None = None,
        preprocess: Callable[..., Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue_capacity = max(1, int(queue_capacity))
        self._base_state = dict(base_state)
        self._preprocessing_session = preprocessing_session
        self._person_detector = person_detector
        self._face_detector = face_detector
        self._body_dir = body_dir
        self._face_dir = face_dir
        self._source_metadata = dict(source_metadata)
        self._request_stop = request_stop
        self._detect = detect or detect_and_save_frame
        self._preprocess = preprocess or preprocess_live_chunk
        self._monotonic = monotonic

        self._queue: queue.Queue[CoreFrame] = queue.Queue(
            maxsize=self.queue_capacity
        )
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepting = True
        self._discard_results = False
        self._active = False
        self._worker_error: str | None = None
        self._all_body_crops: list[dict] = []
        self._all_face_crops: list[dict] = []
        self._chunk_body_crops: dict[int, list[dict]] = {}
        self._chunk_face_crops: dict[int, list[dict]] = {}
        self._metrics = {
            "frames_offered_to_core_queue": 0,
            "frames_enqueued_to_core_queue": 0,
            "frames_dropped_core_queue_oldest": 0,
            "frames_consumed_by_person_detector": 0,
            "frames_sent_to_face_detector": 0,
            "person_detections": 0,
            "faces_detected": 0,
            "faces_rejected_too_small": 0,
            "faces_rejected_low_sharpness": 0,
            "faces_rejected_other_quality": 0,
            "faces_embedded": 0,
            "duplicate_face_evidence_skipped": 0,
            "core_queue_peak": 0,
        }

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("Live core inference session already started.")
            self._started = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="person-creation-live-core-inference",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def offer(self, frame: Any, *, chunk_index: int, source_stem: str) -> bool:
        """Offer one selected frame without ever waiting for queue capacity."""
        item = CoreFrame(
            frame=frame,
            chunk_index=int(chunk_index),
            source_stem=str(source_stem),
        )
        with self._lock:
            self._metrics["frames_offered_to_core_queue"] += 1
            accepting = self._accepting and self._worker_error is None
        if not accepting:
            return False

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._queue.task_done()
                with self._lock:
                    self._metrics["frames_dropped_core_queue_oldest"] += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._metrics["frames_dropped_core_queue_oldest"] += 1
            return False
        with self._lock:
            self._metrics["frames_enqueued_to_core_queue"] += 1
            self._metrics["core_queue_peak"] = max(
                self._metrics["core_queue_peak"],
                self._queue.qsize(),
            )
        return True

    def has_pending_work(self) -> bool:
        with self._lock:
            active = self._active
        return active or not self._queue.empty()

    def finish(self, timeout_seconds: float) -> dict:
        """Drain for a bounded interval; late active results are discarded."""
        timeout = max(0.0, float(timeout_seconds))
        with self._lock:
            self._accepting = False
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        timed_out = bool(thread is not None and thread.is_alive())
        if timed_out:
            with self._lock:
                self._discard_results = True
                if self._worker_error is None:
                    self._worker_error = (
                        "Core inference drain exceeded its bounded deadline."
                    )
            self._discard_queued()
        snapshot = self.public_snapshot()
        snapshot["core_drain_timed_out"] = timed_out
        return snapshot

    def public_snapshot(self) -> dict:
        with self._lock:
            result = dict(self._metrics)
            result.update({
                "core_queue_capacity": self.queue_capacity,
                "core_queue_depth": self._queue.qsize(),
                "core_queue_unfinished_tasks": self._queue.unfinished_tasks,
                "core_inference_active": self._active,
                "core_worker_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "core_worker_error": self._worker_error,
            })
        return result

    def records_snapshot(self) -> dict:
        with self._lock:
            return {
                "body_crops": deepcopy(self._all_body_crops),
                "face_crops": deepcopy(self._all_face_crops),
                "chunk_body_crops": deepcopy(self._chunk_body_crops),
                "chunk_face_crops": deepcopy(self._chunk_face_crops),
            }

    def _discard_queued(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                accepting = self._accepting
            if not accepting and self._queue.empty():
                return
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    self._active = True
                    self._metrics["frames_consumed_by_person_detector"] += 1
                    self._metrics["frames_sent_to_face_detector"] += 1
                self._process_frame(item)
            except Exception as exc:
                with self._lock:
                    self._worker_error = _safe_message(
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._accepting = False
                self._discard_queued()
                if self._request_stop is not None:
                    try:
                        self._request_stop()
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._active = False
                self._queue.task_done()

    def _process_frame(self, item: CoreFrame) -> None:
        buffered = item.frame
        detection_started = self._monotonic()
        captured_monotonic = getattr(buffered, "captured_monotonic", None)
        if captured_monotonic is None:
            captured_monotonic = detection_started
        metadata = {
            **self._source_metadata,
            "timestamp": buffered.timestamp,
            "source_frame_timestamp": buffered.timestamp,
        }
        bodies, faces = self._detect(
            buffered.frame,
            frame_idx=buffered.frame_idx,
            source_stem=item.source_stem,
            source_metadata=metadata,
            body_dir=self._body_dir,
            face_dir=self._face_dir,
            person_detector=self._person_detector,
            face_detector=self._face_detector,
        )
        face_detected = self._monotonic()
        for face in faces:
            face["_latency_timing"] = {
                "source_frame_timestamp": buffered.timestamp,
                "capture_monotonic": float(captured_monotonic),
                "face_detection_started_monotonic": float(detection_started),
                "face_detected_monotonic": float(face_detected),
            }
        with self._lock:
            self._metrics["person_detections"] += len(bodies)
            self._metrics["faces_detected"] += len(faces)

        chunk = LiveChunkResult(
            chunk_index=-(int(buffered.frame_idx) + 1),
            started_at=detection_started,
            elapsed_seconds=max(0.0, self._monotonic() - detection_started),
            stop_requested=False,
            frames_read=1,
            frames_processed=1,
            frames_skipped=0,
            frames_dropped=0,
            body_crops=bodies,
            face_crops=faces,
            body_detection_count=len(bodies),
            face_detection_count=len(faces),
            warnings=[],
        )
        result = self._preprocess(
            chunk=chunk,
            base_state=self._base_state,
            notify=None,
        )
        with self._lock:
            discard = self._discard_results
        if discard:
            return
        self._preprocessing_session.commit_preprocessed(
            chunk,
            result,
            publish=False,
        )

        rejected = dict(result.face_rejection_counts or {})
        other_rejected = sum(
            int(count)
            for reason, count in rejected.items()
            if reason not in {"too_small", "low_sharpness"}
        )
        with self._lock:
            self._all_body_crops.extend(deepcopy(bodies))
            self._all_face_crops.extend(deepcopy(faces))
            self._chunk_body_crops.setdefault(item.chunk_index, []).extend(
                deepcopy(bodies)
            )
            self._chunk_face_crops.setdefault(item.chunk_index, []).extend(
                deepcopy(faces)
            )
            self._metrics["faces_rejected_too_small"] += int(
                rejected.get("too_small", 0)
            )
            self._metrics["faces_rejected_low_sharpness"] += int(
                rejected.get("low_sharpness", 0)
            )
            self._metrics["faces_rejected_other_quality"] += other_rejected
            self._metrics["faces_embedded"] += len(result.face_embeddings)
