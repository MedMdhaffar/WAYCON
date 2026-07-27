"""Live-camera ingestion node compatible with the existing video pipeline."""

from __future__ import annotations

import os
import math
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable

from forensics.person_creation.live_stream import (
    LiveFrameBuffer,
    mask_camera_uri,
    write_stream_report,
)
from forensics.person_creation.nodes.process_video import (
    detect_and_save_frame,
    log_crop_record_example,
    prepare_staging_dirs,
)
from forensics.person_creation.state import PersonCreationState


_FRAME_GAP_WARNING_PREFIX = "No camera frames arrived for "
_EMPTY_CHUNK_WARNING = "Chunk captured no frames; continuing."
_LIVE_CHUNK_SECONDS_ENV = "PERSON_CREATION_LIVE_CHUNK_SECONDS"
_MINIMUM_LIVE_CHUNK_SECONDS = 1.0
_LIVE_PROCESS_EVERY_N_ENV = "PERSON_CREATION_LIVE_PROCESS_EVERY_N_FRAMES"
_DEFAULT_LIVE_PROCESS_EVERY_N_FRAMES = 3
_VLM_IDENTITY_FIELDS = (
    "vlm_status",
    "vlm_state",
    "clothing_description",
    "clothing",
    "clothing_diagnostics",
    "vlm_error",
    "vlm_version",
    "selected_body_crop",
)


@dataclass
class LiveChunkResult:
    chunk_index: int
    started_at: float
    elapsed_seconds: float
    stop_requested: bool
    frames_read: int
    frames_processed: int
    frames_skipped: int
    frames_dropped: int
    body_crops: list[dict]
    face_crops: list[dict]
    body_detection_count: int
    face_detection_count: int
    warnings: list[str]
    frame_gap_active: bool = False
    fast_flushed: bool = False
    sampling_interval_frames: int = 1
    immediate_face_paths: list[str] = field(default_factory=list)
    immediate_body_paths: list[str] = field(default_factory=list)

    def report_metrics(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "started_at": self.started_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "stop_requested": self.stop_requested,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "frames_dropped": self.frames_dropped,
            "body_detections": self.body_detection_count,
            "face_detections": self.face_detection_count,
            "frame_gap_active": self.frame_gap_active,
            "fast_flushed": self.fast_flushed,
            "sampling_interval_frames": self.sampling_interval_frames,
            "immediate_face_count": len(self.immediate_face_paths),
            "warnings": list(self.warnings),
        }


def _notify(state: dict, status: str, snapshot_update: dict | None = None) -> None:
    callback = state.get("_status_callback")
    if callable(callback):
        callback(status, snapshot_update)


def _collect_identity_decisions(analysis_session: Any | None) -> list[dict]:
    """Return live identity receipts from any rolling-analysis implementation."""
    collect = getattr(analysis_session, "identity_decisions", None)
    if not callable(collect):
        return []
    try:
        return [dict(record) for record in collect()]
    except Exception:
        return []


def _stop_requested(stop_event: Any | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _safe_stream_message(message: str) -> str:
    return re.sub(r"rtsps?://\S+", "<camera-source>", str(message), flags=re.IGNORECASE)


def live_overlap_enabled() -> bool:
    return os.getenv("PERSON_CREATION_LIVE_OVERLAP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def live_rolling_analysis_enabled() -> bool:
    configured = os.getenv("PERSON_CREATION_LIVE_ROLLING_ANALYSIS")
    if configured is None:
        # Rolling analysis consumes the embeddings produced by the overlap lane.
        # Once that lane is enabled, silently requiring a second opt-in leaves
        # valid live observations stranded in the preprocessing accumulator.
        return live_overlap_enabled()
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def validate_live_architecture_configuration(
    *,
    overlap_enabled: bool | None = None,
    rolling_enabled: bool | None = None,
) -> tuple[bool, bool]:
    """Validate the explicit dependency between the live core and analysis."""
    overlap = (
        live_overlap_enabled()
        if overlap_enabled is None else bool(overlap_enabled)
    )
    rolling = (
        live_rolling_analysis_enabled()
        if rolling_enabled is None else bool(rolling_enabled)
    )
    if rolling and not overlap:
        raise ValueError(
            "Invalid live configuration: "
            "PERSON_CREATION_LIVE_ROLLING_ANALYSIS=1 requires "
            "PERSON_CREATION_LIVE_OVERLAP=1."
        )
    return overlap, rolling


def live_chunk_duration_seconds(current_duration: Any) -> float:
    """Resolve the sole live chunk-duration override with a safe lower bound."""
    raw = os.getenv(_LIVE_CHUNK_SECONDS_ENV)
    candidate = current_duration if raw is None or not raw.strip() else raw
    try:
        duration = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_LIVE_CHUNK_SECONDS_ENV} must be a number of seconds."
        ) from exc
    if not math.isfinite(duration) or duration < _MINIMUM_LIVE_CHUNK_SECONDS:
        raise ValueError(
            f"{_LIVE_CHUNK_SECONDS_ENV} must be at least "
            f"{_MINIMUM_LIVE_CHUNK_SECONDS:.1f} seconds."
        )
    return duration


def live_process_every_n_frames(explicit: Any = None) -> int:
    """Resolve live-only sampling without changing offline video behavior."""
    raw = (
        explicit
        if explicit is not None
        else os.getenv(
            _LIVE_PROCESS_EVERY_N_ENV,
            str(_DEFAULT_LIVE_PROCESS_EVERY_N_FRAMES),
        )
    )
    if isinstance(raw, bool):
        raise ValueError(f"{_LIVE_PROCESS_EVERY_N_ENV} must be an integer.")
    raw = str(raw).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_LIVE_PROCESS_EVERY_N_ENV} must be an integer."
        ) from exc
    if value < 1:
        raise ValueError(f"{_LIVE_PROCESS_EVERY_N_ENV} must be at least 1.")
    return value


def _publishable_rolling_snapshot(snapshot: dict | None) -> dict:
    """Keep status empty until rolling evidence or a useful warning exists."""
    if not isinstance(snapshot, dict):
        return {}
    try:
        analyzed = int(snapshot.get("analyzed_embedding_count") or 0)
    except (TypeError, ValueError):
        analyzed = 0
    if (
        analyzed > 0
        or snapshot.get("live_identities")
        or snapshot.get("live_recognition_events")
        or snapshot.get("analysis_warning")
    ):
        return snapshot
    return {}


def capture_live_chunk(
    *,
    buffer: Any,
    duration_seconds: float,
    every_n: int,
    source_stem: str,
    source_metadata: dict,
    body_dir: Any,
    face_dir: Any,
    person_detector: Any,
    face_detector: Any,
    frame_timeout_seconds: float = 5.0,
    stop_event: Any | None = None,
    chunk_index: int = 0,
    notify: Callable[[str, dict | None], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    fast_flush_on_face: bool = False,
    immediate_evidence_callback: Callable[[LiveChunkResult], None] | None = None,
    core_session: Any | None = None,
) -> LiveChunkResult:
    """Capture one finite chunk from an already-running live frame buffer."""
    duration_seconds = max(0.0, float(duration_seconds))
    every_n = max(1, int(every_n))
    frame_timeout_seconds = max(0.01, float(frame_timeout_seconds))
    chunk_index = max(0, int(chunk_index))
    chunk_source_stem = (
        source_stem if chunk_index == 0 else f"{source_stem}_chunk_{chunk_index:04d}"
    )
    started_at = monotonic()
    last_frame_at = started_at
    initial_frames_read = int(buffer.frames_read)
    initial_frames_dropped = int(buffer.frames_dropped)
    frames_processed = 0
    frames_skipped = 0
    body_crops: list[dict] = []
    face_crops: list[dict] = []
    warnings: list[str] = []
    frame_gap_active = False
    fast_flushed = False
    immediate_face_paths: list[str] = []
    immediate_body_paths: list[str] = []
    last_progress_notify = started_at

    while not _stop_requested(stop_event):
        now = monotonic()
        remaining = duration_seconds - (now - started_at)
        if remaining <= 0:
            break

        item = buffer.get(timeout=max(0.01, min(1.0, frame_timeout_seconds, remaining)))
        now = monotonic()
        if item is not None and now - started_at >= duration_seconds:
            break
        if item is None:
            if notify is not None and now - last_progress_notify >= 1.0:
                notify("processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        "duration_seconds": duration_seconds,
                    },
                })
                last_progress_notify = now
            if buffer.error:
                raise OSError(buffer.error)
            if _stop_requested(stop_event):
                break
            if buffer.ended and buffer.empty:
                raise OSError("Live camera reader stopped unexpectedly.")
            if (
                now - last_frame_at >= frame_timeout_seconds
                and not frame_gap_active
            ):
                message = (
                    f"No camera frames arrived for {frame_timeout_seconds:g} seconds; "
                    "the live reader remains active."
                )
                warnings = [
                    warning
                    for warning in warnings
                    if not warning.startswith(_FRAME_GAP_WARNING_PREFIX)
                ]
                warnings.append(message)
                frame_gap_active = True
            continue

        last_frame_at = now
        frame_gap_active = False
        warnings = [
            warning
            for warning in warnings
            if not warning.startswith(_FRAME_GAP_WARNING_PREFIX)
        ]
        if _stop_requested(stop_event):
            break
        if item.frame_idx % every_n != 0:
            frames_skipped += 1
            if notify is not None and now - last_progress_notify >= 1.0:
                notify("processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        "duration_seconds": duration_seconds,
                    },
                })
                last_progress_notify = now
            continue

        if _stop_requested(stop_event):
            break
        if core_session is not None:
            core_session.offer(
                item,
                chunk_index=chunk_index,
                source_stem=chunk_source_stem,
            )
            frames_processed += 1
            if notify is not None and now - last_progress_notify >= 1.0:
                notify("processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        **core_session.public_snapshot(),
                        "duration_seconds": duration_seconds,
                        "sampling_interval_frames": every_n,
                    },
                })
                last_progress_notify = now
            continue
        if notify is not None:
            notify("processing_live_frames", None)
        detection_started = monotonic()
        captured_monotonic = getattr(item, "captured_monotonic", None)
        if captured_monotonic is None:
            captured_monotonic = now
        metadata = {
            **source_metadata,
            "timestamp": item.timestamp,
            "source_frame_timestamp": item.timestamp,
        }
        bodies, faces = detect_and_save_frame(
            item.frame,
            frame_idx=item.frame_idx,
            source_stem=chunk_source_stem,
            source_metadata=metadata,
            body_dir=body_dir,
            face_dir=face_dir,
            person_detector=person_detector,
            face_detector=face_detector,
        )
        face_detected = monotonic()
        for face in faces:
            face["_latency_timing"] = {
                "source_frame_timestamp": item.timestamp,
                "capture_monotonic": float(captured_monotonic),
                "face_detection_started_monotonic": float(detection_started),
                "face_detected_monotonic": float(face_detected),
            }
        body_crops.extend(bodies)
        face_crops.extend(faces)
        frames_processed += 1
        if faces and immediate_evidence_callback is not None:
            micro_chunk = LiveChunkResult(
                chunk_index=-(int(item.frame_idx) + 1),
                started_at=detection_started,
                elapsed_seconds=max(0.0, monotonic() - detection_started),
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
                sampling_interval_frames=every_n,
            )
            immediate_evidence_callback(micro_chunk)
            immediate_face_paths.extend(
                str(face.get("path") or "") for face in faces
            )
            immediate_body_paths.extend(
                str(body.get("path") or "") for body in bodies
            )
        if notify is not None and now - last_progress_notify >= 1.0:
            notify("processing_live_frames", {
                "stream_stats": {
                    **buffer.stats(frames_processed, warnings=warnings),
                    "duration_seconds": duration_seconds,
                },
            })
            last_progress_notify = now
        if fast_flush_on_face and faces:
            # This is the regular chunk ending early, not a second work item.
            # Each crop therefore enters preprocessing and embedding exactly once.
            fast_flushed = True
            break

    elapsed_seconds = max(0.0, monotonic() - started_at)
    if core_session is not None:
        core_records = core_session.records_snapshot()
        body_crops = list(
            (core_records.get("chunk_body_crops") or {}).get(chunk_index, [])
        )
        face_crops = list(
            (core_records.get("chunk_face_crops") or {}).get(chunk_index, [])
        )
        immediate_face_paths = [
            str(face.get("path") or "") for face in face_crops
        ]
        immediate_body_paths = [
            str(body.get("path") or "") for body in body_crops
        ]
    return LiveChunkResult(
        chunk_index=chunk_index,
        started_at=started_at,
        elapsed_seconds=elapsed_seconds,
        stop_requested=_stop_requested(stop_event),
        frames_read=max(0, int(buffer.frames_read) - initial_frames_read),
        frames_processed=frames_processed,
        frames_skipped=frames_skipped,
        frames_dropped=max(0, int(buffer.frames_dropped) - initial_frames_dropped),
        body_crops=body_crops,
        face_crops=face_crops,
        body_detection_count=len(body_crops),
        face_detection_count=len(face_crops),
        warnings=warnings,
        frame_gap_active=frame_gap_active,
        fast_flushed=fast_flushed,
        sampling_interval_frames=every_n,
        immediate_face_paths=immediate_face_paths,
        immediate_body_paths=immediate_body_paths,
    )


def process_live_stream(state: PersonCreationState) -> dict:
    from forensics.face_engine.client import FaceEngineClient
    from forensics.person_creation.models.person_detector import get_person_detector

    camera_uri = state["camera_uri"]
    camera_id = str(state.get("camera_id") or "").strip() or None
    duration_seconds = live_chunk_duration_seconds(
        state.get("duration_seconds", 30)
    )
    every_n = live_process_every_n_frames(state.get("process_every_n"))
    config = dict(state.get("live_stream_config") or {})
    overlap_enabled, rolling_enabled = validate_live_architecture_configuration()
    from forensics.global_memory.identity_policy import IdentityPolicyConfig
    from forensics.person_creation.nodes.identity_config import load_identity_config
    from forensics.person_creation.quality_config import load_quality_filter_config

    effective_configuration = {
        "quality_filter": load_quality_filter_config().to_dict(),
        "live_chunk_seconds": duration_seconds,
        "live_process_every_n_frames": every_n,
        "core_inference_queue_capacity": 2,
        "identity_clustering": load_identity_config(state),
        "identity_policy": IdentityPolicyConfig.from_environment().as_dict(),
    }
    print(
        "[process_live_stream] effective_configuration="
        f"{effective_configuration}",
        flush=True,
    )
    # The reader-facing queue is deliberately fixed: live capture always keeps
    # the two newest frames and never builds a stale backlog.
    buffer_max_size = 2
    frame_timeout_seconds = max(1.0, float(config.get("frame_timeout_seconds", 5.0)))
    masked_uri = mask_camera_uri(camera_uri)
    safe_camera_id = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id or "").strip("_")[:64]
    source_stem = f"live{safe_camera_id}" if safe_camera_id else "live"
    body_dir, face_dir = prepare_staging_dirs(state["output_dir"])
    person_detector = get_person_detector()
    face_detector = FaceEngineClient()
    buffer = LiveFrameBuffer(
        camera_uri,
        max_size=buffer_max_size,
        open_timeout_ms=int(config.get("open_timeout_ms", 10_000)),
        read_timeout_ms=int(config.get("read_timeout_ms", 5_000)),
        shutdown_timeout_seconds=config.get("shutdown_timeout_seconds"),
        reconnect_initial_delay_seconds=float(
            config.get("reconnect_initial_delay_seconds", 0.5)
        ),
        reconnect_max_delay_seconds=float(
            config.get("reconnect_max_delay_seconds", 5.0)
        ),
        reconnect_backoff_multiplier=float(
            config.get("reconnect_backoff_multiplier", 2.0)
        ),
        startup_max_attempts=int(config.get("startup_max_attempts", 3)),
        startup_timeout_seconds=config.get("startup_timeout_seconds"),
        maximum_outage_seconds=config.get("maximum_outage_seconds"),
    )
    stop_event = state.get("_stop_event")
    if overlap_enabled and stop_event is None:
        stop_event = threading.Event()
    preprocessing_session = None
    preprocessing_started = False
    final_preprocessing_snapshot = None
    analysis_session = None
    analysis_started = False
    final_analysis_snapshot = None
    final_canonical_state: dict = {}
    core_session = None
    core_evidence_active = False
    final_core_snapshot: dict = {}
    vlm_session = None
    vlm_closed = False
    vlm_drain_timeout = max(
        0.0,
        float(config.get("vlm_drain_timeout_seconds", 5.0)),
    )
    stop_budget_seconds = max(
        0.1,
        float(config.get("stop_budget_seconds", 30.0)),
    )
    stop_timings: dict[str, float] = {}
    publication_lock = threading.RLock()
    publication_sequence = 0
    latest_published_evidence_version = -1

    def canonical_publication(
        snapshot: dict,
        *,
        vlm_enrichment: bool = False,
    ) -> dict | None:
        """Build the sole live identity publication from canonical analysis."""
        nonlocal publication_sequence, latest_published_evidence_version
        rolling = _publishable_rolling_snapshot(snapshot)
        if not rolling or analysis_session is None:
            return None
        with publication_lock:
            if vlm_enrichment:
                current = _publishable_rolling_snapshot(
                    analysis_session.public_snapshot()
                )
                if not current:
                    return None
                enrichment_by_id = {
                    str(identity.get("live_identity_id") or ""): identity
                    for identity in rolling.get("live_identities") or []
                    if isinstance(identity, dict)
                    and identity.get("live_identity_id")
                }
                current_identities = []
                enriched_clothing: dict[Any, dict] = {}
                incoming_clothing = rolling.get("per_cluster_clothing")
                incoming_clothing = (
                    incoming_clothing
                    if isinstance(incoming_clothing, dict)
                    else {}
                )
                for raw_identity in current.get("live_identities") or []:
                    if not isinstance(raw_identity, dict):
                        continue
                    identity = deepcopy(raw_identity)
                    live_id = str(identity.get("live_identity_id") or "")
                    enrichment = enrichment_by_id.get(live_id)
                    if enrichment is not None:
                        for field in _VLM_IDENTITY_FIELDS:
                            if field in enrichment:
                                identity[field] = deepcopy(enrichment[field])
                        source_cluster = enrichment.get("cluster_label")
                        clothing = incoming_clothing.get(
                            source_cluster,
                            incoming_clothing.get(str(source_cluster)),
                        )
                        if (
                            isinstance(clothing, dict)
                            and (
                                not clothing.get("live_identity_id")
                                or str(clothing.get("live_identity_id")) == live_id
                            )
                        ):
                            current_cluster = identity.get("cluster_label")
                            enriched_clothing[current_cluster] = deepcopy(clothing)
                    current_identities.append(identity)
                current["live_identities"] = current_identities
                current["per_cluster_clothing"] = enriched_clothing
                for key, value in rolling.items():
                    if str(key).startswith("vlm_"):
                        current[key] = deepcopy(value)
                rolling = current
            try:
                evidence_version = int(
                    rolling.get("evidence_version")
                    if rolling.get("evidence_version") is not None
                    else rolling.get("analysis_version") or 0
                )
            except (TypeError, ValueError):
                evidence_version = 0
            if vlm_enrichment:
                evidence_version = max(
                    evidence_version,
                    latest_published_evidence_version,
                )
            elif evidence_version < latest_published_evidence_version:
                return None
            latest_published_evidence_version = max(
                latest_published_evidence_version,
                evidence_version,
            )
            publication_sequence = max(
                publication_sequence + 1,
                int(rolling.get("publication_sequence") or 0),
            )
            generated_at = datetime.now(timezone.utc).isoformat()
            rolling = deepcopy(rolling)
            rolling.update({
                "evidence_version": evidence_version,
                "publication_sequence": publication_sequence,
                "generated_at": generated_at,
            })
            canonical_provider = getattr(analysis_session, "canonical_state", None)
            canonical = (
                canonical_provider()
                if callable(canonical_provider)
                else {}
            )
            clothing = rolling.get("per_cluster_clothing")
            if isinstance(clothing, dict):
                canonical["per_cluster_clothing"] = deepcopy(clothing)
            return {
                **canonical,
                "rolling_analysis": rolling,
            }

    def close_vlm(timeout_seconds: float | None = None) -> dict | None:
        nonlocal vlm_closed
        if vlm_session is None:
            return None
        if not vlm_closed:
            vlm_closed = True
            snapshot = vlm_session.close(
                vlm_drain_timeout
                if timeout_seconds is None
                else min(vlm_drain_timeout, max(0.0, timeout_seconds))
            )
        else:
            snapshot = vlm_session.public_snapshot()
        if (
            analysis_session is not None
            and not snapshot.get("live_identities")
        ):
            analysis_snapshot = analysis_session.public_snapshot()
            analysis_snapshot.update({
                key: value
                for key, value in snapshot.items()
                if str(key).startswith("vlm_")
            })
            snapshot = analysis_snapshot
        return _publishable_rolling_snapshot(snapshot)

    def current_analysis_snapshot(*, decorate_vlm: bool = True) -> dict | None:
        if analysis_session is None:
            return None
        snapshot = analysis_session.public_snapshot()
        if decorate_vlm and vlm_session is not None:
            snapshot = vlm_session.observe(
                snapshot,
                _collect_identity_decisions(analysis_session),
            )
        return _publishable_rolling_snapshot(snapshot)

    if overlap_enabled:
        from forensics.person_creation.live_session import LivePreprocessingSession

        queue_capacity = max(
            1,
            int(config.get("preprocessing_queue_capacity", 2)),
        )
        join_timeout = max(
            0.01,
            float(config.get("preprocessing_join_timeout_seconds", 120.0)),
        )

        def publish_preprocessing(snapshot: dict) -> None:
            _notify(state, "processing_live_frames", {
                "live_preprocessing": snapshot,
            })

        def request_analysis(version: int) -> None:
            if analysis_session is not None:
                analysis_session.request_version(version)

        preprocessing_session = LivePreprocessingSession(
            base_state=state,
            queue_capacity=queue_capacity,
            join_timeout_seconds=join_timeout,
            notify=publish_preprocessing,
            request_stop=stop_event.set,
            on_accumulator_advanced=request_analysis if rolling_enabled else None,
            rolling_analysis=rolling_enabled,
        )
        if rolling_enabled:
            from forensics.person_creation.live_analysis import (
                LiveRollingAnalysisSession,
            )

            analysis_join_timeout = max(
                0.01,
                float(config.get("analysis_join_timeout_seconds", 120.0)),
            )

            def publish_analysis(snapshot: dict) -> None:
                if vlm_session is not None:
                    snapshot = vlm_session.observe(
                        snapshot,
                        _collect_identity_decisions(analysis_session),
                    )
                publication = canonical_publication(snapshot)
                if publication is not None:
                    _notify(state, "processing_live_frames", publication)

            def publish_vlm(snapshot: dict) -> None:
                publication = canonical_publication(
                    snapshot,
                    vlm_enrichment=True,
                )
                if publication is not None:
                    _notify(state, "processing_live_frames", publication)

            from forensics.person_creation.live_vlm import (
                LiveIdentityVLMCoordinator,
            )

            analysis_session = LiveRollingAnalysisSession(
                snapshot_provider=preprocessing_session.analysis_snapshot,
                notify=publish_analysis,
                join_timeout_seconds=analysis_join_timeout,
                job_id=state.get("_job_id"),
                identity_decisions=True,
            )
            vlm_session = LiveIdentityVLMCoordinator(
                job_id=str(state.get("_job_id") or "live-job"),
                queue_capacity=min(
                    2,
                    max(1, int(config.get("vlm_queue_capacity", 2))),
                ),
                notify=publish_vlm,
                core_work_pending=lambda: bool(
                    core_session is not None and core_session.has_pending_work()
                ),
                core_queue_depth=lambda: (
                    int(core_session.public_snapshot().get("core_queue_depth") or 0)
                    if core_session is not None
                    else 0
                ),
            )
        from forensics.person_creation.live_core import LiveCoreInferenceSession

        core_session = LiveCoreInferenceSession(
            base_state=state,
            preprocessing_session=preprocessing_session,
            person_detector=person_detector,
            face_detector=face_detector,
            body_dir=body_dir,
            face_dir=face_dir,
            source_metadata={
                "source_type": "live_camera",
                "camera_id": camera_id,
                "video": masked_uri,
                "video_path": masked_uri,
                "source_uri": masked_uri,
            },
            queue_capacity=2,
            request_stop=stop_event.set,
        )
    # Evidence is intentionally not pruned here; long sessions can create many
    # crop files in _staging before the unchanged downstream graph runs once.
    body_crops: list[dict] = []
    face_crops: list[dict] = []
    chunk_summaries: list[dict] = []
    warnings: list[str] = []
    active_frame_gap_warning: str | None = None
    consecutive_empty_chunks = 0
    totals = {
        "captured_frames": 0,
        "sampling_selected_frames": 0,
        "sampling_skipped_frames": 0,
        "frame_queue_dropped_frames": 0,
        "processed_frames": 0,
        "face_detector_frames": 0,
        "selected_frames_pending_at_stop": 0,
        "preprocessing_queue_dropped_chunks": 0,
        # Compatibility aliases retained for existing status consumers.
        "frames_read": 0,
        "frames_processed": 0,
        "frames_skipped": 0,
        "frames_dropped": 0,
        "body_detections": 0,
        "face_detections": 0,
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
        "core_queue_depth": 0,
        "core_queue_peak": 0,
    }

    try:
        _notify(state, "connecting_camera")
        if analysis_session is not None:
            analysis_session.start()
            analysis_started = True
        if preprocessing_session is not None:
            preprocessing_session.start()
            preprocessing_started = True
        if core_session is not None:
            core_session.start()
        buffer.start()
        _notify(state, "buffering_stream", {
            "effective_configuration": effective_configuration,
            "stream_stats": {
                **buffer.stats(0),
                "duration_seconds": duration_seconds,
                **({
                    "live_preprocessing": preprocessing_session.public_snapshot(),
                } if preprocessing_session is not None else {}),
                **({
                    "rolling_analysis": current_analysis_snapshot(
                        decorate_vlm=False
                    ),
                } if analysis_session is not None else {}),
            },
            **({
                "live_preprocessing": preprocessing_session.public_snapshot(),
            } if preprocessing_session is not None else {}),
            **({
                "rolling_analysis": current_analysis_snapshot(
                    decorate_vlm=False
                ),
            } if analysis_session is not None else {}),
        })
        chunk_index = 0

        def submit_immediate_evidence(micro_chunk: LiveChunkResult) -> None:
            if preprocessing_session is not None:
                preprocessing_session.submit_chunk(micro_chunk)

        while not _stop_requested(stop_event):
            core_offered_before = (
                int(
                    core_session.public_snapshot().get(
                        "frames_offered_to_core_queue", 0
                    )
                )
                if core_session is not None else 0
            )
            try:
                chunk = capture_live_chunk(
                    buffer=buffer,
                    duration_seconds=duration_seconds,
                    every_n=every_n,
                    source_stem=source_stem,
                    source_metadata={
                        "source_type": "live_camera",
                        "camera_id": camera_id,
                        "video": masked_uri,
                        "video_path": masked_uri,
                        "source_uri": masked_uri,
                    },
                    body_dir=body_dir,
                    face_dir=face_dir,
                    person_detector=person_detector,
                    face_detector=face_detector,
                    frame_timeout_seconds=frame_timeout_seconds,
                    stop_event=stop_event,
                    chunk_index=chunk_index,
                    notify=(
                        None
                        if core_session is not None
                        else lambda status, update=None: _notify(
                            state, status, update
                        )
                    ),
                    fast_flush_on_face=False,
                    immediate_evidence_callback=(
                        None if core_session is not None else (
                            submit_immediate_evidence
                            if preprocessing_session is not None
                            else None
                        )
                    ),
                    core_session=core_session,
                )
            except OSError as exc:
                warning = _safe_stream_message(
                    f"Live camera reader failed: {exc}"
                )
                warnings.append(warning)
                print(f"[process_live_stream] error: {warning}")
                raise OSError(warning) from exc

            body_crops.extend(chunk.body_crops)
            face_crops.extend(chunk.face_crops)
            core_offered_after = (
                int(
                    core_session.public_snapshot().get(
                        "frames_offered_to_core_queue", 0
                    )
                )
                if core_session is not None else core_offered_before
            )
            chunk_used_core = core_offered_after > core_offered_before
            core_evidence_active = core_evidence_active or chunk_used_core
            totals["frames_read"] += chunk.frames_read
            totals["frames_processed"] += chunk.frames_processed
            totals["frames_skipped"] += chunk.frames_skipped
            totals["frames_dropped"] += chunk.frames_dropped
            totals["captured_frames"] += chunk.frames_read
            totals["sampling_selected_frames"] += chunk.frames_processed
            totals["sampling_skipped_frames"] += chunk.frames_skipped
            totals["frame_queue_dropped_frames"] += chunk.frames_dropped
            totals["processed_frames"] += chunk.frames_processed
            totals["face_detector_frames"] += chunk.frames_processed
            totals["body_detections"] += chunk.body_detection_count
            totals["face_detections"] += chunk.face_detection_count
            if preprocessing_session is not None and not chunk_used_core:
                immediate_faces = set(chunk.immediate_face_paths)
                immediate_bodies = set(chunk.immediate_body_paths)
                remaining_faces = [
                    crop for crop in chunk.face_crops
                    if str(crop.get("path") or "") not in immediate_faces
                ]
                remaining_bodies = [
                    crop for crop in chunk.body_crops
                    if str(crop.get("path") or "") not in immediate_bodies
                ]
                if remaining_faces or remaining_bodies:
                    preprocessing_session.submit_chunk(replace(
                        chunk,
                        face_crops=remaining_faces,
                        body_crops=remaining_bodies,
                        face_detection_count=len(remaining_faces),
                        body_detection_count=len(remaining_bodies),
                    ))

            if core_session is not None and core_evidence_active:
                core_metrics = core_session.public_snapshot()
                totals.update({
                    key: core_metrics.get(key, totals.get(key, 0))
                    for key in (
                        "frames_offered_to_core_queue",
                        "frames_enqueued_to_core_queue",
                        "frames_dropped_core_queue_oldest",
                        "frames_consumed_by_person_detector",
                        "frames_sent_to_face_detector",
                        "person_detections",
                        "faces_detected",
                        "faces_rejected_too_small",
                        "faces_rejected_low_sharpness",
                        "faces_rejected_other_quality",
                        "faces_embedded",
                        "duplicate_face_evidence_skipped",
                        "core_queue_depth",
                        "core_queue_peak",
                    )
                })

            summary = chunk.report_metrics()
            summary["sampling_interval_frames"] = every_n
            summary["warnings"] = [
                _safe_stream_message(item) for item in summary.get("warnings", [])
            ]
            frame_gap_warnings = [
                warning
                for warning in summary["warnings"]
                if warning.startswith(_FRAME_GAP_WARNING_PREFIX)
            ]
            continuing_frame_gap = bool(
                active_frame_gap_warning is not None
                and chunk.frame_gap_active
                and chunk.frames_read == 0
            )
            if continuing_frame_gap:
                summary["warnings"] = [
                    warning
                    for warning in summary["warnings"]
                    if not warning.startswith(_FRAME_GAP_WARNING_PREFIX)
                ]
            for warning in summary["warnings"]:
                if (
                    not warning.startswith(_FRAME_GAP_WARNING_PREFIX)
                    and warning not in warnings
                ):
                    warnings.append(warning)

            if chunk.frame_gap_active and not continuing_frame_gap:
                active_frame_gap_warning = (
                    frame_gap_warnings[-1]
                    if frame_gap_warnings
                    else (
                        f"No camera frames arrived for {frame_timeout_seconds:g} seconds; "
                        "the live reader remains active."
                    )
                )
            elif chunk.frames_read > 0:
                active_frame_gap_warning = None

            warnings = [
                warning
                for warning in warnings
                if not warning.startswith(_FRAME_GAP_WARNING_PREFIX)
            ]
            if active_frame_gap_warning is not None:
                warnings.append(active_frame_gap_warning)

            if chunk.frames_read == 0:
                consecutive_empty_chunks += 1
                if (
                    consecutive_empty_chunks == 1
                    and _EMPTY_CHUNK_WARNING not in summary["warnings"]
                ):
                    summary["warnings"].append(_EMPTY_CHUNK_WARNING)
                if _EMPTY_CHUNK_WARNING not in warnings:
                    warnings.append(_EMPTY_CHUNK_WARNING)
            else:
                consecutive_empty_chunks = 0
                warnings = [
                    warning
                    for warning in warnings
                    if warning != _EMPTY_CHUNK_WARNING
                ]
            summary["consecutive_empty_chunks"] = consecutive_empty_chunks
            chunk_summaries.append(summary)

            completed_chunks = len(chunk_summaries)
            session_totals = dict(totals)
            progress_stats = {
                **buffer.stats(totals["frames_processed"], warnings=warnings),
                **session_totals,
                "duration_seconds": duration_seconds,
                "duration_seconds_per_chunk": duration_seconds,
                "sampling_interval_frames": every_n,
                "effective_configuration": effective_configuration,
                "continuous": True,
                "chunk_index": chunk.chunk_index,
                "completed_chunks": completed_chunks,
                "last_chunk": summary,
                "session_totals": session_totals,
                "stop_requested": _stop_requested(stop_event),
                "consecutive_empty_chunks": consecutive_empty_chunks,
            }
            if preprocessing_session is not None:
                progress_stats["live_preprocessing"] = (
                    preprocessing_session.public_snapshot()
                )
            analysis_publication = (
                canonical_publication(current_analysis_snapshot())
                if analysis_session is not None
                else None
            )
            _notify(state, "processing_live_frames", {
                "chunk_index": chunk.chunk_index,
                "completed_chunks": completed_chunks,
                "last_chunk": summary,
                "session_totals": session_totals,
                "stop_requested": _stop_requested(stop_event),
                "continuous": True,
                "duration_seconds_per_chunk": duration_seconds,
                "stream_stats": progress_stats,
                **({
                    "live_preprocessing": preprocessing_session.public_snapshot(),
                } if preprocessing_session is not None else {}),
                **(analysis_publication or {}),
            })
            print(
                f"[process_live_stream] chunk={chunk.chunk_index} "
                f"elapsed={chunk.elapsed_seconds:.1f}s read={chunk.frames_read} "
                f"processed={chunk.frames_processed} body={chunk.body_detection_count} "
                f"face={chunk.face_detection_count} "
                f"stopped={str(chunk.stop_requested).lower()}"
            )
            chunk_index += 1

            if chunk.stop_requested or _stop_requested(stop_event):
                break
            if buffer.error:
                raise OSError(_safe_stream_message(
                    f"Live camera reader failed: {buffer.error}"
                ))
            if buffer.ended and buffer.empty:
                raise OSError(
                    "Live camera reader stopped before an explicit Stop request."
                )
    finally:
        stop_started = time.monotonic()
        stop_deadline = stop_started + stop_budget_seconds
        stop_timings["stop_request_accepted_ms"] = 0.0

        def remaining_stop_budget() -> float:
            return max(0.01, stop_deadline - time.monotonic())

        print("[process_live_stream] before reader shutdown", flush=True)
        reader_lifecycle_error: Exception | None = None
        try:
            stage_started = time.monotonic()
            request_reader_stop = getattr(buffer, "request_stop", None)
            join_reader = getattr(buffer, "join", None)
            if callable(request_reader_stop) and callable(join_reader):
                request_reader_stop()
                stop_timings["reader_stop_signal_ms"] = round(
                    (time.monotonic() - stage_started) * 1000.0,
                    3,
                )
                stage_started = time.monotonic()
                join_reader(min(5.0, remaining_stop_budget()))
            else:
                buffer.stop()
                stop_timings["reader_stop_signal_ms"] = 0.0
            stop_timings["reader_join_ms"] = round(
                (time.monotonic() - stage_started) * 1000.0,
                3,
            )
            stop_timings["reader_release_and_join_ms"] = (
                stop_timings["reader_stop_signal_ms"]
                + stop_timings["reader_join_ms"]
            )
            print("[process_live_stream] after reader shutdown", flush=True)
        except Exception as exc:
            reader_lifecycle_error = exc

        stop_timings["final_accumulator_submission_ms"] = 0.0
        core_preprocessing_deadline = time.monotonic() + min(
            10.0,
            remaining_stop_budget(),
        )
        if core_session is not None:
            stage_started = time.monotonic()
            final_core_snapshot = core_session.finish(
                max(
                    0.0,
                    min(
                        remaining_stop_budget(),
                        core_preprocessing_deadline - time.monotonic(),
                    ),
                )
            )
            stop_timings["core_inference_drain_ms"] = round(
                (time.monotonic() - stage_started) * 1000.0,
                3,
            )
            if final_core_snapshot.get("core_worker_error"):
                warnings.append(
                    _safe_stream_message(
                        final_core_snapshot["core_worker_error"]
                    )
                )

        if reader_lifecycle_error is not None:
            # The reader may still own native capture state. Shut down the other
            # lanes without running a final preview pass, then fail closed.
            cleanup_errors: list[str] = []
            try:
                if preprocessing_session is not None and preprocessing_started:
                    preprocessing_session.finish()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
            try:
                if analysis_session is not None and analysis_started:
                    analysis_session.abort()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
            try:
                close_vlm()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
            if cleanup_errors:
                reader_lifecycle_error.add_note(
                    "Additional live worker cleanup failed: "
                    + ", ".join(cleanup_errors)
                    + "."
                )
            raise reader_lifecycle_error

        preprocessing_error: Exception | None = None
        try:
            if preprocessing_session is not None and preprocessing_started:
                print(
                    "[process_live_stream] before preprocessing drain",
                    flush=True,
                )
                if hasattr(preprocessing_session, "join_timeout_seconds"):
                    preprocessing_session.join_timeout_seconds = min(
                        preprocessing_session.join_timeout_seconds,
                        (
                            max(
                                0.01,
                                core_preprocessing_deadline - time.monotonic(),
                            )
                            if core_session is not None else 10.0
                        ),
                        remaining_stop_budget(),
                    )
                stage_started = time.monotonic()
                preprocessing_session.finish()
                stop_timings["preprocessing_drain_ms"] = round(
                    (time.monotonic() - stage_started) * 1000.0,
                    3,
                )
                final_preprocessing_snapshot = (
                    preprocessing_session.public_snapshot()
                )
                print(
                    "[process_live_stream] after preprocessing drain",
                    flush=True,
                )
        except Exception as exc:
            preprocessing_error = exc
            stop_timings["preprocessing_drain_ms"] = round(
                (time.monotonic() - stage_started) * 1000.0,
                3,
            )
            final_preprocessing_snapshot = (
                preprocessing_session.public_snapshot()
                if preprocessing_session is not None else {}
            )
            warnings.append(
                _safe_stream_message(
                    f"Preprocessing drain incomplete: {type(exc).__name__}"
                )
            )
            if core_session is None:
                if analysis_session is not None and analysis_started:
                    analysis_session.abort()
                close_vlm()
                raise

        if analysis_session is not None and analysis_started:
            print(
                "[process_live_stream] before rolling analysis drain",
                flush=True,
            )
            stage_started = time.monotonic()
            try:
                if hasattr(analysis_session, "join_timeout_seconds"):
                    analysis_session.join_timeout_seconds = min(
                        analysis_session.join_timeout_seconds,
                        5.0,
                        remaining_stop_budget(),
                    )
                analysis_session.finish(
                    preprocessing_session.accumulator_version
                )
            except Exception as exc:
                warnings.append(
                    _safe_stream_message(
                        f"Rolling analysis drain incomplete: {type(exc).__name__}"
                    )
                )
            finally:
                stop_timings["rolling_analysis_drain_ms"] = round(
                    (time.monotonic() - stage_started) * 1000.0,
                    3,
                )
                collect_canonical = getattr(
                    analysis_session,
                    "canonical_state",
                    None,
                )
                if callable(collect_canonical):
                    final_canonical_state = collect_canonical()
                stage_started = time.monotonic()
                final_analysis_snapshot = close_vlm(
                    min(5.0, remaining_stop_budget())
                )
                stop_timings["vlm_drain_ms"] = round(
                    (time.monotonic() - stage_started) * 1000.0,
                    3,
                )
            print(
                "[process_live_stream] after rolling analysis drain",
                flush=True,
            )
        else:
            stage_started = time.monotonic()
            final_analysis_snapshot = close_vlm(
                min(5.0, remaining_stop_budget())
            )
            stop_timings["vlm_drain_ms"] = round(
                (time.monotonic() - stage_started) * 1000.0,
                3,
            )
        stop_timings["canonical_reconciliation_ms"] = 0.0
        stop_timings["total_stop_ms"] = round(
            (time.monotonic() - stop_started) * 1000.0,
            3,
        )
        stop_timings["budget_ms"] = round(stop_budget_seconds * 1000.0, 3)

    if not _stop_requested(stop_event):
        raise RuntimeError(
            "Live camera capture ended without an explicit Stop request."
        )

    if core_session is not None and core_evidence_active:
        core_records = core_session.records_snapshot()
        body_crops = list(core_records.get("body_crops") or [])
        face_crops = list(core_records.get("face_crops") or [])
        final_core_snapshot = {
            **core_session.public_snapshot(),
            **final_core_snapshot,
        }
        totals.update({
            key: final_core_snapshot.get(key, totals.get(key, 0))
            for key in (
                "frames_offered_to_core_queue",
                "frames_enqueued_to_core_queue",
                "frames_dropped_core_queue_oldest",
                "frames_consumed_by_person_detector",
                "frames_sent_to_face_detector",
                "person_detections",
                "faces_detected",
                "faces_rejected_too_small",
                "faces_rejected_low_sharpness",
                "faces_rejected_other_quality",
                "faces_embedded",
                "duplicate_face_evidence_skipped",
                "core_queue_depth",
                "core_queue_peak",
            )
        })
        totals["body_detections"] = totals["person_detections"]
        totals["face_detections"] = totals["faces_detected"]
        totals["processed_frames"] = totals[
            "frames_consumed_by_person_detector"
        ]
        totals["face_detector_frames"] = totals["frames_sent_to_face_detector"]

    session_totals = dict(totals)
    session_totals["captured_frames"] = (
        int(buffer.frames_read)
        if core_evidence_active else int(totals["captured_frames"])
    )
    session_totals["frames_read"] = (
        int(buffer.frames_read)
        if core_evidence_active else int(totals["frames_read"])
    )
    session_totals["frames_selected_by_stride"] = int(
        totals["sampling_selected_frames"]
    )
    session_totals["frame_queue_dropped_frames"] = (
        int(buffer.frames_dropped)
        if core_evidence_active else int(totals["frame_queue_dropped_frames"])
    )
    accounted = (
        session_totals["sampling_selected_frames"]
        + session_totals["sampling_skipped_frames"]
        + session_totals["frame_queue_dropped_frames"]
    )
    session_totals["selected_frames_pending_at_stop"] = max(
        0,
        session_totals["captured_frames"] - accounted,
    )
    session_totals["frame_accounting_invariant"] = (
        "captured_frames = sampling_selected_frames + "
        "sampling_skipped_frames + frame_queue_dropped_frames + "
        "selected_frames_pending_at_stop"
    )
    completed_chunks = len(chunk_summaries)
    last_chunk = chunk_summaries[-1] if chunk_summaries else None
    stats = buffer.stats(totals["frames_processed"], warnings=warnings)
    stats.update(session_totals)
    stats["duration_seconds"] = duration_seconds
    stats["duration_seconds_per_chunk"] = duration_seconds
    stats["sampling_interval_frames"] = every_n
    stats["continuous"] = True
    stats["completed_chunks"] = completed_chunks
    stats["stop_requested"] = _stop_requested(stop_event)
    stats["session_totals"] = session_totals
    stats["chunks"] = chunk_summaries
    if final_core_snapshot:
        stats["core_inference"] = final_core_snapshot
    canonical_associations = list(
        final_canonical_state.get("associations") or []
    )
    canonical_unattached = list(
        final_canonical_state.get("unattached_bodies") or []
    )
    stats["body_face_diagnostics"] = {
        "person_detections": totals["body_detections"],
        # The current detector runs on full selected frames, not person ROIs.
        "person_rois_submitted_to_face_detection": 0,
        "face_detections_inside_person_rois": 0,
        "accepted_faces": int(
            (final_preprocessing_snapshot or {}).get("quality_face_crops") or 0
        ),
        "faces_associated_with_a_body": len({
            str(item.get("face_path") or "")
            for item in canonical_associations
            if item.get("face_path")
        }),
        "bodies_associated_with_a_face_cluster": len({
            str(item.get("body_path") or "")
            for item in canonical_associations
            if item.get("body_path")
        }),
        "unattached_body_count": len(canonical_unattached),
        "unattached_body_reason": (
            "no_safe_face_cluster_assignment"
            if canonical_unattached else None
        ),
        "reid_reasons": deepcopy(
            final_canonical_state.get("reid_reasons") or {}
        ),
    }
    chunk_durations = [
        float(item.get("elapsed_seconds") or 0.0) for item in chunk_summaries
    ]
    stats["regular_chunk_count"] = sum(
        1 for item in chunk_summaries if not item.get("fast_flushed")
    )
    stats["fast_flush_count"] = sum(
        1 for item in chunk_summaries if item.get("fast_flushed")
    )
    stats["fast_flush_reason"] = None
    stats["suppressed_fast_flush_count"] = 0
    stats["chunk_duration_min_seconds"] = (
        round(min(chunk_durations), 3) if chunk_durations else 0.0
    )
    stats["chunk_duration_average_seconds"] = (
        round(sum(chunk_durations) / len(chunk_durations), 3)
        if chunk_durations else 0.0
    )
    stats["chunk_duration_max_seconds"] = (
        round(max(chunk_durations), 3) if chunk_durations else 0.0
    )
    stats["regular_chunk_elapsed_seconds"] = [
        round(value, 3) for value in chunk_durations
    ]
    if final_preprocessing_snapshot is not None:
        stats["live_preprocessing"] = final_preprocessing_snapshot
        stats["preprocessing_queue_depth"] = int(
            final_preprocessing_snapshot.get("queue_depth") or 0
        )
        stats["preprocessing_queue_peak"] = int(
            final_preprocessing_snapshot.get("maximum_queue_depth") or 0
        )
    if final_analysis_snapshot is not None:
        stats["rolling_analysis"] = final_analysis_snapshot
        for key in (
            "analysis_runs_started",
            "analysis_versions_conflated",
            "analysis_versions_skipped_unchanged",
            "vlm_queue_depth",
            "vlm_queue_peak",
            "vlm_jobs_deferred_for_core_work",
            "vlm_observations_received",
            "vlm_jobs_eligible",
            "vlm_jobs_submitted",
            "vlm_jobs_replaced",
            "vlm_jobs_started",
            "vlm_jobs_completed",
            "vlm_jobs_failed",
            "vlm_jobs_timed_out",
            "vlm_results_merged",
            "vlm_results_published",
        ):
            stats[key] = int(final_analysis_snapshot.get(key) or 0)
        stats["vlm_max_deferral_ms"] = float(
            final_analysis_snapshot.get("vlm_max_deferral_ms") or 0.0
        )
        stats["vlm_last_error"] = final_analysis_snapshot.get("vlm_last_error")
    stats["consecutive_empty_chunks"] = consecutive_empty_chunks
    stats["stop_timings"] = stop_timings
    terminal_publication = (
        canonical_publication(
            final_analysis_snapshot,
            vlm_enrichment=True,
        )
        if final_analysis_snapshot is not None
        else None
    )
    if terminal_publication is not None:
        final_analysis_snapshot = deepcopy(
            terminal_publication["rolling_analysis"]
        )
        stats["rolling_analysis"] = deepcopy(final_analysis_snapshot)
        final_canonical_state = {
            **final_canonical_state,
            **{
                key: deepcopy(value)
                for key, value in terminal_publication.items()
                if key != "rolling_analysis"
            },
        }
    _notify(state, "stopping", {
        "effective_configuration": effective_configuration,
        "chunk_index": last_chunk["chunk_index"] if last_chunk else None,
        "completed_chunks": completed_chunks,
        "last_chunk": last_chunk,
        "session_totals": session_totals,
        "stop_requested": _stop_requested(stop_event),
        "continuous": True,
        "duration_seconds_per_chunk": duration_seconds,
        "sampling_interval_frames": every_n,
        "stream_stats": stats,
        **({
            "live_preprocessing": final_preprocessing_snapshot,
        } if final_preprocessing_snapshot is not None else {}),
        **(terminal_publication or {}),
    })
    print("[process_live_stream] before report write", flush=True)
    report_path = write_stream_report(
        state["output_dir"],
        camera_uri=camera_uri,
        camera_id=camera_id,
        duration_seconds=duration_seconds,
        stats=stats,
    )
    print("[process_live_stream] after report write", flush=True)
    public_face_crops = [
        {
            key: value
            for key, value in crop.items()
            if key != "_latency_timing"
        }
        for crop in face_crops
    ]
    log_crop_record_example(
        "live face",
        public_face_crops[0] if public_face_crops else None,
    )
    print(
        f"[process_live_stream] camera={camera_id or 'unassigned'}: "
        f"chunks={completed_chunks} read={stats['frames_read']} "
        f"processed={totals['frames_processed']} dropped={stats['frames_dropped']} "
        f"body={len(body_crops)} face={len(face_crops)}"
    )
    result = {
        "body_crops": body_crops,
        "face_crops": public_face_crops,
        "camera_uri": "",
        "video_paths": [masked_uri],
        "source_type": "live_camera",
        "source_uri_masked": masked_uri,
        "stream_stats": stats,
        "stream_report_path": report_path,
        # Identities already persisted to Global Memory while capture was live;
        # finalize reuses these instead of registering a second person.
        "live_identity_decisions": _collect_identity_decisions(analysis_session),
        "effective_configuration": effective_configuration,
        **final_canonical_state,
        # Rolling live jobs always finalize from this ledger, including empty
        # or partially enriched sessions.  Falling back to the batch tail after
        # Stop would re-run completed evidence.
        "_canonical_live_state": bool(rolling_enabled),
        "_canonical_live_identities": list(
            (final_analysis_snapshot or {}).get("live_identities") or []
        ),
    }
    print("[process_live_stream] returning accumulated state", flush=True)
    return result
