"""Live-camera ingestion node compatible with the existing video pipeline."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
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
    return os.getenv(
        "PERSON_CREATION_LIVE_ROLLING_ANALYSIS",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}


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
            if notify is not None:
                notify("processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        "duration_seconds": duration_seconds,
                    },
                })
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
            if notify is not None:
                notify("processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        "duration_seconds": duration_seconds,
                    },
                })
            continue

        if _stop_requested(stop_event):
            break
        if notify is not None:
            notify("processing_live_frames", None)
        metadata = {**source_metadata, "timestamp": item.timestamp}
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
        body_crops.extend(bodies)
        face_crops.extend(faces)
        frames_processed += 1
        if notify is not None:
            notify("processing_live_frames", {
                "stream_stats": {
                    **buffer.stats(frames_processed, warnings=warnings),
                    "duration_seconds": duration_seconds,
                },
            })

    elapsed_seconds = max(0.0, monotonic() - started_at)
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
    )


def process_live_stream(state: PersonCreationState) -> dict:
    from forensics.face_engine.client import FaceEngineClient
    from forensics.person_creation.models.person_detector import get_person_detector

    camera_uri = state["camera_uri"]
    camera_id = str(state.get("camera_id") or "").strip() or None
    duration_seconds = int(state.get("duration_seconds", 30))
    every_n = max(1, int(state.get("process_every_n", 5)))
    config = dict(state.get("live_stream_config") or {})
    overlap_enabled = live_overlap_enabled()
    rolling_enabled = live_rolling_analysis_enabled()
    if rolling_enabled and not overlap_enabled:
        raise ValueError(
            "PERSON_CREATION_LIVE_ROLLING_ANALYSIS requires "
            "PERSON_CREATION_LIVE_OVERLAP=1."
        )
    buffer_max_size = max(1, min(int(config.get("buffer_max_size", 30)), 300))
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
    vlm_session = None
    vlm_closed = False
    vlm_drain_timeout = max(
        0.0,
        float(config.get("vlm_drain_timeout_seconds", 5.0)),
    )

    def close_vlm() -> dict | None:
        nonlocal vlm_closed
        if vlm_session is None:
            return None
        if not vlm_closed:
            vlm_closed = True
            return vlm_session.close(vlm_drain_timeout)
        return vlm_session.public_snapshot()

    def current_analysis_snapshot() -> dict | None:
        if analysis_session is None:
            return None
        snapshot = analysis_session.public_snapshot()
        if vlm_session is not None:
            snapshot = vlm_session.observe(
                snapshot,
                _collect_identity_decisions(analysis_session),
            )
        return snapshot

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
                _notify(state, "processing_live_frames", {
                    "rolling_analysis": snapshot,
                })

            def publish_vlm(snapshot: dict) -> None:
                _notify(state, "processing_live_frames", {
                    "rolling_analysis": snapshot,
                })

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
                queue_capacity=max(1, int(config.get("vlm_queue_capacity", 2))),
                notify=publish_vlm,
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
        "frames_read": 0,
        "frames_processed": 0,
        "frames_skipped": 0,
        "frames_dropped": 0,
        "body_detections": 0,
        "face_detections": 0,
    }

    try:
        _notify(state, "connecting_camera")
        buffer.start()
        if analysis_session is not None:
            analysis_session.start()
            analysis_started = True
        if preprocessing_session is not None:
            preprocessing_session.start()
            preprocessing_started = True
        _notify(state, "buffering_stream", {
            "stream_stats": {
                **buffer.stats(0),
                "duration_seconds": duration_seconds,
                **({
                    "live_preprocessing": preprocessing_session.public_snapshot(),
                } if preprocessing_session is not None else {}),
                **({
                    "rolling_analysis": current_analysis_snapshot(),
                } if analysis_session is not None else {}),
            },
            **({
                "live_preprocessing": preprocessing_session.public_snapshot(),
            } if preprocessing_session is not None else {}),
            **({
                "rolling_analysis": current_analysis_snapshot(),
            } if analysis_session is not None else {}),
        })
        chunk_index = 0
        while not _stop_requested(stop_event):
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
                    notify=lambda status, update=None: _notify(state, status, update),
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
            totals["frames_read"] += chunk.frames_read
            totals["frames_processed"] += chunk.frames_processed
            totals["frames_skipped"] += chunk.frames_skipped
            totals["frames_dropped"] += chunk.frames_dropped
            totals["body_detections"] += chunk.body_detection_count
            totals["face_detections"] += chunk.face_detection_count
            if preprocessing_session is not None:
                preprocessing_session.submit_chunk(chunk)

            summary = chunk.report_metrics()
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
                **({
                    "rolling_analysis": current_analysis_snapshot(),
                } if analysis_session is not None else {}),
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
        print("[process_live_stream] before buffer.stop", flush=True)
        reader_lifecycle_error: Exception | None = None
        try:
            buffer.stop()
            print("[process_live_stream] after buffer.stop", flush=True)
        except Exception as exc:
            reader_lifecycle_error = exc

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

        try:
            if preprocessing_session is not None and preprocessing_started:
                print(
                    "[process_live_stream] before preprocessing drain",
                    flush=True,
                )
                preprocessing_session.finish()
                final_preprocessing_snapshot = (
                    preprocessing_session.public_snapshot()
                )
                print(
                    "[process_live_stream] after preprocessing drain",
                    flush=True,
                )
        except Exception:
            if analysis_session is not None and analysis_started:
                analysis_session.abort()
            close_vlm()
            raise
        else:
            if analysis_session is not None and analysis_started:
                print(
                    "[process_live_stream] before rolling analysis drain",
                    flush=True,
                )
                try:
                    analysis_session.finish(
                        preprocessing_session.accumulator_version
                    )
                finally:
                    final_analysis_snapshot = close_vlm()
                print(
                    "[process_live_stream] after rolling analysis drain",
                    flush=True,
                )
            else:
                final_analysis_snapshot = close_vlm()

    if not _stop_requested(stop_event):
        raise RuntimeError(
            "Live camera capture ended without an explicit Stop request."
        )

    session_totals = dict(totals)
    completed_chunks = len(chunk_summaries)
    last_chunk = chunk_summaries[-1] if chunk_summaries else None
    stats = buffer.stats(totals["frames_processed"], warnings=warnings)
    stats.update(session_totals)
    stats["duration_seconds"] = duration_seconds
    stats["duration_seconds_per_chunk"] = duration_seconds
    stats["continuous"] = True
    stats["completed_chunks"] = completed_chunks
    stats["stop_requested"] = _stop_requested(stop_event)
    stats["session_totals"] = session_totals
    stats["chunks"] = chunk_summaries
    if final_preprocessing_snapshot is not None:
        stats["live_preprocessing"] = final_preprocessing_snapshot
    if final_analysis_snapshot is not None:
        stats["rolling_analysis"] = final_analysis_snapshot
    stats["consecutive_empty_chunks"] = consecutive_empty_chunks
    _notify(state, "stopping", {
        "chunk_index": last_chunk["chunk_index"] if last_chunk else None,
        "completed_chunks": completed_chunks,
        "last_chunk": last_chunk,
        "session_totals": session_totals,
        "stop_requested": _stop_requested(stop_event),
        "continuous": True,
        "duration_seconds_per_chunk": duration_seconds,
        "stream_stats": stats,
        **({
            "live_preprocessing": final_preprocessing_snapshot,
        } if final_preprocessing_snapshot is not None else {}),
        **({
            "rolling_analysis": final_analysis_snapshot,
        } if final_analysis_snapshot is not None else {}),
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
    log_crop_record_example("live face", face_crops[0] if face_crops else None)
    print(
        f"[process_live_stream] camera={camera_id or 'unassigned'}: "
        f"chunks={completed_chunks} read={stats['frames_read']} "
        f"processed={totals['frames_processed']} dropped={stats['frames_dropped']} "
        f"body={len(body_crops)} face={len(face_crops)}"
    )
    result = {
        "body_crops": body_crops,
        "face_crops": face_crops,
        "camera_uri": "",
        "video_paths": [masked_uri],
        "source_type": "live_camera",
        "source_uri_masked": masked_uri,
        "stream_stats": stats,
        "stream_report_path": report_path,
        # Identities already persisted to Global Memory while capture was live;
        # finalize reuses these instead of registering a second person.
        "live_identity_decisions": _collect_identity_decisions(analysis_session),
    }
    print("[process_live_stream] returning accumulated state", flush=True)
    return result
