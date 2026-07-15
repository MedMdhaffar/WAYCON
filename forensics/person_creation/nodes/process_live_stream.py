"""Live-camera ingestion node compatible with the existing video pipeline."""

from __future__ import annotations

import re
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
            "warnings": list(self.warnings),
        }


def _notify(state: dict, status: str, snapshot_update: dict | None = None) -> None:
    callback = state.get("_status_callback")
    if callable(callback):
        callback(status, snapshot_update)


def _stop_requested(stop_event: Any | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _safe_stream_message(message: str) -> str:
    return re.sub(r"rtsps?://\S+", "<camera-source>", str(message), flags=re.IGNORECASE)


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
            if buffer.error:
                raise OSError(buffer.error)
            if _stop_requested(stop_event):
                break
            if buffer.ended and buffer.empty:
                if buffer.frames_read == 0:
                    raise OSError(
                        "Camera stream opened but no frames arrived. Check the stream codec and permissions."
                    )
                break
            if now - last_frame_at >= frame_timeout_seconds:
                message = (
                    f"No camera frames arrived for {frame_timeout_seconds:g} seconds; "
                    "ending capture early."
                )
                warnings.append(message)
                break
            continue

        last_frame_at = now
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
    )


def process_live_stream(state: PersonCreationState) -> dict:
    from forensics.face_engine.client import FaceEngineClient
    from forensics.person_creation.models.person_detector import get_person_detector

    camera_uri = state["camera_uri"]
    camera_id = str(state.get("camera_id") or "").strip() or None
    duration_seconds = int(state.get("duration_seconds", 30))
    every_n = max(1, int(state.get("process_every_n", 5)))
    config = dict(state.get("live_stream_config") or {})
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
    )
    stop_event = state.get("_stop_event")
    # Evidence is intentionally not pruned here; long sessions can create many
    # crop files in _staging before the unchanged downstream graph runs once.
    body_crops: list[dict] = []
    face_crops: list[dict] = []
    chunk_summaries: list[dict] = []
    warnings: list[str] = []
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
        _notify(state, "buffering_stream", {
            "stream_stats": {
                **buffer.stats(0),
                "duration_seconds": duration_seconds,
            },
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
                if not chunk_summaries:
                    raise
                warning = _safe_stream_message(
                    f"Live stream failed after partial capture: {exc}"
                )
                warnings.append(warning)
                print(f"[process_live_stream] warning: {warning}")
                break

            body_crops.extend(chunk.body_crops)
            face_crops.extend(chunk.face_crops)
            totals["frames_read"] += chunk.frames_read
            totals["frames_processed"] += chunk.frames_processed
            totals["frames_skipped"] += chunk.frames_skipped
            totals["frames_dropped"] += chunk.frames_dropped
            totals["body_detections"] += chunk.body_detection_count
            totals["face_detections"] += chunk.face_detection_count
            warnings.extend(_safe_stream_message(item) for item in chunk.warnings)

            summary = chunk.report_metrics()
            summary["warnings"] = [
                _safe_stream_message(item) for item in summary.get("warnings", [])
            ]
            if chunk.frames_read == 0:
                warning = f"Chunk {chunk.chunk_index} captured no frames; continuing."
                summary["warnings"].append(warning)
                warnings.append(warning)
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
            }
            _notify(state, "processing_live_frames", {
                "chunk_index": chunk.chunk_index,
                "completed_chunks": completed_chunks,
                "last_chunk": summary,
                "session_totals": session_totals,
                "stop_requested": _stop_requested(stop_event),
                "continuous": True,
                "duration_seconds_per_chunk": duration_seconds,
                "stream_stats": progress_stats,
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
                warning = _safe_stream_message(
                    f"Live stream failed after partial capture: {buffer.error}"
                )
                warnings.append(warning)
                break
            if buffer.ended and buffer.empty:
                warnings.append("Live stream ended; using the captured partial evidence.")
                break
    finally:
        print("[process_live_stream] before buffer.stop", flush=True)
        buffer.stop()
        print("[process_live_stream] after buffer.stop", flush=True)

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
    _notify(state, "stopping", {
        "chunk_index": last_chunk["chunk_index"] if last_chunk else None,
        "completed_chunks": completed_chunks,
        "last_chunk": last_chunk,
        "session_totals": session_totals,
        "stop_requested": _stop_requested(stop_event),
        "continuous": True,
        "duration_seconds_per_chunk": duration_seconds,
        "stream_stats": stats,
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
    }
    print("[process_live_stream] returning accumulated state", flush=True)
    return result
