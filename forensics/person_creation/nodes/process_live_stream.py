"""Live-camera ingestion node compatible with the existing video pipeline."""

from __future__ import annotations

import re
import time

from forensics.person_creation.gst_stream import ConnectionState, GstFrameBuffer
from forensics.person_creation.live_stream import (
    mask_camera_uri,
    write_stream_report,
)
from forensics.person_creation.nodes.process_video import (
    detect_and_save_frame,
    prepare_staging_dirs,
)
from forensics.person_creation.state import PersonCreationState


def _notify(state: dict, status: str, snapshot_update: dict | None = None) -> None:
    callback = state.get("_status_callback")
    if callable(callback):
        callback(status, snapshot_update)


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

    # segment_incomplete/reconnect bookkeeping: set by the buffer's reconnect callbacks,
    # read back after the capture loop. Presence-gate integration (marking presence
    # "unknown" during RECONNECTING, restarting debounce after) hooks into the same
    # on_disconnect/on_reconnected callbacks once the Tier 0 presence gate lands.
    reconnect_events: list[dict] = []

    def _on_disconnect(reason: str) -> None:
        reconnect_events.append({"event": "disconnected", "reason": reason, "ts": time.time()})
        _notify(state, "camera_disconnected", {"reconnect_reason": reason})

    def _on_reconnected() -> None:
        reconnect_events.append({"event": "reconnected", "ts": time.time()})
        _notify(state, "camera_reconnected")

    buffer = GstFrameBuffer(
        camera_uri,
        max_size=buffer_max_size,
        codec=str(config.get("codec", "h264")),
        decoder=config.get("decoder"),
        latency_ms=int(config.get("latency_ms", 200)),
        stall_timeout_seconds=frame_timeout_seconds,
        on_disconnect=_on_disconnect,
        on_reconnected=_on_reconnected,
    )

    _notify(state, "connecting_camera")
    buffer.start()
    _notify(state, "buffering_stream", {
        "stream_stats": {
            **buffer.stats(0),
            "duration_seconds": duration_seconds,
        },
    })

    started_at = time.monotonic()
    last_frame_at = started_at
    frames_processed = 0
    body_crops: list[dict] = []
    face_crops: list[dict] = []
    warnings: list[str] = []

    try:
        while time.monotonic() - started_at < duration_seconds:
            item = buffer.get(timeout=min(1.0, frame_timeout_seconds))
            now = time.monotonic()
            if item is None:
                # buffer.error is set while GstFrameBuffer is mid-reconnect (see
                # gst_stream.py); it is not fatal by itself -- the buffer keeps retrying
                # with backoff in the background. If the outage outlasts
                # frame_timeout_seconds the branch below ends this segment early anyway.
                if buffer.ended and buffer.empty:
                    if buffer.frames_read == 0:
                        raise OSError(
                            "Camera stream opened but no frames arrived. Check the stream codec and permissions."
                        )
                    break
                if now - last_frame_at >= frame_timeout_seconds:
                    message = f"No camera frames arrived for {frame_timeout_seconds:g} seconds; ending capture early."
                    warnings.append(message)
                    break
                continue

            last_frame_at = now
            if item.frame_idx % every_n != 0:
                _notify(state, "processing_live_frames", {
                    "stream_stats": {
                        **buffer.stats(frames_processed, warnings=warnings),
                        "duration_seconds": duration_seconds,
                    },
                })
                continue
            _notify(state, "processing_live_frames")
            metadata = {
                "source_type": "live_camera",
                "camera_id": camera_id,
                "video": masked_uri,
                "video_path": masked_uri,
                "source_uri": masked_uri,
                "timestamp": item.timestamp,
            }
            bodies, faces = detect_and_save_frame(
                item.frame,
                frame_idx=item.frame_idx,
                source_stem=source_stem,
                source_metadata=metadata,
                body_dir=body_dir,
                face_dir=face_dir,
                person_detector=person_detector,
                face_detector=face_detector,
            )
            body_crops.extend(bodies)
            face_crops.extend(faces)
            frames_processed += 1
            _notify(state, "processing_live_frames", {
                "stream_stats": {
                    **buffer.stats(frames_processed, warnings=warnings),
                    "duration_seconds": duration_seconds,
                },
            })
    finally:
        buffer.stop()

    if buffer.frames_read == 0:
        raise OSError("Camera stream opened but no frames arrived before the capture timeout.")

    stats = buffer.stats(frames_processed, warnings=warnings)
    stats["duration_seconds"] = duration_seconds
    stats["reconnect_events"] = reconnect_events
    segment_incomplete = any(e["event"] == "disconnected" for e in reconnect_events)
    report_path = write_stream_report(
        state["output_dir"],
        camera_uri=camera_uri,
        camera_id=camera_id,
        duration_seconds=duration_seconds,
        stats=stats,
    )
    print(
        f"[process_live_stream] camera={camera_id or 'unassigned'}: "
        f"read={stats['frames_read']} processed={frames_processed} "
        f"dropped={stats['frames_dropped']} body={len(body_crops)} face={len(face_crops)}"
    )
    return {
        "body_crops": body_crops,
        "face_crops": face_crops,
        "camera_uri": "",
        "video_paths": [masked_uri],
        "source_type": "live_camera",
        "source_uri_masked": masked_uri,
        "stream_stats": stats,
        "stream_report_path": report_path,
        "segment_incomplete": segment_incomplete,
    }
