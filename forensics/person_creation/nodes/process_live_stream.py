"""Process one already-captured segment of in-memory frames from a live camera.

Ingestion (RTSP connect, buffering, reconnect/backoff, presence gating, segment
accumulation) happens entirely outside the graph now -- see gst_stream.GstFrameBuffer,
presence_segmentation.PresenceGatedIngestion/SegmentBatch, and
single_segment_capture.capture_fixed_duration_segment (the legacy fixed-window path
used by service.py's /api/person/start camera_uri contract). This node's only job is
to run the shared person/face detectors over an already-captured stack of frames and
persist quality crops to disk -- the same as process_video.py does for a video file,
but over `state["segment_frames"]` (in-memory numpy arrays) instead of opening a video
path with cv2.VideoCapture.
"""

from __future__ import annotations

import re

from forensics.person_creation.live_stream import write_stream_report
from forensics.person_creation.nodes.process_video import (
    detect_and_save_frame,
    prepare_staging_dirs,
)
from forensics.person_creation.state import PersonCreationState
from forensics.person_creation.status_reporting import notify as _notify


def process_live_stream(state: PersonCreationState) -> dict:
    from forensics.face_engine.local_client import LocalFaceEngine
    from forensics.person_creation.models.person_detector import get_person_detector

    frames = state.get("segment_frames") or []
    timestamps = state.get("segment_frame_timestamps") or []
    segment_id = str(state.get("segment_id") or "unknown_segment")
    camera_id = str(state.get("camera_id") or "").strip() or None
    every_n = max(1, int(state.get("process_every_n", 5)))
    masked_uri = state.get("source_uri_masked", "")
    safe_camera_id = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id or "").strip("_")[:64]
    source_stem = f"seg_{safe_camera_id or 'live'}_{segment_id[:8]}"
    body_dir, face_dir = prepare_staging_dirs(state["output_dir"])
    person_detector = get_person_detector()
    face_detector = LocalFaceEngine()

    _notify("processing_live_frames", {
        "stream_stats": {"segment_id": segment_id, "frame_count": len(frames)},
    })

    body_crops: list[dict] = []
    face_crops: list[dict] = []
    frames_processed = 0
    for idx, frame in enumerate(frames):
        if idx % every_n != 0:
            continue
        ts = timestamps[idx] if idx < len(timestamps) else None
        metadata = {
            "source_type": "live_camera",
            "camera_id": camera_id,
            "segment_id": segment_id,
            "video": masked_uri or f"segment:{segment_id}",
            "video_path": masked_uri or f"segment:{segment_id}",
            "source_uri": masked_uri,
            "timestamp": ts,
        }
        bodies, faces = detect_and_save_frame(
            frame,
            frame_idx=idx,
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
        _notify("processing_live_frames", {
            "stream_stats": {
                "segment_id": segment_id,
                "frame_count": len(frames),
                "frames_processed": frames_processed,
            },
        })

    if not frames:
        raise OSError("process_live_stream received an empty segment (no captured frames).")

    stats = {
        "segment_id": segment_id,
        "frame_count": len(frames),
        "frames_processed": frames_processed,
        "segment_incomplete": bool(state.get("segment_incomplete", False)),
    }
    report_path = write_stream_report(
        state["output_dir"],
        camera_uri=masked_uri,
        camera_id=camera_id,
        duration_seconds=int(state.get("duration_seconds", 0)),
        stats=stats,
    )
    print(
        f"[process_live_stream] segment={segment_id} camera={camera_id or 'unassigned'}: "
        f"frames={len(frames)} processed={frames_processed} body={len(body_crops)} face={len(face_crops)}"
    )
    return {
        "body_crops": body_crops,
        "face_crops": face_crops,
        "camera_uri": "",
        "source_type": "live_camera",
        "stream_stats": stats,
        "stream_report_path": report_path,
        "segment_incomplete": bool(state.get("segment_incomplete", False)),
        # Drop the raw frame tensor stack once consumed -- state holds references
        # (crop file paths) + compact metadata from here on, not the frame stack.
        "segment_frames": [],
    }
