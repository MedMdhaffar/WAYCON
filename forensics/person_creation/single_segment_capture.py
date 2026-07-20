"""Fixed-duration RTSP capture for the legacy /api/person/start camera_uri contract.

Ingestion has moved out of the graph entirely -- see nodes/process_live_stream.py's
docstring. This is the minimal glue that keeps the existing HTTP contract
(camera_uri + duration_seconds, one fixed-length capture window) working without
resurrecting camera polling inside a graph node: it drives gst_stream.GstFrameBuffer
directly and returns a single presence_segmentation.SegmentBatch covering the whole
window.

For continuous, event-triggered, presence-gated capture (10s-capped segments,
opened/closed by actual presence rather than a fixed timer), use
presence_segmentation.PresenceGatedIngestion instead -- that's the realtime main-loop
path this helper is a stopgap for.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from forensics.person_creation.gst_stream import GstFrameBuffer
from forensics.person_creation.presence_segmentation import SegmentBatch, utc_now_iso


def capture_fixed_duration_segment(
    camera_uri: str,
    duration_seconds: int,
    *,
    codec: str = "h264",
    decoder: str | None = None,
    buffer_max_size: int = 30,
    frame_timeout_seconds: float = 5.0,
    status_callback: Callable[[str, dict | None], None] | None = None,
) -> SegmentBatch:
    segment = SegmentBatch(
        segment_id=str(uuid.uuid4()),
        seq_num=1,
        codec=codec,
        segment_start_ts=utc_now_iso(),
    )

    def _on_disconnect(reason: str) -> None:
        segment.segment_incomplete = True
        if status_callback is not None:
            status_callback("camera_disconnected", {"reconnect_reason": reason})

    def _on_reconnected() -> None:
        if status_callback is not None:
            status_callback("camera_reconnected", None)

    buffer = GstFrameBuffer(
        camera_uri,
        max_size=buffer_max_size,
        codec=codec,
        decoder=decoder,
        stall_timeout_seconds=frame_timeout_seconds,
        on_disconnect=_on_disconnect,
        on_reconnected=_on_reconnected,
    )

    if status_callback is not None:
        status_callback("connecting_camera", None)
    buffer.start()

    started = time.monotonic()
    last_frame_at = started
    try:
        while time.monotonic() - started < duration_seconds:
            item = buffer.get(timeout=min(1.0, frame_timeout_seconds))
            now = time.monotonic()
            if item is None:
                if now - last_frame_at >= frame_timeout_seconds:
                    if buffer.frames_read == 0:
                        raise OSError(
                            "Camera stream opened but no frames arrived. Check the stream codec and permissions."
                        )
                    segment.segment_incomplete = True
                    break
                continue

            last_frame_at = now
            segment.add_frame(item.frame, item.timestamp)
            if status_callback is not None:
                status_callback("buffering_stream", {
                    "stream_stats": {
                        "frames_read": buffer.frames_read,
                        "frames_dropped": buffer.frames_dropped,
                        "duration_seconds": duration_seconds,
                    },
                })
    finally:
        buffer.stop()

    if buffer.frames_read == 0:
        raise OSError("Camera stream opened but no frames arrived before the capture timeout.")

    segment.close(reason="duration_elapsed", incomplete=segment.segment_incomplete)
    return segment
