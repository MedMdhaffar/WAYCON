"""Lightweight live-camera configuration smoke checks (no model or camera I/O)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from forensics.person_creation.graph import build_graph, route_ingestion
from forensics.person_creation.live_stream import mask_camera_uri, write_stream_report
from forensics.person_creation.nodes.describe_clothing import async_vlm_config
from forensics.person_creation.service import build_initial_state


RAW_URI = "rtsp://example-user:example-password@camera.example.invalid/live"
MASKED_URI = "rtsp://****@camera.example.invalid/live"


def main() -> None:
    assert mask_camera_uri(RAW_URI) == MASKED_URI

    camera_state = build_initial_state({
        "name": "malek",
        "input_type": "camera_uri",
        "camera_uri": RAW_URI,
        "camera_id": "103",
        "duration_seconds": 30,
        "every_n": 5,
        "output_dir": "forensics/person_db/runs/malek_live_test",
    })
    assert camera_state["camera_uri"] == RAW_URI
    assert camera_state["source_uri_masked"] == MASKED_URI
    assert camera_state["source_type"] == "live_camera"
    assert camera_state["duration_seconds"] == 30
    assert route_ingestion(camera_state) == "process_live_stream"

    video_state = build_initial_state({
        "name": "video-check",
        "video_paths": ["/tmp/example.mp4"],
    }, validate_video_paths=False)
    assert video_state["video_paths"] == ["/tmp/example.mp4"]
    assert route_ingestion(video_state) == "process_video"

    graph = build_graph().get_graph()
    node_ids = set(graph.nodes)
    assert {"process_video", "process_live_stream", "finalize"} <= node_ids

    with tempfile.TemporaryDirectory() as tmp:
        report_path = write_stream_report(
            tmp,
            camera_uri=RAW_URI,
            camera_id="103",
            duration_seconds=30,
            stats={
                "stream_opened": True,
                "frames_read": 12,
                "frames_processed": 3,
                "frames_dropped": 2,
                "buffer_max_size": 30,
                "first_frame_time": "2026-01-01T00:00:00+00:00",
                "last_frame_time": "2026-01-01T00:00:01+00:00",
            },
        )
        report_text = Path(report_path).read_text(encoding="utf-8")
        report = json.loads(report_text)
        assert RAW_URI not in report_text
        assert report["camera_uri_masked"] == MASKED_URI
        assert report["frames_processed"] == 3

    async_enabled, timeout = async_vlm_config()
    assert isinstance(async_enabled, bool)
    assert timeout >= 1
    print("live camera configuration smoke checks: OK")


if __name__ == "__main__":
    main()
