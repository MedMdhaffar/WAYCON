"""Lightweight live-camera configuration smoke checks (no model or camera I/O).

Camera ingestion is no longer HTTP-triggered (see service.py's module docstring
and realtime_main.py) -- service.py's build_initial_state() now rejects
input_type=camera_uri outright. This checks that contract, plus the still-shared
utilities (mask_camera_uri, write_stream_report, route_ingestion, the graph's
node shape) that realtime_main.py and nodes/process_live_stream.py still rely on.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from forensics.person_creation.graph import build_graph, route_ingestion
from forensics.person_creation.live_stream import mask_camera_uri, write_stream_report
from forensics.person_creation.nodes.describe_clothing import async_vlm_config
from forensics.person_creation.service import build_initial_state, StartRequestError


RAW_URI = "rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101"
MASKED_URI = "rtsp://admin:****@192.168.1.64:554/Streaming/Channels/101"


def main() -> None:
    assert mask_camera_uri(RAW_URI) == MASKED_URI

    # service.py's HTTP job-launcher only accepts offline video-file enrollment now.
    try:
        build_initial_state({
            "name": "malek",
            "input_type": "camera_uri",
            "camera_uri": RAW_URI,
            "camera_id": "103",
            "duration_seconds": 30,
        })
        raise AssertionError("build_initial_state should reject input_type=camera_uri")
    except StartRequestError:
        pass

    # But the graph itself still supports a camera_uri-tagged run -- that's what
    # realtime_main.py drives directly, bypassing service.py's HTTP validation.
    assert route_ingestion({"input_type": "camera_uri"}) == "process_live_stream"
    assert route_ingestion({"input_type": "video_file"}) == "process_video"

    video_state = build_initial_state({
        "name": "video-check",
        "video_paths": ["/tmp/example.mp4"],
    }, validate_video_paths=False)
    assert video_state["video_paths"] == ["/tmp/example.mp4"]
    assert route_ingestion(video_state) == "process_video"

    graph = build_graph().get_graph()
    node_ids = set(graph.nodes)
    assert {"process_video", "process_live_stream", "finalize"} <= node_ids
    assert "load_models" not in node_ids  # not a graph node anymore -- see load_models.py

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
