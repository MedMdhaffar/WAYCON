"""Test service.py's HTTP surface: the /api/images path-traversal fix, and that
/api/person/start correctly rejects camera_uri (now realtime_main.py's job) while
still accepting offline video-file jobs. No GPU/model/DB access needed --
build_initial_state() and /api/images are pure validation/filesystem logic.

Usage:
    python -m forensics.person_creation.tools.test_service_api
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def test_images_path_traversal_blocked(failures: list[str]) -> None:
    print("=== Scenario 1: /api/images rejects paths outside PROFILE_ROOT ===")
    import forensics.person_creation.service as service
    from forensics.person_identifier.config import Config as PIConfig

    allowed_root = PIConfig.load().PROFILE_ROOT
    allowed_root.mkdir(parents=True, exist_ok=True)

    inside = allowed_root / "test_person" / "face_crops"
    inside.mkdir(parents=True, exist_ok=True)
    inside_file = inside / "a.jpg"
    inside_file.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")

    outside_dir = Path(tempfile.mkdtemp())
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("should not be servable")

    client = service.app.test_client()

    resp = client.get("/api/images", query_string={"path": str(inside_file)})
    print(f"  GET /api/images?path=<inside PROFILE_ROOT> -> {resp.status_code}")
    if resp.status_code != 200:
        failures.append(f"Scenario 1: expected 200 for a path inside PROFILE_ROOT, got {resp.status_code}")

    resp = client.get("/api/images", query_string={"path": str(outside_file)})
    print(f"  GET /api/images?path=<outside PROFILE_ROOT> -> {resp.status_code} {resp.get_json()}")
    if resp.status_code != 403:
        failures.append(f"Scenario 1: expected 403 for a path outside PROFILE_ROOT, got {resp.status_code}")

    resp = client.get("/api/images", query_string={"path": f"{allowed_root}/../../../../etc/passwd"})
    print(f"  GET /api/images?path=<traversal via ..> -> {resp.status_code} {resp.get_json()}")
    if resp.status_code != 403:
        failures.append(f"Scenario 1: expected 403 for a traversal path, got {resp.status_code}")

    resp = client.get("/api/images", query_string={"path": ""})
    print(f"  GET /api/images?path=<empty> -> {resp.status_code}")
    if resp.status_code != 400:
        failures.append(f"Scenario 1: expected 400 for an empty path, got {resp.status_code}")

    outside_file.unlink()
    outside_dir.rmdir()
    print()


def test_camera_uri_rejected_video_file_accepted(failures: list[str]) -> None:
    print("=== Scenario 2: /api/person/start rejects camera_uri, accepts video_file ===")
    from forensics.person_creation.service import build_initial_state, StartRequestError

    try:
        build_initial_state({
            "name": "test",
            "input_type": "camera_uri",
            "camera_uri": "rtsp://user:pass@example.invalid/stream",
        })
        failures.append("Scenario 2: build_initial_state should reject input_type=camera_uri")
    except StartRequestError as exc:
        print(f"  camera_uri correctly rejected: {exc}")

    state = build_initial_state({
        "name": "test",
        "video_paths": ["/tmp/example.mp4"],
    }, validate_video_paths=False)
    print(f"  video_file accepted: input_type={state['input_type']} source_type={state['source_type']}")
    if state["input_type"] != "video_file" or state["source_type"] != "video_file":
        failures.append(f"Scenario 2: unexpected state for a video_file job: {state}")
    if "camera_uri" in state or "source_uri_masked" in state:
        failures.append("Scenario 2: video_file state should carry no camera-related fields")

    print()


def main() -> int:
    print("=== service.py API tests ===\n")
    failures: list[str] = []
    test_images_path_traversal_blocked(failures)
    test_camera_uri_rejected_video_file_accepted(failures)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All service.py API scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
