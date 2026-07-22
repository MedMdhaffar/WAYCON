from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from forensics.person_creation.nodes import filter_quality as quality
from forensics.person_creation.nodes import finalize
from forensics.person_creation.nodes import process_video
from forensics.person_creation.quality_config import DEFAULT_QUALITY_FILTER_CONFIG


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, _frame):
        return list(self.detections)


def _write_records(tmp_path: Path, source_stem: str, source_type: str):
    body_dir = tmp_path / "body"
    face_dir = tmp_path / "face"
    body_dir.mkdir(parents=True)
    face_dir.mkdir(parents=True)
    frame = np.random.default_rng(7).integers(0, 256, (180, 180, 3), dtype=np.uint8)
    source = "rtsp://user:secret@camera.local/live" if source_type == "live_camera" else "video.mp4"
    return process_video.detect_and_save_frame(
        frame,
        frame_idx=25,
        source_stem=source_stem,
        source_metadata={
            "source_type": source_type,
            "video": source,
            "video_path": source,
        },
        body_dir=body_dir,
        face_dir=face_dir,
        person_detector=FakeDetector([{"bbox": [10, 10, 100, 160], "confidence": 0.9}]),
        face_detector=FakeDetector([{"bbox": [30, 20, 110, 100], "confidence": 0.95}]),
    )


def _face(path: Path, *, bbox=None, sharpness=100.0) -> dict:
    return {
        "path": str(path),
        "frame_idx": 1,
        "bbox": bbox or [0, 0, 80, 80],
        "confidence": 0.9,
        "sharpness": sharpness,
        "source_type": "live_camera",
        "video": "<camera-source>",
        "video_path": "<camera-source>",
    }


def test_chunk_zero_live_face_path_exists_and_matches_write(tmp_path):
    bodies, faces = _write_records(tmp_path, "live", "live_camera")

    assert Path(faces[0]["path"]).name == "live_face_f000025_f00.jpg"
    assert Path(faces[0]["path"]).is_file()
    assert Path(bodies[0]["path"]).is_file()


def test_later_chunk_live_face_path_exists_and_matches_write(tmp_path):
    _bodies, faces = _write_records(tmp_path, "live_chunk_0001", "live_camera")

    expected = tmp_path / "face" / "live_chunk_0001_face_f000025_f00.jpg"
    assert faces[0]["path"] == str(expected)
    assert expected.is_file()


def test_recorded_paths_equal_actual_imwrite_arguments(monkeypatch, tmp_path):
    original_imwrite = process_video.cv2.imwrite
    written_paths = []

    def recording_imwrite(path, crop):
        written_paths.append(path)
        return original_imwrite(path, crop)

    monkeypatch.setattr(process_video.cv2, "imwrite", recording_imwrite)
    bodies, faces = _write_records(tmp_path, "live", "live_camera")

    assert bodies[0]["path"] in written_paths
    assert faces[0]["path"] in written_paths


def test_failed_imwrite_is_detected(monkeypatch, tmp_path):
    monkeypatch.setattr(process_video.cv2, "imwrite", lambda *_args: False)

    with pytest.raises(OSError, match="cv2.imwrite failed"):
        process_video._write_crop(
            str(tmp_path / "live_face_f000025_f00.jpg"),
            np.zeros((80, 80, 3), dtype=np.uint8),
        )


def test_missing_and_unreadable_files_have_explicit_reasons(tmp_path):
    missing = _face(tmp_path / "missing.jpg")
    unreadable_path = tmp_path / "unreadable.jpg"
    unreadable_path.write_bytes(b"not an image")

    missing_reason, _ = quality._face_diagnostic(missing)
    unreadable_reason, _ = quality._face_diagnostic(_face(unreadable_path))

    assert missing_reason == "missing_file"
    assert unreadable_reason == "unreadable"


def test_rejection_counts_sum_and_only_five_samples_are_logged(tmp_path, capsys):
    records = [_face(tmp_path / f"missing-{index}.jpg") for index in range(6)]

    result = quality.filter_quality({"body_crops": [], "face_crops": records})
    output = capsys.readouterr().out

    counts = result["face_rejection_counts"]
    assert counts["missing_file"] == 6
    assert sum(counts.values()) == 6
    assert output.count("face rejected sample:") == 5
    assert "missing_file=6" in output


def test_live_and_video_records_expose_required_quality_fields(tmp_path):
    _live_bodies, live_faces = _write_records(tmp_path / "live", "live", "live_camera")
    _video_bodies, video_faces = _write_records(tmp_path / "video", "clip", "video_file")
    required = set(process_video._REQUIRED_QUALITY_FIELDS)

    assert required <= live_faces[0].keys()
    assert required <= video_faces[0].keys()
    assert {key: type(live_faces[0][key]) for key in required} == {
        key: type(video_faces[0][key]) for key in required
    }


def test_schema_example_masks_camera_credentials(tmp_path, capsys):
    _bodies, faces = _write_records(tmp_path, "live", "live_camera")

    process_video.log_crop_record_example("live face", faces[0])
    output = capsys.readouterr().out

    assert "user" not in output
    assert "secret" not in output
    assert "<camera-source>" in output


@pytest.mark.parametrize(
    ("enabled", "detected", "quality", "expected"),
    [
        ("1", 1, 0, True),
        ("0", 1, 0, False),
        ("1", 0, 0, False),
        ("1", 1, 1, False),
    ],
)
def test_keep_staging_activates_only_for_empty_face_condition(
    monkeypatch, enabled, detected, quality, expected
):
    monkeypatch.setenv("PERSON_CREATION_KEEP_STAGING_ON_EMPTY_FACES", enabled)
    state = {
        "face_crops": [{}] * detected,
        "quality_face_crops": [{}] * quality,
        "total_quality_face_crops": quality,
    }

    assert finalize._keep_staging_on_empty_faces(state) is expected


def test_keep_staging_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("PERSON_CREATION_KEEP_STAGING_ON_EMPTY_FACES", raising=False)

    assert finalize._keep_staging_on_empty_faces({
        "face_crops": [{}],
        "quality_face_crops": [],
        "total_quality_face_crops": 0,
    }) is False


def test_quality_thresholds_are_unchanged():
    assert DEFAULT_QUALITY_FILTER_CONFIG.to_dict() == {
        "face_min_width": 50,
        "face_min_height": 50,
        "face_min_sharpness": 20.0,
        "body_min_height": 80,
        "body_min_area": 3000,
        "body_min_sharpness": 50.0,
    }
