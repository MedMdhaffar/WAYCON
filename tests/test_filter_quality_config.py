from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from forensics.person_creation.nodes.filter_quality import filter_quality
from forensics.person_creation.quality_config import load_quality_filter_config


def _face(tmp_path: Path, width: int, height: int, sharpness: float = 50.0) -> dict:
    path = tmp_path / f"face-{width}x{height}-{sharpness}.jpg"
    assert cv2.imwrite(str(path), np.full((height, width, 3), 127, dtype=np.uint8))
    return {
        "path": str(path),
        "frame_idx": 1,
        "video": "video.mp4",
        "bbox": [0, 0, width, height],
        "confidence": 0.9,
        "sharpness": sharpness,
    }


def _body(width: int, height: int, sharpness: float = 50.0) -> dict:
    return {
        "path": "body.jpg",
        "frame_idx": 1,
        "video": "video.mp4",
        "bbox": [0, 0, width, height],
        "confidence": 0.9,
        "sharpness": sharpness,
    }


def test_default_configuration_preserves_production_thresholds(monkeypatch):
    for name in (
        "FACE_MIN_WIDTH",
        "FACE_MIN_HEIGHT",
        "FACE_MIN_SHARPNESS",
        "BODY_MIN_HEIGHT",
        "BODY_MIN_AREA",
        "BODY_MIN_SHARPNESS",
    ):
        monkeypatch.delenv(f"PERSON_CREATION_{name}", raising=False)

    assert load_quality_filter_config().to_dict() == {
        "face_min_width": 50,
        "face_min_height": 50,
        "face_min_sharpness": 20.0,
        "body_min_height": 80,
        "body_min_area": 3000,
        "body_min_sharpness": 50.0,
    }


def test_explicit_configuration_overrides_environment(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_WIDTH", "50")
    config = load_quality_filter_config({"face_min_width": 40})

    assert config.face_min_width == 40


def test_environment_configuration_is_used(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_WIDTH", "50")
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_HEIGHT", "50")

    config = load_quality_filter_config()

    assert (config.face_min_width, config.face_min_height) == (50, 50)


def test_environment_configuration_changes_filter_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_WIDTH", "50")
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_HEIGHT", "50")
    face = _face(tmp_path, 50, 50)

    result = filter_quality({"body_crops": [], "face_crops": [face]})

    assert result["quality_face_crops"] == [face]


@pytest.mark.parametrize("minimum", [60, 50, 40])
def test_face_dimension_boundary_is_inclusive(monkeypatch, tmp_path, minimum):
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_WIDTH", str(minimum))
    monkeypatch.setenv("PERSON_CREATION_FACE_MIN_HEIGHT", str(minimum))
    accepted = _face(tmp_path, minimum, minimum)
    rejected_width = _face(tmp_path, minimum - 1, minimum)
    rejected_height = _face(tmp_path, minimum, minimum - 1)
    state = {
        "body_crops": [],
        "face_crops": [accepted, rejected_width, rejected_height],
    }
    result = filter_quality(state)

    assert result["quality_face_crops"] == [accepted]
    assert result["face_rejection_counts"]["too_small"] == 2


def test_face_sharpness_boundary_is_inclusive(tmp_path):
    accepted = _face(tmp_path, 60, 60, 20.0)
    rejected = _face(tmp_path, 60, 60, 19.999)

    result = filter_quality({"body_crops": [], "face_crops": [accepted, rejected]})

    assert result["quality_face_crops"] == [accepted]
    assert result["face_rejection_counts"]["low_sharpness"] == 1


def test_controlled_60_50_40_comparison_uses_identical_crops(monkeypatch, tmp_path):
    crops = [_face(tmp_path, size, size, 80.0) for size in (39, 40, 49, 50, 59, 60)]
    accepted = {}
    for minimum in (60, 50, 40):
        monkeypatch.setenv("PERSON_CREATION_FACE_MIN_WIDTH", str(minimum))
        monkeypatch.setenv("PERSON_CREATION_FACE_MIN_HEIGHT", str(minimum))
        result = filter_quality({"body_crops": [], "face_crops": crops})
        accepted[minimum] = len(result["quality_face_crops"])

    assert accepted == {60: 1, 50: 3, 40: 5}


def test_body_boundaries_and_defaults_are_unchanged():
    accepted = _body(38, 80, 50.0)
    too_short = _body(100, 79, 50.0)
    too_small_area = _body(37, 80, 50.0)
    too_blurry = _body(38, 80, 49.999)

    result = filter_quality({
        "body_crops": [accepted, too_short, too_small_area, too_blurry],
        "face_crops": [],
    })

    assert result["quality_body_crops"] == [accepted]


@pytest.mark.parametrize(
    "settings",
    [
        {"face_min_width": 0},
        {"face_min_height": -1},
        {"face_min_sharpness": -0.1},
        {"body_min_height": 0},
        {"body_min_area": 0},
        {"body_min_sharpness": -1},
        {"not_a_setting": 1},
    ],
)
def test_invalid_configuration_raises_clear_value_error(settings):
    with pytest.raises(ValueError):
        load_quality_filter_config(settings)


def test_filter_output_contract_is_unchanged(tmp_path):
    face = _face(tmp_path, 60, 60)
    result = filter_quality({"body_crops": [], "face_crops": [face]})

    assert set(result) == {
        "quality_body_crops",
        "quality_face_crops",
        "total_quality_body_crops",
        "total_quality_face_crops",
        "face_rejection_counts",
    }
    assert result["quality_face_crops"][0] is face
