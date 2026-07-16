from pathlib import Path

import cv2

from forensics.person_creation.quality_config import (
    DEFAULT_QUALITY_FILTER_CONFIG,
    QualityFilterConfig,
    load_quality_filter_config,
)


_MIN_BODY_H = DEFAULT_QUALITY_FILTER_CONFIG.body_min_height
_MIN_BODY_AREA = DEFAULT_QUALITY_FILTER_CONFIG.body_min_area
_MIN_FACE_W = DEFAULT_QUALITY_FILTER_CONFIG.face_min_width
_MIN_FACE_H = DEFAULT_QUALITY_FILTER_CONFIG.face_min_height
_MIN_SHARPNESS = DEFAULT_QUALITY_FILTER_CONFIG.face_min_sharpness
_FACE_REJECTION_REASONS = (
    "missing_file",
    "unreadable",
    "empty",
    "too_small",
    "low_sharpness",
    "bad_brightness",
    "other",
)


def _body_ok(crop: dict, config: QualityFilterConfig | None = None) -> bool:
    config = config or DEFAULT_QUALITY_FILTER_CONFIG
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return (
        h >= config.body_min_height
        and w * h >= config.body_min_area
        and crop["sharpness"] >= config.body_min_sharpness
    )


def _face_ok(crop: dict, config: QualityFilterConfig | None = None) -> bool:
    config = config or DEFAULT_QUALITY_FILTER_CONFIG
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return (
        w >= config.face_min_width
        and h >= config.face_min_height
        and crop["sharpness"] >= config.face_min_sharpness
    )


def _face_diagnostic(
    crop: dict,
    config: QualityFilterConfig | None = None,
) -> tuple[str | None, dict]:
    config = config or DEFAULT_QUALITY_FILTER_CONFIG
    raw_path = str(crop.get("path") or "")
    path = Path(raw_path)
    exists = bool(raw_path) and path.is_file()
    diagnostic = {
        "basename": path.name or "<missing-path>",
        "exists": exists,
        "file_size": path.stat().st_size if exists else 0,
        "decoded_shape": None,
        "width": None,
        "height": None,
        "sharpness": crop.get("sharpness"),
        "brightness": None,
    }
    if not exists:
        return "missing_file", diagnostic

    try:
        image = cv2.imread(str(path))
    except Exception:
        image = None
    if image is None:
        return "unreadable", diagnostic
    diagnostic["decoded_shape"] = list(image.shape)
    if image.size == 0 or image.ndim < 2:
        return "empty", diagnostic

    height, width = image.shape[:2]
    diagnostic["width"] = int(width)
    diagnostic["height"] = int(height)
    diagnostic["brightness"] = float(image.mean())
    try:
        x1, y1, x2, y2 = crop["bbox"]
        bbox_width = float(x2) - float(x1)
        bbox_height = float(y2) - float(y1)
        sharpness = float(crop["sharpness"])
    except (KeyError, TypeError, ValueError):
        return "other", diagnostic

    if bbox_width < config.face_min_width or bbox_height < config.face_min_height:
        return "too_small", diagnostic
    if sharpness < config.face_min_sharpness:
        return "low_sharpness", diagnostic
    # There is no brightness rejection threshold in the current algorithm.
    return None, diagnostic


def _print_face_rejection_sample(diagnostic: dict, reason: str) -> None:
    print(
        "[filter_quality] face rejected sample: "
        f"basename={diagnostic['basename']} exists={diagnostic['exists']} "
        f"file_size={diagnostic['file_size']} "
        f"decoded_shape={diagnostic['decoded_shape']} "
        f"width={diagnostic['width']} height={diagnostic['height']} "
        f"sharpness={diagnostic['sharpness']} "
        f"brightness={diagnostic['brightness']} reason={reason}"
    )


def filter_quality(state: dict) -> dict:
    config = load_quality_filter_config()
    quality_body = [c for c in state["body_crops"] if _body_ok(c, config)]
    quality_face = []
    rejected_counts = {reason: 0 for reason in _FACE_REJECTION_REASONS}
    rejected_samples = 0
    for crop in state["face_crops"]:
        reason, diagnostic = _face_diagnostic(crop, config)
        if reason is None:
            quality_face.append(crop)
            continue
        rejected_counts[reason] += 1
        if rejected_samples < 5:
            _print_face_rejection_sample(diagnostic, reason)
            rejected_samples += 1

    rejected_total = len(state["face_crops"]) - len(quality_face)
    assert sum(rejected_counts.values()) == rejected_total
    print(f"[filter_quality] body: {len(state['body_crops'])} → {len(quality_body)} | face: {len(state['face_crops'])} → {len(quality_face)}")
    print(
        "[filter_quality] face rejected: "
        + " ".join(
            f"{reason}={rejected_counts[reason]}"
            for reason in _FACE_REJECTION_REASONS
        )
    )
    return {
        "quality_body_crops": quality_body,
        "quality_face_crops": quality_face,
        "total_quality_body_crops": len(quality_body),
        "total_quality_face_crops": len(quality_face),
        "face_rejection_counts": rejected_counts,
    }
