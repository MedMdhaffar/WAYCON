from pathlib import Path

import cv2


_MIN_BODY_H = 80
_MIN_BODY_AREA = 3000
_MIN_FACE_W = 60
_MIN_FACE_H = 60
_MIN_SHARPNESS = 50.0
_FACE_REJECTION_REASONS = (
    "missing_file",
    "unreadable",
    "empty",
    "too_small",
    "low_sharpness",
    "bad_brightness",
    "other",
)


def _body_ok(crop: dict) -> bool:
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return h >= _MIN_BODY_H and w * h >= _MIN_BODY_AREA and crop["sharpness"] >= _MIN_SHARPNESS


def _face_ok(crop: dict) -> bool:
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return w >= _MIN_FACE_W and h >= _MIN_FACE_H and crop["sharpness"] >= _MIN_SHARPNESS


def _face_diagnostic(crop: dict) -> tuple[str | None, dict]:
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

    if bbox_width < _MIN_FACE_W or bbox_height < _MIN_FACE_H:
        return "too_small", diagnostic
    if sharpness < _MIN_SHARPNESS:
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
    quality_body = [c for c in state["body_crops"] if _body_ok(c)]
    quality_face = []
    rejected_counts = {reason: 0 for reason in _FACE_REJECTION_REASONS}
    rejected_samples = 0
    for crop in state["face_crops"]:
        reason, diagnostic = _face_diagnostic(crop)
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
