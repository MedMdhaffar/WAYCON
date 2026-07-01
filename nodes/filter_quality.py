_MIN_BODY_H = 80
_MIN_BODY_AREA = 3000
_MIN_FACE_W = 60
_MIN_FACE_H = 60
_MIN_SHARPNESS = 50.0


def _body_ok(crop: dict) -> bool:
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return h >= _MIN_BODY_H and w * h >= _MIN_BODY_AREA and crop["sharpness"] >= _MIN_SHARPNESS


def _face_ok(crop: dict) -> bool:
    x1, y1, x2, y2 = crop["bbox"]
    w, h = x2 - x1, y2 - y1
    return w >= _MIN_FACE_W and h >= _MIN_FACE_H and crop["sharpness"] >= _MIN_SHARPNESS


def filter_quality(state: dict) -> dict:
    quality_body = [c for c in state["body_crops"] if _body_ok(c)]
    quality_face = [c for c in state["face_crops"] if _face_ok(c)]
    print(f"[filter_quality] body: {len(state['body_crops'])} → {len(quality_body)} | face: {len(state['face_crops'])} → {len(quality_face)}")
    return {"quality_body_crops": quality_body, "quality_face_crops": quality_face}
