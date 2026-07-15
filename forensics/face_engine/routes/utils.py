from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Request


class BadImage(ValueError):
    pass


def image_from_request(request: Request) -> np.ndarray:
    raw = None
    if "image" in request.files:
        raw = request.files["image"].read()
    elif request.is_json:
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("image_b64")
        if encoded:
            try:
                raw = base64.b64decode(encoded)
            except Exception as exc:
                raise BadImage(f"invalid image_b64: {exc}") from exc

    if not raw:
        raise BadImage("provide multipart field 'image' or JSON image_b64")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise BadImage("invalid or unreadable image")
    return image

