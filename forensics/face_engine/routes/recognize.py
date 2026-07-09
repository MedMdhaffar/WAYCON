from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from forensics.face_engine.routes.utils import BadImage, image_from_request
from forensics.global_memory import GlobalMemory
from forensics.global_memory.config import SIMILARITY_THRESHOLD

bp = Blueprint("face_engine_recognize", __name__)


def _embedding_from_payload() -> tuple[list[float] | None, tuple[dict, int] | None]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if "embedding" in payload:
            vec = np.asarray(payload.get("embedding"), dtype=np.float32).reshape(-1)
            if vec.shape[0] != 512:
                return None, ({"error": f"embedding dimension is {vec.shape[0]}, expected 512"}, 422)
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                return None, ({"error": "embedding must be non-zero"}, 422)
            vec = vec / norm
            return vec.astype(float).tolist(), None

    try:
        image = image_from_request(request)
    except BadImage as exc:
        return None, ({"error": str(exc)}, 400)

    try:
        embedding = current_app.config["FACE_EMBEDDER"].embed(image)
    except Exception as exc:
        return None, ({"error": f"face embedding failed: {exc}"}, 500)
    return embedding, None


@bp.post("/recognize")
def recognize():
    payload = request.get_json(silent=True) if request.is_json else {}
    embedding, error = _embedding_from_payload()
    if error is not None:
        body, status = error
        return jsonify(body), status

    top_k = int((payload or {}).get("top_k", 5))
    threshold = float((payload or {}).get("threshold", SIMILARITY_THRESHOLD))

    gm = GlobalMemory()
    try:
        matches = gm.query_by_face(embedding, top_k=top_k, threshold=threshold)
    finally:
        gm.close()

    compact = [
        {
            "person_id": item["person_id"],
            "name": item["name"],
            "similarity": item["similarity"],
        }
        for item in matches
    ]
    return jsonify({"matches": compact, "recognized": bool(compact)})

