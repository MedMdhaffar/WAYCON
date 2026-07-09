from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from forensics.face_engine.routes.utils import BadImage, image_from_request

bp = Blueprint("face_engine_embed", __name__)


@bp.post("/embed")
def embed():
    try:
        image = image_from_request(request)
    except BadImage as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        embedding = current_app.config["FACE_EMBEDDER"].embed(image)
    except Exception as exc:
        return jsonify({"error": f"face embedding failed: {exc}"}), 500

    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.shape[0] != 512:
        return jsonify({"error": f"embedding dimension is {vec.shape[0]}, expected 512"}), 500
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) >= 1e-5:
        return jsonify({"error": f"embedding norm is {norm}, expected 1.0"}), 500
    return jsonify({"embedding": vec.astype(float).tolist(), "dim": 512, "norm": norm})

