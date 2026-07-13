from __future__ import annotations

from flask import Blueprint, current_app, jsonify

bp = Blueprint("face_engine_health", __name__)


@bp.get("/health")
def health():
    detector = current_app.config["FACE_DETECTOR"]
    embedder = current_app.config["FACE_EMBEDDER"]
    models_loaded = detector.is_loaded() and embedder.is_loaded()
    return jsonify({
        "service": "waycon-face-engine",
        "api_version": 1,
        "status": "ok" if models_loaded else "loading",
        "device": embedder.device,
        "models_loaded": models_loaded,
    })
