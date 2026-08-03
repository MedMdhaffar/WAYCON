from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from forensics.face_engine.routes.utils import BadImage, image_from_request

bp = Blueprint("face_engine_detect", __name__)


@bp.post("/detect")
def detect():
    try:
        image = image_from_request(request)
    except BadImage as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        faces = current_app.config["FACE_DETECTOR"].detect(image)
    except Exception as exc:
        return jsonify({"error": f"face detection failed: {exc}"}), 500

    return jsonify({"faces": faces, "count": len(faces)})

