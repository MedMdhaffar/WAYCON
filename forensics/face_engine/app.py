"""Standalone face_engine HTTP service.

No longer on person_creation's critical path -- the pipeline (load_models.py,
process_video.py, process_live_stream.py, embed_all_faces.py, auto_pair.py,
tools/add_face_photos.py) calls forensics.face_engine.local_client.LocalFaceEngine
in-process instead, to avoid a per-crop HTTP round-trip / JPEG re-encode and to keep
GPU-resident models in the same process as the rest of the realtime pipeline.

This app (and its /recognize route, which talks to GlobalMemory directly) is kept
for standalone/external use only -- e.g. a non-Python consumer, or querying face
recognition remotely without pulling in the full person_creation pipeline. It is not
started or required by the realtime pipeline or by forensics/person_creation/service.py.
"""

from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS

from forensics.face_engine import config
from forensics.face_engine.models.detector import get_face_detector
from forensics.face_engine.models.embedder import get_face_embedder
from forensics.face_engine.routes.detect import bp as detect_bp
from forensics.face_engine.routes.embed import bp as embed_bp
from forensics.face_engine.routes.health import bp as health_bp
from forensics.face_engine.routes.recognize import bp as recognize_bp


def create_app(load_models: bool = True) -> Flask:
    app = Flask(__name__)
    CORS(app)

    detector = get_face_detector()
    embedder = get_face_embedder()
    if load_models:
        detector.load(device=config.DEVICE)
        embedder.load(device=config.DEVICE)

    app.config["FACE_DETECTOR"] = detector
    app.config["FACE_EMBEDDER"] = embedder

    app.register_blueprint(health_bp)
    app.register_blueprint(detect_bp)
    app.register_blueprint(embed_bp)
    app.register_blueprint(recognize_bp)

    @app.errorhandler(Exception)
    def _json_error(exc):
        return jsonify({"error": str(exc)}), 500

    return app


app = create_app(load_models=False)


def main() -> None:
    global app
    app = create_app(load_models=True)
    app.run(host=config.HOST, port=config.PORT, debug=False)


if __name__ == "__main__":
    main()
