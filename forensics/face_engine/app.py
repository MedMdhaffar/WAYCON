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
    app.run(host="0.0.0.0", port=config.PORT, debug=False)


if __name__ == "__main__":
    main()
