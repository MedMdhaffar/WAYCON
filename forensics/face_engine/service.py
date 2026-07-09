from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from forensics.face_engine import DEFAULT_HOST, DEFAULT_PORT, SERVICE_NAME
from forensics.face_engine.path_utils import resolve_file_path, resolve_output_dir
from forensics.face_engine.schemas import (
    error_response,
    parse_top_k,
    require_embedding,
    require_json,
    require_path_field,
)
from forensics.person_creation.global_memory.config import (
    FACE_AUTO_MATCH_THRESHOLD,
    FACE_NO_MATCH_THRESHOLD,
)
from forensics.person_creation.models.device import device_info, resolve_device

app = Flask(__name__)

_MODEL_LOCK = threading.Lock()
_SELECTED_DEVICE: str | None = None


def _detector_loaded() -> bool:
    from forensics.person_creation.models.face_detector import get_face_detector

    return getattr(get_face_detector(), "_model", None) is not None


def _embedder_loaded() -> bool:
    from forensics.person_creation.models.face_embedder import get_face_embedder

    return bool(get_face_embedder().is_loaded())


def _models_loaded() -> dict[str, bool]:
    return {"detector": _detector_loaded(), "embedder": _embedder_loaded()}


def _load_models() -> None:
    global _SELECTED_DEVICE

    with _MODEL_LOCK:
        device = _SELECTED_DEVICE or resolve_device("auto")
        _SELECTED_DEVICE = device

        from forensics.person_creation.models.face_detector import get_face_detector
        from forensics.person_creation.models.face_embedder import get_face_embedder

        detector = get_face_detector()
        embedder = get_face_embedder()
        if getattr(detector, "_model", None) is None:
            detector.load(device=device)
        if not embedder.is_loaded():
            embedder.load(device=device)


def _read_image(path: Path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path.as_posix()}")
    return image


def _read_image_bytes(image_bytes: bytes):
    import cv2

    if not image_bytes:
        raise ValueError("missing image bytes")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image


def _crop(frame: np.ndarray, bbox: list[float], padding: int = 2) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def _face_payload(det: dict) -> dict:
    return {
        "bbox": [float(v) for v in det.get("bbox", [])],
        "confidence": float(det.get("score", det.get("confidence", 0.0))),
    }


def _embedding_payload(embedding: list[float]) -> dict:
    return {
        "embedding": [float(v) for v in embedding],
        "embedding_dim": len(embedding),
        "normalized": True,
    }


def _decision(similarity: float) -> str:
    if similarity >= FACE_AUTO_MATCH_THRESHOLD:
        return "auto_match"
    if similarity >= FACE_NO_MATCH_THRESHOLD:
        return "review"
    return "no_match"


def _detect(image: np.ndarray) -> list[dict]:
    _load_models()
    from forensics.person_creation.models.face_detector import get_face_detector

    return get_face_detector().detect(image)


def _embed(image: np.ndarray) -> list[float]:
    _load_models()
    from forensics.person_creation.models.face_embedder import get_face_embedder

    return get_face_embedder().embed(image)


@app.get("/health")
def health():
    info = device_info("auto")
    return jsonify({
        "ok": True,
        "service": SERVICE_NAME,
        "torch_version": info.get("torch_version"),
        "torch_cuda_version": info.get("torch_cuda_version"),
        "cuda_available": bool(info.get("cuda_available", False)),
        "selected_device": _SELECTED_DEVICE or info.get("selected_device"),
        "gpu_name": info.get("gpu_name"),
        "models_loaded": _models_loaded(),
    })


@app.post("/detect")
def detect():
    try:
        body = require_json()
        image_path = resolve_file_path(require_path_field(body, "image_path"))
        image = _read_image(image_path)
        faces = [_face_payload(det) for det in _detect(image)]
        return jsonify({"ok": True, "faces": faces})
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


@app.post("/detect-bytes")
def detect_bytes():
    try:
        uploaded = request.files.get("image")
        if uploaded is None:
            raise ValueError("missing image field")
        image = _read_image_bytes(uploaded.read())
        faces = [_face_payload(det) for det in _detect(image)]
        return jsonify({"ok": True, "faces": faces})
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


@app.post("/embed")
def embed():
    try:
        body = require_json()
        crop_path = resolve_file_path(require_path_field(body, "face_crop_path"))
        image = _read_image(crop_path)
        embedding = _embed(image)
        return jsonify({"ok": True, **_embedding_payload(embedding)})
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


@app.post("/detect-and-embed")
def detect_and_embed():
    try:
        body = require_json()
        image_path = resolve_file_path(require_path_field(body, "image_path"))
        save_dir = None
        if body.get("save_crops_dir"):
            save_dir = resolve_output_dir(str(body["save_crops_dir"]))
            save_dir.mkdir(parents=True, exist_ok=True)

        image = _read_image(image_path)
        faces = []
        detections = _detect(image)
        for idx, det in enumerate(detections):
            face = _face_payload(det)
            crop = _crop(image, face["bbox"])
            if crop.size == 0:
                continue

            crop_path = None
            if save_dir is not None:
                import cv2

                out_path = save_dir / f"{image_path.stem}_face_{idx:03d}.jpg"
                if cv2.imwrite(str(out_path), crop):
                    crop_path = out_path.as_posix()

            embedding = _embed(crop)
            faces.append({
                **face,
                "crop_path": crop_path,
                **_embedding_payload(embedding),
            })
        return jsonify({"ok": True, "faces": faces})
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


@app.post("/recognize")
def recognize():
    try:
        body = require_json()
        top_k = parse_top_k(body)
        has_path = bool(body.get("face_crop_path"))
        has_embedding = body.get("embedding") is not None
        if has_path == has_embedding:
            raise ValueError("provide exactly one of face_crop_path or embedding")

        if has_path:
            crop_path = resolve_file_path(require_path_field(body, "face_crop_path"))
            embedding = _embed(_read_image(crop_path))
        else:
            embedding = require_embedding(body.get("embedding"))

        from forensics.person_creation.global_memory import GlobalMemoryStore

        with GlobalMemoryStore() as store:
            matches = store.search_by_face(embedding, top_k=top_k, threshold=None)

        enriched = []
        for match in matches:
            similarity = float(match.get("similarity", 0.0))
            enriched.append({
                "person_id": match.get("person_id"),
                "name": match.get("name"),
                "identity_source": match.get("identity_source"),
                "similarity": similarity,
                "decision": _decision(similarity),
            })

        return jsonify({
            "ok": True,
            "matches": enriched,
            "review_threshold": FACE_NO_MATCH_THRESHOLD,
            "auto_match_threshold": FACE_AUTO_MATCH_THRESHOLD,
        })
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        return error_response(str(exc), 500)


def main() -> None:
    host = os.getenv("FACE_ENGINE_HOST", DEFAULT_HOST)
    port = int(os.getenv("FACE_ENGINE_PORT", str(DEFAULT_PORT)))
    try:
        _load_models()
    except Exception as exc:
        print(f"[face_engine] warning: startup model load failed: {exc}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
