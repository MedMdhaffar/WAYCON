from __future__ import annotations

from pathlib import Path

import cv2

from forensics.person_creation.global_memory.similarity import mean_normalized_embeddings


class FacePhotoRegistrationError(Exception):
    """Raised when phone photos cannot produce a usable face embedding."""


def _crop(frame, bbox: list[float], padding: int = 2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def _best_face(detections: list[dict]) -> dict | None:
    if not detections:
        return None

    def key(det: dict) -> tuple[float, float]:
        x1, y1, x2, y2 = det["bbox"]
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        return area, float(det.get("score", 0.0))

    return max(detections, key=key)


def _load_models():
    from forensics.person_creation.models.face_detector import get_face_detector
    from forensics.person_creation.models.face_embedder import get_face_embedder
    from forensics.person_creation.models.device import resolve_device

    detector = get_face_detector()
    embedder = get_face_embedder()
    device = resolve_device("auto")
    if getattr(detector, "_model", None) is None:
        detector.load(device=device)
    if not embedder.is_loaded():
        embedder.load(device=device)
    return detector, embedder


def embed_face_image(image_path: str) -> list[float]:
    result = embed_face_photo_records([image_path])
    if result["skipped"]:
        reason = result["skipped"][0].get("reason", "no usable face")
        raise FacePhotoRegistrationError(f"{image_path}: {reason}")
    return result["embeddings"][0]


def embed_face_photos(image_paths: list[str]) -> list[float]:
    result = embed_face_photo_records(image_paths)
    if not result["embeddings"]:
        reasons = "; ".join(f"{s['image_path']}: {s['reason']}" for s in result["skipped"])
        raise FacePhotoRegistrationError(f"no valid faces found in phone photos ({reasons})")
    return mean_normalized_embeddings(result["embeddings"])


def embed_face_photo_records(image_paths: list[str], save_crops_dir: str | Path | None = None) -> dict:
    if not image_paths:
        raise FacePhotoRegistrationError("image_paths is empty")

    detector, embedder = _load_models()
    save_dir = Path(save_crops_dir) if save_crops_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[list[float]] = []
    sources: list[dict] = []
    skipped: list[dict] = []

    for idx, raw in enumerate(image_paths):
        path = Path(raw)
        img = cv2.imread(str(path))
        if img is None:
            skipped.append({"image_path": str(path), "reason": "unreadable image"})
            continue

        face = _best_face(detector.detect(img))
        if face is None:
            skipped.append({"image_path": str(path), "reason": "no face detected"})
            continue

        crop = _crop(img, face["bbox"])
        if crop is None or crop.size == 0:
            skipped.append({"image_path": str(path), "reason": "empty face crop"})
            continue

        try:
            embedding = embedder.embed(crop)
        except Exception as exc:
            skipped.append({"image_path": str(path), "reason": f"embedding failed: {exc}"})
            continue

        crop_path = None
        if save_dir:
            out_name = f"phone_{idx:03d}_{path.stem}.jpg"
            out_path = save_dir / out_name
            if cv2.imwrite(str(out_path), crop):
                crop_path = str(out_path)

        embeddings.append(embedding)
        sources.append({
            "image_path": str(path),
            "face_crop_path": crop_path,
            "score": float(face.get("score", 0.0)),
        })

    return {"embeddings": embeddings, "sources": sources, "skipped": skipped}
