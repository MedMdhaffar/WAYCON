import os

import cv2
from pathlib import Path


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _local_embed(path: str) -> list[float]:
    from forensics.person_creation.models.face_embedder import get_face_embedder

    img = cv2.imread(str(Path(path).resolve()))
    if img is None:
        raise ValueError("unreadable_face_crop")
    return get_face_embedder().embed(img)


def _face_engine_embed(path: str) -> list[float]:
    from forensics.face_engine.client import FaceEngineClient

    result = FaceEngineClient().embed(str(path))
    embedding = result.get("embedding")
    if not embedding:
        raise ValueError("face_engine returned no embedding")
    return [float(v) for v in embedding]


def embed_all_faces(state: dict) -> dict:
    """Embed every quality face crop before any identity decision is made."""
    use_face_engine = _enabled("PERSON_CREATION_USE_FACE_ENGINE")
    fallback_local = _enabled("FACE_ENGINE_FALLBACK_LOCAL")
    mode = "face_engine" if use_face_engine else "local"
    records: list[dict] = []
    failed: list[dict] = []

    if use_face_engine:
        print("[embed_all_faces] using face_engine for video face crop embeddings")
    else:
        print("[embed_all_faces] using local FaceEmbedder for video face crop embeddings")

    for crop in state.get("quality_face_crops", []):
        path = crop.get("path")
        if not path:
            continue
        try:
            if use_face_engine:
                try:
                    emb = _face_engine_embed(path)
                except Exception as exc:
                    if not fallback_local:
                        raise RuntimeError(f"face_engine embedding failed and local fallback is disabled: {exc}") from exc
                    print(f"[embed_all_faces] warning: face_engine failed for {path}; falling back to local embedder: {exc}")
                    emb = _local_embed(path)
            else:
                emb = _local_embed(path)
        except Exception as exc:
            if use_face_engine and not fallback_local:
                raise
            print(f"[embed_all_faces] warning: failed to embed {path}: {exc}")
            reason = "unreadable_face_crop" if str(exc) == "unreadable_face_crop" else f"embedding_failed: {exc}"
            failed.append({**crop, "crop_path": path, "reason": reason})
            continue

        records.append({
            "crop_path": path,
            "embedding": emb,
            "frame_idx": crop.get("frame_idx"),
            "video": crop.get("video"),
            "bbox": crop.get("bbox"),
            "sharpness": crop.get("sharpness", 0.0),
        })

    print(f"[embed_all_faces] embedded {len(records)} / {len(state.get('quality_face_crops', []))} quality faces ({mode})")
    return {"all_face_embeddings": records, "failed_face_embeddings": failed}
