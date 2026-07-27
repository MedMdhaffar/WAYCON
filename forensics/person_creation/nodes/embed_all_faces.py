import cv2
from pathlib import Path
import numpy as np


def embed_all_faces(state: dict) -> dict:
    """Embed every quality face crop before any identity decision is made."""
    from forensics.face_engine.client import FaceEngineClient

    embedder = FaceEngineClient()
    records: list[dict] = []
    failed: list[dict] = []

    for crop in state.get("quality_face_crops", []):
        path = crop.get("path")
        if not path:
            continue
        try:
            img = cv2.imread(str(Path(path).resolve()))
            if img is None:
                print(f"[embed_all_faces] warning: unreadable face crop skipped: {path}")
                failed.append({**crop, "crop_path": path, "reason": "unreadable_face_crop"})
                continue
            vector = np.asarray(embedder.embed(img), dtype=np.float64)
            if (
                vector.ndim != 1
                or vector.size == 0
                or not np.isfinite(vector).all()
            ):
                raise ValueError("invalid face embedding")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("invalid face embedding normalization")
            emb = (vector / norm).astype(float).tolist()
        except Exception as exc:
            print(f"[embed_all_faces] warning: failed to embed {path}: {exc}")
            failed.append({**crop, "crop_path": path, "reason": f"embedding_failed: {exc}"})
            continue

        records.append({
            "crop_path": path,
            "embedding": emb,
            "frame_idx": crop.get("frame_idx"),
            "video": crop.get("video"),
            "bbox": crop.get("bbox"),
            "sharpness": crop.get("sharpness", 0.0),
            "face_quality": crop.get("_face_quality", {}),
        })

    print(f"[embed_all_faces] embedded {len(records)} / {len(state.get('quality_face_crops', []))} quality faces")
    return {"all_face_embeddings": records, "failed_face_embeddings": failed}
