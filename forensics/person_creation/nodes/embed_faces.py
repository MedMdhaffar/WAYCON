import cv2
import numpy as np
from pathlib import Path


def embed_faces(state: dict) -> dict:
    """Compute the mean face embedding from confirmed face crops.

    Source of truth is `associations[*].face_path` — only faces that the human
    paired with a body in HITL. Falls back to `quality_face_crops` if no
    associations exist (matches select_best.py's fallback contract).
    """
    from forensics.face_engine.client import FaceEngineClient

    embedder = FaceEngineClient()

    associations = state.get("associations") or []
    if associations:
        face_paths = list(dict.fromkeys(a["face_path"] for a in associations if a.get("face_path")))
        source = "associations"
    else:
        face_paths = [c["path"] for c in state.get("quality_face_crops", []) if c.get("path")]
        source = "quality_face_crops (no associations — fallback)"

    if not face_paths:
        print("[embed_faces] no face crops to embed")
        return {"face_embeddings": [], "mean_face_embedding": []}

    embeddings = []
    for p in face_paths:
        img = cv2.imread(str(Path(p).resolve()))
        if img is None:
            continue
        emb = embedder.embed(img).tolist()
        embeddings.append(emb)

    if not embeddings:
        print(f"[embed_faces] no readable face crops from {source}")
        return {"face_embeddings": [], "mean_face_embedding": []}

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    print(f"[embed_faces] embedded {len(embeddings)} face crops from {source}, mean embedding norm={float(np.linalg.norm(mean_emb)):.4f}")
    return {
        "face_embeddings": embeddings,
        "mean_face_embedding": mean_emb.tolist(),
    }
