from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    av = av / max(float(np.linalg.norm(av)), 1e-12)
    bv = bv / max(float(np.linalg.norm(bv)), 1e-12)
    return float(np.dot(av, bv))


def _local_embed(path: str) -> list[float]:
    import cv2
    from forensics.person_creation.models.face_embedder import get_face_embedder

    img = cv2.imread(str(Path(path).resolve()))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return get_face_embedder().embed(img)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local FaceEmbedder and face_engine embeddings for face crops.")
    parser.add_argument("crops", nargs="+", help="Face crop image paths.")
    args = parser.parse_args()

    from forensics.face_engine.client import FaceEngineClient
    from forensics.person_creation.models.face_embedder import get_face_embedder

    get_face_embedder().load(device="auto")
    client = FaceEngineClient()

    ok = 0
    for raw in args.crops:
        try:
            local = _local_embed(raw)
            service = client.embed(raw).get("embedding") or []
            sim = _cosine(local, service)
            print(f"{raw}: cosine={sim:.8f}")
            ok += 1
        except Exception as exc:
            print(f"{raw}: error: {exc}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
