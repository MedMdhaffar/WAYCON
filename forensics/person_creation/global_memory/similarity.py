from __future__ import annotations

import numpy as np


def _as_vector(embedding: list[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D embedding")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def normalize_embedding(embedding: list[float] | np.ndarray) -> list[float]:
    arr = _as_vector(embedding, "embedding")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return (arr / norm).astype(float).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    av = np.asarray(normalize_embedding(a), dtype=np.float64)
    bv = np.asarray(normalize_embedding(b), dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError(f"embedding dimensions differ: {av.size} != {bv.size}")
    return float(np.dot(av, bv))


def mean_normalized_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("embeddings list is empty")
    vectors = [np.asarray(normalize_embedding(e), dtype=np.float64) for e in embeddings]
    first_shape = vectors[0].shape
    if any(v.shape != first_shape for v in vectors):
        raise ValueError("all embeddings must have the same dimension")
    return normalize_embedding(np.mean(np.stack(vectors, axis=0), axis=0))
