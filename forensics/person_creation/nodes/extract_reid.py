from pathlib import Path

import numpy as np

from forensics.person_creation.models.reid_embedder import REID_MODEL_NAME


_SIGNAL_TYPE = "same_day_supporting_appearance"


def _error_block(message: str, source_crops: list[str] | None = None) -> dict:
    return {
        "model": REID_MODEL_NAME,
        "embedding_dim": None,
        "embedding": None,
        "source_crops": source_crops or [],
        "per_crop": [],
        "signal_type": _SIGNAL_TYPE,
        "error": message,
    }


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _extract_for_paths(paths: list[str]) -> dict:
    from forensics.person_creation.models.reid_embedder import get_reid_embedder

    valid_paths = [p for p in paths if p and Path(p).exists()]
    missing_paths = [p for p in paths if p and not Path(p).exists()]
    for path in missing_paths:
        print(f"[extract_reid][warn] missing crop skipped: {path}")

    if not valid_paths:
        return _error_block("No valid body crops for ReID")

    try:
        embeddings = get_reid_embedder().embed_batch(valid_paths)
    except Exception as exc:
        return _error_block(str(exc), valid_paths)

    if not embeddings:
        return _error_block("No ReID embeddings produced", valid_paths)

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    mean_embedding = _normalize(np.mean(embedding_array, axis=0))
    if mean_embedding is None:
        return _error_block("ReID mean embedding has zero norm", valid_paths)

    return {
        "model": REID_MODEL_NAME,
        "embedding_dim": int(mean_embedding.shape[0]),
        "embedding": mean_embedding.astype(float).tolist(),
        "source_crops": valid_paths,
        "per_crop": [
            {"crop": crop_path, "embedding": embedding}
            for crop_path, embedding in zip(valid_paths, embeddings)
        ],
        "signal_type": _SIGNAL_TYPE,
    }


def extract_reid(state: dict) -> dict:
    """Compute same-day body appearance embeddings from selected body crops."""

    best_by_person = state.get("best_body_crops_by_person") or {}
    reid_by_person: dict[str, dict] = {}

    if best_by_person:
        for person_id, paths in best_by_person.items():
            reid_by_person[person_id] = _extract_for_paths(list(paths or []))
            block = reid_by_person[person_id]
            if block.get("embedding") is None:
                print(f"[extract_reid] {person_id}: {block.get('error')}")
            else:
                print(
                    f"[extract_reid] {person_id}: {len(block['source_crops'])} crops, "
                    f"dim={block['embedding_dim']}"
                )

        first_person = next(iter(reid_by_person), None)
        return {
            "reid_by_person": reid_by_person,
            "reid": reid_by_person.get(first_person, _error_block("No valid body crops for ReID")),
        }

    reid = _extract_for_paths(list(state.get("best_body_crops") or []))
    if reid.get("embedding") is None:
        print(f"[extract_reid] {reid.get('error')}")
    else:
        print(f"[extract_reid] {len(reid['source_crops'])} crops, dim={reid['embedding_dim']}")

    return {"reid": reid, "reid_by_person": {}}
