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
    from forensics.person_creation import config

    valid_paths = [p for p in paths if p and Path(p).exists()]
    missing_paths = [p for p in paths if p and not Path(p).exists()]
    for path in missing_paths:
        print(f"[extract_reid][warn] missing crop skipped: {path}")

    if not valid_paths:
        return _error_block("No valid body crops for ReID")

    try:
        embeddings = get_reid_embedder().embed_batch(valid_paths, batch_size=config.REID_BATCH_SIZE)
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

    from forensics.person_creation.models.reid_embedder import get_reid_embedder, release_reid_embedder
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom
    from forensics.person_creation import config

    best_by_person = state.get("best_body_crops_by_person") or {}

    log_memory("before loading OSNet ReID")
    try:
        get_reid_embedder().load(device=config.REID_DEVICE)
    except Exception as exc:
        # Same graceful-degradation as before: missing torchreid / OOM here
        # just means every person gets a ReID error block below, not a
        # crashed node — the per-crop try/except in _extract_for_paths
        # would hit the same failure again and record it there.
        print(f"[extract_reid][warn] ReID unavailable: {exc}")
    log_memory("after loading OSNet ReID")

    try:
        if best_by_person:
            reid_by_person: dict[str, dict] = {}
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

        if state.get("person_tracks"):
            # Multi-person mode without per-person crops: never fall back to
            # the global crop list — it would mix people into one embedding.
            print(
                "[extract_reid][warn] multi-person mode but no per-person body "
                "crops; skipping global fallback to avoid mixing people"
            )
            return {
                "reid_by_person": {},
                "reid": _error_block("No per-person body crops in multi-person mode"),
            }

        reid = _extract_for_paths(list(state.get("best_body_crops") or []))
        if reid.get("embedding") is None:
            print(f"[extract_reid] {reid.get('error')}")
        else:
            print(f"[extract_reid] {len(reid['source_crops'])} crops, dim={reid['embedding_dim']}")

        return {"reid": reid, "reid_by_person": {}}
    except Exception as exc:
        raise clarify_oom(exc, "extract_reid") from exc
    finally:
        release_reid_embedder()
        cleanup_memory("extract_reid")
