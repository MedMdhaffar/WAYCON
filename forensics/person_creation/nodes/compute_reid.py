from __future__ import annotations


def _paths_for_cluster(per_cluster: dict, cid: int) -> list[str]:
    paths = per_cluster.get(cid, per_cluster.get(str(cid), []))
    return [str(p) for p in paths[:5]]


def compute_reid(state: dict) -> dict:
    """Compute one body ReID embedding per identity cluster from best crops."""
    clusters = state.get("identity_clusters") or []
    per_cluster_best = state.get("per_cluster_best_body_crops") or {}
    reid_model = None
    if state.get("reid_available"):
        from forensics.person_creation.models.reid_extractor import get_reid_extractor

        reid_model = get_reid_extractor(config=state.get("reid_config"), device="auto")

    embeddings: dict[int, list[float] | None] = {}
    crop_counts: dict[int, int] = {}
    reasons: dict[int, str] = {}

    for cluster in clusters:
        cid = int(cluster["cluster_id"])
        crop_paths = _paths_for_cluster(per_cluster_best, cid)

        if reid_model is None or not reid_model.is_available():
            embeddings[cid] = None
            crop_counts[cid] = 0
            unavailable = state.get("reid_unavailable_reason")
            reasons[cid] = f"reid_model_unavailable: {unavailable}" if unavailable else "no_reid_model_configured"
            continue

        vec = reid_model.compute_mean_embedding(crop_paths)
        count = int(getattr(reid_model, "last_success_count", 0))
        crop_counts[cid] = count

        if vec is None:
            embeddings[cid] = None
            reasons[cid] = "no_reid_embedding_produced"
            continue

        embeddings[cid] = [float(x) for x in vec.tolist()]
        reasons[cid] = "computed"

    computed = sum(1 for value in embeddings.values() if value is not None)
    print(f"[compute_reid] computed profile ReID for {computed}/{len(clusters)} cluster(s)")
    return {
        "reid_embeddings": embeddings,
        "reid_crop_counts": crop_counts,
        "reid_reasons": reasons,
    }
