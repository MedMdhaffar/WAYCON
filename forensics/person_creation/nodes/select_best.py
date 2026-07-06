from pathlib import Path

_N_BEST = 5


def _select_from_associations(associations: list[dict]) -> list[str]:
    valid_assoc = [a for a in associations if Path(a.get("body_path", "")).exists()]
    if not valid_assoc:
        return []

    sorted_assoc = sorted(valid_assoc, key=lambda a: a["frame_idx"])
    frame_min = sorted_assoc[0]["frame_idx"]
    frame_max = sorted_assoc[-1]["frame_idx"]
    span = max(frame_max - frame_min, 1)

    segments: list[list[dict]] = [[] for _ in range(_N_BEST)]
    for assoc in sorted_assoc:
        seg_idx = min(int((assoc["frame_idx"] - frame_min) / span * _N_BEST), _N_BEST - 1)
        segments[seg_idx].append(assoc)

    best: list[str] = []
    used_paths: set[str] = set()

    for seg in segments:
        if seg:
            top = max(seg, key=lambda a: a["body_sharpness"])
            best.append(top["body_path"])
            used_paths.add(top["body_path"])

    if len(best) < _N_BEST:
        remaining = [a for a in sorted_assoc if a["body_path"] not in used_paths]
        remaining.sort(key=lambda a: a["body_sharpness"], reverse=True)
        for assoc in remaining:
            if len(best) >= _N_BEST:
                break
            best.append(assoc["body_path"])
    return best


def select_best(state: dict) -> dict:
    """Pick the top-5 sharpest, temporally spread body crops per identity.

    Input state:  `associations`, `identity_clusters` (falls back to
                  `quality_body_crops` when there are no associations).
    Output state: `per_cluster_best_body_crops` and single-cluster `best_body_crops`.
    """
    associations = state.get("associations") or []
    clusters = state.get("identity_clusters") or []

    if clusters:
        per_cluster: dict[int, list[str]] = {}
        for cluster in clusters:
            cid = int(cluster["cluster_id"])
            cluster_assoc = [a for a in associations if int(a.get("cluster_id", -1)) == cid]
            per_cluster[cid] = _select_from_associations(cluster_assoc)
        first = sorted(per_cluster)[0] if per_cluster else None
        print(f"[select_best] selected best body crops for {len(per_cluster)} cluster(s)")
        return {
            "per_cluster_best_body_crops": per_cluster,
            "best_body_crops": per_cluster.get(first, []) if first is not None else [],
        }

    if not associations:
        print("[select_best] no associations - falling back to sharpest quality body crops")
        fallback = [c for c in state.get("quality_body_crops", []) if Path(c["path"]).exists()]
        fallback = sorted(fallback, key=lambda c: c["sharpness"], reverse=True)
        return {"best_body_crops": [c["path"] for c in fallback[:_N_BEST]]}

    best = _select_from_associations(associations)
    if not best:
        print("[select_best] all association body crops missing - empty best_body_crops")
        return {"best_body_crops": []}

    print(f"[select_best] selected {len(best)} body crops for VLM")
    return {"best_body_crops": best}
