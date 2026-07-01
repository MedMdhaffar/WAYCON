from pathlib import Path

_N_BEST = 5


def select_best(state: dict) -> dict:
    associations = state["associations"]

    if not associations:
        print("[select_best] no associations — falling back to sharpest quality body crops")
        fallback = [c for c in state["quality_body_crops"] if Path(c["path"]).exists()]
        fallback = sorted(fallback, key=lambda c: c["sharpness"], reverse=True)
        return {"best_body_crops": [c["path"] for c in fallback[:_N_BEST]]}

    # Filter out any association whose body file was deleted
    valid_assoc = [a for a in associations if Path(a["body_path"]).exists()]
    if not valid_assoc:
        print("[select_best] all association body crops missing — empty best_body_crops")
        return {"best_body_crops": []}

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

    print(f"[select_best] selected {len(best)} body crops for VLM")
    return {"best_body_crops": best}
