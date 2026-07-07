from pathlib import Path


_N_BEST = 5


def _select_best_from_associations(associations: list[dict]) -> list[str]:
    valid_assoc = [a for a in associations if a.get("body_path") and Path(a["body_path"]).exists()]
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
        if not seg:
            continue
        top = max(seg, key=lambda a: a.get("body_sharpness", 0.0))
        best.append(top["body_path"])
        used_paths.add(top["body_path"])

    if len(best) < _N_BEST:
        remaining = [a for a in sorted_assoc if a["body_path"] not in used_paths]
        remaining.sort(key=lambda a: a.get("body_sharpness", 0.0), reverse=True)
        for assoc in remaining:
            if len(best) >= _N_BEST:
                break
            best.append(assoc["body_path"])

    return best


def select_best_per_person(state: dict) -> dict:
    person_tracks = list(state.get("person_tracks") or [])
    best_by_person: dict[str, list[str]] = {}

    for track in person_tracks:
        person_id = track["person_id"]
        best_by_person[person_id] = _select_best_from_associations(track.get("associations") or [])
        if not best_by_person[person_id]:
            print(f"[select_best_per_person][warn] {person_id}: no valid body crops")

    legacy_best = next((paths for paths in best_by_person.values() if paths), [])
    print(
        f"[select_best_per_person] selected crops for {len(best_by_person)} people "
        f"({sum(len(v) for v in best_by_person.values())} total)"
    )
    return {
        "best_body_crops_by_person": best_by_person,
        "best_body_crops": legacy_best,
    }
