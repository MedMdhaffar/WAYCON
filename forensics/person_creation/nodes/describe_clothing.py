import cv2
from pathlib import Path


_UNKNOWN = {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"}


def _read_crops(paths: list[str]) -> list:
    crops = []
    for p in paths:
        try:
            img = cv2.imread(str(Path(p).resolve()))
            if img is not None:
                crops.append(img)
        except Exception:
            continue
    return crops


def _describe_paths(describer, paths: list[str]) -> tuple[str, dict]:
    crops = _read_crops(paths)
    if not crops:
        return "", dict(_UNKNOWN)
    return describer.describe(crops)


def describe_clothing(state: dict) -> dict:
    """Run the clothing VLM once per identity over its best body crops.

    Input state:  `per_cluster_best_body_crops` (falls back to `best_body_crops`).
    Output state: `per_cluster_clothing` and single-cluster `clothing_raw` /
                  `clothing_structured`.
    """
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    describer = get_clothing_describer()
    per_cluster_best = state.get("per_cluster_best_body_crops") or {}

    if per_cluster_best:
        per_cluster_clothing: dict[int, dict] = {}
        first_raw = ""
        first_structured = dict(_UNKNOWN)
        first = True
        for raw_cid, paths in per_cluster_best.items():
            cid = int(raw_cid)
            raw, structured = _describe_paths(describer, paths)
            per_cluster_clothing[cid] = {"raw": raw, "structured": structured}
            if first:
                first_raw = raw
                first_structured = structured
                first = False
        print(f"[describe_clothing] described clothing for {len(per_cluster_clothing)} cluster(s)")
        return {
            "per_cluster_clothing": per_cluster_clothing,
            "clothing_raw": first_raw,
            "clothing_structured": first_structured,
        }

    raw, structured = _describe_paths(describer, state.get("best_body_crops", []))
    if not raw:
        print("[describe_clothing] no crops available")
    else:
        print(f"[describe_clothing] {structured}")
    return {"clothing_raw": raw, "clothing_structured": structured}
