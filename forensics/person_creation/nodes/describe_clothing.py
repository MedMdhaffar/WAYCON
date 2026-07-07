import cv2
from pathlib import Path


_UNKNOWN = {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"}
_COLOR_WORDS = {
    "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
    "orange", "brown", "purple", "pink", "beige", "tan", "navy",
}


def _validate_clothing(structured: dict, cluster_id=None) -> dict:
    out = dict(_UNKNOWN)
    out.update(structured or {})
    shoes = str(out.get("shoes") or "").strip()
    if shoes.lower() in _COLOR_WORDS:
        out["shoes"] = f"{shoes} shoes"
        label = f" cluster={cluster_id}" if cluster_id is not None else ""
        print(f"[describe_clothing] warning:{label} shoes was color-only; normalized to '{out['shoes']}'")

    top = str(out.get("top") or "unknown").strip()
    bottom = str(out.get("bottom") or "unknown").strip()
    shoes = str(out.get("shoes") or "unknown").strip()
    full = str(out.get("full") or "").lower()
    bottom_l = bottom.lower()
    full_says_shorts = "shorts" in full
    full_says_pants = any(word in full for word in ("pants", "trousers", "jeans"))
    bottom_says_shorts = "shorts" in bottom_l
    bottom_says_pants = any(word in bottom_l for word in ("pants", "trousers", "jeans"))
    if (bottom_says_pants and full_says_shorts) or (bottom_says_shorts and full_says_pants):
        out["full"] = f"{top}, {bottom}, {shoes}"
        label = f" cluster={cluster_id}" if cluster_id is not None else ""
        print(f"[describe_clothing] warning:{label} full contradicted bottom; using fallback full='{out['full']}'")
    return out


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
            structured = _validate_clothing(structured, cluster_id=cid)
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
    structured = _validate_clothing(structured)
    if not raw:
        print("[describe_clothing] no crops available")
    else:
        print(f"[describe_clothing] {structured}")
    return {"clothing_raw": raw, "clothing_structured": structured}
