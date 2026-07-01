import cv2
from pathlib import Path


def describe_clothing(state: dict) -> dict:
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    describer = get_clothing_describer()
    paths = state["best_body_crops"]

    crops = []
    for p in paths:
        try:
            img = cv2.imread(str(Path(p).resolve()))
            if img is not None:
                crops.append(img)
        except Exception:
            continue

    if not crops:
        print("[describe_clothing] no crops available")
        return {
            "clothing_raw": "",
            "clothing_structured": {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"},
        }

    raw, structured = describer.describe(crops)
    print(f"[describe_clothing] {structured}")
    return {"clothing_raw": raw, "clothing_structured": structured}
