from pathlib import Path

import cv2


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


def describe_clothing_per_person(state: dict) -> dict:
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    describer = get_clothing_describer()
    best_by_person = state.get("best_body_crops_by_person") or {}
    raw_by_person: dict[str, str] = {}
    clothing_by_person: dict[str, dict] = {}

    for person_id, paths in best_by_person.items():
        crops = _read_crops(paths)
        if not crops:
            raw_by_person[person_id] = ""
            clothing_by_person[person_id] = dict(_UNKNOWN)
            continue

        raw, structured = describer.describe(crops)
        raw_by_person[person_id] = raw
        clothing_by_person[person_id] = structured
        print(f"[describe_clothing_per_person] {person_id}: {structured}")

    first_person = next(iter(clothing_by_person), None)
    return {
        "clothing_raw_by_person": raw_by_person,
        "clothing_by_person": clothing_by_person,
        "clothing_raw": raw_by_person.get(first_person, "") if first_person else "",
        "clothing_structured": clothing_by_person.get(first_person, dict(_UNKNOWN)) if first_person else dict(_UNKNOWN),
    }
