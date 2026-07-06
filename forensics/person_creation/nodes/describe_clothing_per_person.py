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
    from forensics.person_creation.models.clothing_describer import (
        get_clothing_describer,
        release_clothing_describer,
    )
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom
    from forensics.person_creation import config

    best_by_person = state.get("best_body_crops_by_person") or {}
    raw_by_person: dict[str, str] = {}
    clothing_by_person: dict[str, dict] = {}

    if not best_by_person:
        return {
            "clothing_raw_by_person": {},
            "clothing_by_person": {},
            "clothing_raw": "",
            "clothing_structured": dict(_UNKNOWN),
        }

    describer = get_clothing_describer()
    log_memory("before loading InternVL")
    try:
        describer.load(device=config.VLM_DEVICE)
        log_memory("after loading InternVL")

        for person_id, paths in best_by_person.items():
            # VLM_BATCH_SIZE caps how many crops go into a single forward
            # pass — keep it low (default 1) on tight-VRAM GPUs.
            capped_paths = list(paths)[:config.VLM_BATCH_SIZE] if config.LOW_MEMORY_MODE else list(paths)
            crops = _read_crops(capped_paths)
            if not crops:
                raw_by_person[person_id] = ""
                clothing_by_person[person_id] = dict(_UNKNOWN)
                continue

            raw, structured = describer.describe(crops)
            raw_by_person[person_id] = raw
            clothing_by_person[person_id] = structured
            print(f"[describe_clothing_per_person] {person_id}: {structured}")
            del crops
            cleanup_memory(f"describe_clothing_per_person/{person_id}")
    except Exception as exc:
        raise clarify_oom(exc, "describe_clothing_per_person") from exc
    finally:
        release_clothing_describer()
        cleanup_memory("describe_clothing_per_person")

    first_person = next(iter(clothing_by_person), None)
    return {
        "clothing_raw_by_person": raw_by_person,
        "clothing_by_person": clothing_by_person,
        "clothing_raw": raw_by_person.get(first_person, "") if first_person else "",
        "clothing_structured": clothing_by_person.get(first_person, dict(_UNKNOWN)) if first_person else dict(_UNKNOWN),
    }
