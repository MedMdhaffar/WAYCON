from pathlib import Path

from forensics.person_creation.models.color_extractor import (
    COLOR_EXTRACTOR_NAME,
    SIGNAL_TYPE,
    aggregate_color_signals,
    get_color_extractor,
)


def _error_block(message: str, source_crops: list[str] | None = None) -> dict:
    return {
        "extractor": COLOR_EXTRACTOR_NAME,
        "source_crops": source_crops or [],
        "top": None,
        "bottom": None,
        "shoes": None,
        "per_crop_count": 0,
        "signal_type": SIGNAL_TYPE,
        "error": message,
    }


def _extract_for_paths(paths: list[str]) -> tuple[dict, dict]:
    valid_paths = [p for p in paths if p and Path(p).exists()]
    for path in [p for p in paths if p and not Path(p).exists()]:
        print(f"[extract_colors][warn] missing crop skipped: {path}")

    if not valid_paths:
        return _error_block("No valid body crops for color extraction"), {"per_crop": []}

    try:
        per_crop = get_color_extractor().extract_batch(valid_paths)
    except Exception as exc:
        return _error_block(str(exc), valid_paths), {"per_crop": []}

    if not per_crop:
        return _error_block("No color signals produced", valid_paths), {"per_crop": []}

    aggregated = aggregate_color_signals(per_crop)
    return aggregated, {"per_crop": per_crop}


def extract_colors(state: dict) -> dict:
    """Compute deterministic same-day color signals from selected body crops."""

    best_by_person = state.get("best_body_crops_by_person") or {}
    signals_by_person: dict[str, dict] = {}
    debug_by_person: dict[str, dict] = {}

    if best_by_person:
        for person_id, paths in best_by_person.items():
            signals, debug = _extract_for_paths(list(paths or []))
            signals_by_person[person_id] = signals
            debug_by_person[person_id] = debug
            if signals.get("error"):
                print(f"[extract_colors] {person_id}: {signals['error']}")
            else:
                top = signals.get("top") or {}
                bottom = signals.get("bottom") or {}
                shoes = signals.get("shoes") or {}
                print(
                    f"[extract_colors] {person_id}: "
                    f"top={top.get('dominant')} bottom={bottom.get('dominant')} "
                    f"shoes={shoes.get('dominant')}"
                )

        first_person = next(iter(signals_by_person), None)
        return {
            "color_signals_by_person": signals_by_person,
            "color_signals_debug_by_person": debug_by_person,
            "color_signals": signals_by_person.get(first_person, _error_block("No valid body crops for color extraction")),
            "color_signals_debug": debug_by_person.get(first_person, {"per_crop": []}),
        }

    signals, debug = _extract_for_paths(list(state.get("best_body_crops") or []))
    if signals.get("error"):
        print(f"[extract_colors] {signals['error']}")
    else:
        print(
            f"[extract_colors] top={(signals.get('top') or {}).get('dominant')} "
            f"bottom={(signals.get('bottom') or {}).get('dominant')} "
            f"shoes={(signals.get('shoes') or {}).get('dominant')}"
        )
    return {
        "color_signals": signals,
        "color_signals_debug": debug,
        "color_signals_by_person": {},
        "color_signals_debug_by_person": {},
    }
