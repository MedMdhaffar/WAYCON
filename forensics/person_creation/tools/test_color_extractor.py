import argparse

from forensics.person_creation.models.color_extractor import COLOR_EXTRACTOR_NAME
from forensics.person_creation.nodes.extract_colors import _extract_for_paths


def _line(label: str, block: dict | None) -> str:
    if not block:
        return f"{label}: unknown"
    return (
        f"{label}: {block.get('dominant')}, "
        f"confidence={block.get('confidence')}, rgb_mean={block.get('rgb_mean')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deterministic body-crop color signals.")
    parser.add_argument("--crops", nargs="+", required=True, help="Body crop image paths")
    args = parser.parse_args()

    signals, debug = _extract_for_paths(args.crops)
    per_crop = debug.get("per_crop") or []

    print(f"Extractor: {COLOR_EXTRACTOR_NAME}")
    print(f"Valid crops: {len(signals.get('source_crops') or [])}")
    print("")
    for item in per_crop:
        top = ((item.get("top") or {}).get("dominant") or {}).get("name")
        bottom = ((item.get("bottom") or {}).get("dominant") or {}).get("name")
        shoes = ((item.get("shoes") or {}).get("dominant") or {}).get("name")
        print(f"{item.get('crop')}: top={top} bottom={bottom} shoes={shoes}")

    print("")
    print("Aggregated:")
    print(_line("Top", signals.get("top")))
    print(_line("Bottom", signals.get("bottom")))
    print(_line("Shoes", signals.get("shoes")))
    if signals.get("error"):
        print(f"Error: {signals['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
