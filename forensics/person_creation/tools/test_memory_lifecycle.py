"""Validate that every heavy model can be loaded and released cleanly.

Run:
    python -m forensics.person_creation.tools.test_memory_lifecycle
    python -m forensics.person_creation.tools.test_memory_lifecycle --include-vlm

This does not run the pipeline — it just exercises each model's
get_xxx()/load()/release_xxx() lifecycle in isolation and prints GPU memory
before/after so you can confirm cleanup actually happens. Exits non-zero if
any load/release raises.
"""

import argparse
import traceback

from forensics.person_creation.utils.memory import log_memory, cleanup_memory
from forensics.person_creation import config


def _check(label: str, load_fn, release_fn) -> bool:
    print(f"\n=== {label} ===")
    log_memory(f"before loading {label}")
    try:
        load_fn()
        log_memory(f"after loading {label}")
    except Exception:
        print(f"[test_memory_lifecycle] FAILED to load {label}:")
        traceback.print_exc()
        try:
            release_fn()
        except Exception:
            pass
        cleanup_memory(label)
        return False

    try:
        release_fn()
    except Exception:
        print(f"[test_memory_lifecycle] FAILED to release {label}:")
        traceback.print_exc()
        return False

    log_memory(f"after releasing {label}")
    cleanup_memory(label)
    return True


def _check_person_detector() -> bool:
    from forensics.person_creation.models.person_detector import (
        get_person_detector,
        release_person_detector,
    )
    from pathlib import Path

    yolo_path = str(Path(__file__).parents[4] / "yolo26m.pt")
    if not Path(yolo_path).exists():
        print(f"[test_memory_lifecycle] skipping person_detector — weights not found at {yolo_path}")
        return True
    return _check(
        "person_detector",
        lambda: get_person_detector().load(model_path=yolo_path, device=config.DETECTOR_DEVICE),
        release_person_detector,
    )


def _check_face_detector() -> bool:
    from forensics.person_creation.models.face_detector import (
        get_face_detector,
        release_face_detector,
    )
    return _check(
        "face_detector",
        lambda: get_face_detector().load(device=config.DETECTOR_DEVICE),
        release_face_detector,
    )


def _check_face_embedder() -> bool:
    from forensics.person_creation.models.face_embedder import (
        get_face_embedder,
        release_face_embedder,
    )
    return _check(
        "face_embedder",
        lambda: get_face_embedder().load(device=config.FACE_DEVICE),
        release_face_embedder,
    )


def _check_reid_embedder() -> bool:
    from forensics.person_creation.models.reid_embedder import (
        get_reid_embedder,
        release_reid_embedder,
    )
    return _check(
        "reid_embedder",
        lambda: get_reid_embedder().load(device=config.REID_DEVICE),
        release_reid_embedder,
    )


def _check_clothing_describer() -> bool:
    from forensics.person_creation.models.clothing_describer import (
        get_clothing_describer,
        release_clothing_describer,
    )
    return _check(
        "clothing_describer (InternVL)",
        lambda: get_clothing_describer().load(device=config.VLM_DEVICE),
        release_clothing_describer,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-vlm", action="store_true",
        help="Also load/release InternVL (slow, largest download+memory footprint)",
    )
    args = parser.parse_args()

    results = {
        "person_detector": _check_person_detector(),
        "face_detector": _check_face_detector(),
        "face_embedder": _check_face_embedder(),
        "reid_embedder": _check_reid_embedder(),
    }
    if args.include_vlm:
        results["clothing_describer"] = _check_clothing_describer()

    print("\n=== summary ===")
    ok = True
    for name, passed in results.items():
        print(f"{name}: {'OK' if passed else 'FAILED'}")
        ok = ok and passed

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
