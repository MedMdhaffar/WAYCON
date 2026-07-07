"""Lightweight replacement for the old eager `load_models` node.

Validates inputs and reports the memory configuration but loads NO heavy
models — each heavy node (process_video, embed_all_faces, compute_reid,
describe_clothing) now loads its own model on entry and releases it on exit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def prepare_runtime(state: dict) -> dict:
    """Validate paths, create output directory, print memory config,
    check CUDA availability, but do NOT load heavy models.
    """
    from forensics.person_creation import config
    from forensics.person_creation.models.reid_extractor import normalize_reid_config
    from forensics.person_creation.utils.memory import log_memory

    missing = [p for p in state.get("video_paths", []) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"video_paths not found on disk: {missing}")

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config.print_memory_config()
    log_memory("prepare_runtime")

    # ReID availability is a dependency check only — the model itself is
    # loaded (and released) inside compute_reid.
    reid_config = normalize_reid_config(state.get("reid_config"))
    reid_available = importlib.util.find_spec("torchreid") is not None
    reid_unavailable_reason = "" if reid_available else "torchreid is not installed"
    if not reid_available:
        print("[prepare_runtime] torchreid not installed — profile ReID will be skipped")

    print("[prepare_runtime] runtime ready (no heavy models loaded)")
    return {
        "reid_config": reid_config,
        "reid_available": reid_available,
        "reid_unavailable_reason": reid_unavailable_reason,
    }
