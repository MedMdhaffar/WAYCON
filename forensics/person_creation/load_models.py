"""Model loading, called once before the pipeline starts -- not a LangGraph node.

    RTSP camera -> load_models() -> GStreamer pipeline -> Tier 0 presence gate
    -> segment accumulator -> queue -> sync pipeline (this graph) -> ...

Previously `load_models` was the first node in graph.py, re-run (cheaply, thanks to
the reload guards on the individual model loaders) on every `graph.stream()` call.
Per the realtime architecture, models load once, resident for the process lifetime,
before ingestion (the GStreamer pipeline) even starts -- so this is now a plain
function the process entrypoint calls once, not a node the graph re-enters per
segment. service.py calls it eagerly at process startup and merges its return value
into every job's initial_state, exactly like the old node's output used to be merged
into graph state.
"""

import os
import threading
from pathlib import Path

_YOLO_MODEL_PATH = str(Path(__file__).parents[2] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")

_lock = threading.Lock()
_loaded = False
_last_result: dict | None = None


def load_models(reid_config: dict | None = None) -> dict:
    """Load every model once, resident for the process lifetime.

    Every underlying .load() (person detector, face detector/embedder, clothing
    describer, pose, reid) is itself guarded against reloading once resident, so this
    function is safe -- and cheap -- to call more than once per process; only the
    first call does real work for a given reid_config.
    """
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.models.pose_estimator import get_pose_estimator
    from forensics.person_creation.models.reid_extractor import get_reid_extractor, normalize_reid_config
    from forensics.face_engine.local_client import LocalFaceEngine

    global _loaded, _last_result

    get_person_detector().load(model_path=_YOLO_MODEL_PATH, device="auto")
    LocalFaceEngine().ensure_healthy()
    get_clothing_describer().load(model_id=_INTERNVL_MODEL_ID, device="auto")

    # Optional auto_pair pose cue. It is a no-op when the optional dependency
    # is not installed.
    get_pose_estimator().load(device="auto")

    normalized_reid_config = normalize_reid_config(reid_config)
    reid_model = get_reid_extractor(config=normalized_reid_config, device="auto")
    reid_available = reid_model.is_available()
    reid_unavailable_reason = reid_model.unavailable_reason or ""

    print("[load_models] all models ready")
    result = {
        "reid_config": normalized_reid_config,
        "reid_available": reid_available,
        "reid_unavailable_reason": reid_unavailable_reason,
    }
    with _lock:
        _loaded = True
        _last_result = result
    return result


def models_loaded() -> bool:
    """Cheap health-check predicate -- True once load_models() has run at least once."""
    with _lock:
        return _loaded


def last_load_result() -> dict | None:
    with _lock:
        return dict(_last_result) if _last_result is not None else None
