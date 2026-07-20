import os
from pathlib import Path

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")


def load_models(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.models.pose_estimator import get_pose_estimator
    from forensics.person_creation.models.reid_extractor import get_reid_extractor, normalize_reid_config
    from forensics.face_engine.local_client import LocalFaceEngine

    # Every underlying .load() (person detector, face detector/embedder, clothing
    # describer, pose, reid) is itself guarded against reloading once resident, so
    # this node is safe -- and cheap -- to run more than once per process (e.g. once
    # per segment/job in the current per-request graph.stream() model). Models load
    # once at process startup and stay resident for the process lifetime.
    get_person_detector().load(model_path=_YOLO_MODEL_PATH, device="auto")
    LocalFaceEngine().ensure_healthy()
    get_clothing_describer().load(model_id=_INTERNVL_MODEL_ID, device="auto")

    # Optional auto_pair pose cue. It is a no-op when the optional dependency
    # is not installed.
    get_pose_estimator().load(device="auto")

    reid_config = normalize_reid_config(state.get("reid_config"))
    reid_model = get_reid_extractor(config=reid_config, device="auto")
    reid_available = reid_model.is_available()
    reid_unavailable_reason = reid_model.unavailable_reason or ""

    print("[load_models] all models ready")
    return {
        "reid_config": reid_config,
        "reid_available": reid_available,
        "reid_unavailable_reason": reid_unavailable_reason,
    }
