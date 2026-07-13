import os
from pathlib import Path

from forensics.person_creation.utils.profiling import profile_measure

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")


def load_models(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.models.pose_estimator import get_pose_estimator
    from forensics.person_creation.models.reid_extractor import get_reid_extractor, normalize_reid_config
    from forensics.face_engine.client import FaceEngineClient

    with profile_measure(
        "model.person_detector.load", metadata={"model": Path(_YOLO_MODEL_PATH).name}
    ):
        get_person_detector().load(model_path=_YOLO_MODEL_PATH, device="auto")
    with profile_measure("model.face_engine.health"):
        FaceEngineClient().ensure_healthy()
    with profile_measure(
        "model.clothing_vlm.load", metadata={"model": _INTERNVL_MODEL_ID}
    ):
        get_clothing_describer().load(model_id=_INTERNVL_MODEL_ID, device="auto")

    # Optional auto_pair pose cue. It is a no-op when the optional dependency
    # is not installed.
    with profile_measure("model.pose_estimator.load"):
        get_pose_estimator().load(device="auto")

    reid_config = normalize_reid_config(state.get("reid_config"))
    with profile_measure(
        "model.reid.load", metadata={"model": reid_config.get("model")}
    ):
        reid_model = get_reid_extractor(config=reid_config, device="auto")
    reid_available = reid_model.is_available()
    reid_unavailable_reason = reid_model.unavailable_reason or ""

    print("[load_models] all models ready")
    return {
        "reid_config": reid_config,
        "reid_available": reid_available,
        "reid_unavailable_reason": reid_unavailable_reason,
    }
