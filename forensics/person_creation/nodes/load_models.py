import os
from pathlib import Path

from forensics.person_creation.models.device import resolve_device

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-2B")


def load_models(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.face_detector import get_face_detector
    from forensics.person_creation.models.face_embedder import get_face_embedder
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.models.body_reid import get_body_reid
    from forensics.person_creation.models.pose_estimator import get_pose_estimator

    device = resolve_device("auto")

    get_person_detector().load(model_path=_YOLO_MODEL_PATH, device=device)
    get_face_detector().load(device=device)
    get_face_embedder().load(device=device)
    get_clothing_describer().load(model_id=_INTERNVL_MODEL_ID, device=device)

    # Optional auto_pair association cues. Both are no-ops (cue disabled,
    # nothing raised) when their optional dependency isn't installed.
    get_body_reid().load(device=device)
    get_pose_estimator().load(device=device)

    print("[load_models] all models ready")
    return {}
