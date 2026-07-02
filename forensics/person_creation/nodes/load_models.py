import os
from pathlib import Path

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")


def load_models(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.face_detector import get_face_detector
    from forensics.person_creation.models.face_embedder import get_face_embedder
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    get_person_detector().load(model_path=_YOLO_MODEL_PATH, device="auto")
    get_face_detector().load(device="auto")
    get_face_embedder().load(device="auto")
    get_clothing_describer().load(model_id=_INTERNVL_MODEL_ID, device="auto")

    print("[load_models] all models ready")
    return {}
