import os
from pathlib import Path

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")
_INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")


def load_models(state: dict) -> dict:
    from forensics.person_creation.models.device import (
        set_device_status,
        resolve_device,
        validate_model_devices,
    )
    from forensics.person_creation.models.body_reid import get_body_reid
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.models.pose_estimator import get_pose_estimator
    from forensics.person_creation.models.reid_extractor import get_reid_extractor, normalize_reid_config
    from forensics.face_engine.client import FaceEngineClient

    requested_device = str(state.get("device") or "auto")
    env_device = os.getenv("PERSON_CREATION_DEVICE")
    resolved_device = resolve_device(requested_device)
    initial_devices = {
        "person_detector": "not_loaded",
        "clothing_describer": "not_loaded",
        "pose_estimator": "not_loaded",
        "reid_extractor": "not_loaded",
        "body_reid": get_body_reid().device,
    }
    set_device_status(
        requested=requested_device,
        environment=env_device,
        resolved=resolved_device,
        models=initial_devices,
        models_loaded=False,
    )
    print(
        "[load_models] device selection: "
        f"requested={requested_device!r} "
        f"PERSON_CREATION_DEVICE={env_device!r} "
        f"resolved={resolved_device!r}"
    )

    person_detector = get_person_detector()
    person_detector.load(model_path=_YOLO_MODEL_PATH, device=resolved_device)
    FaceEngineClient().ensure_healthy()
    clothing_describer = get_clothing_describer()
    clothing_describer.load(model_id=_INTERNVL_MODEL_ID, device=resolved_device)

    # Optional auto_pair pose cue. It is a no-op when the optional dependency
    # is not installed.
    pose_estimator = get_pose_estimator()
    pose_estimator.load(device=resolved_device)

    reid_config = normalize_reid_config(state.get("reid_config"))
    reid_model = get_reid_extractor(config=reid_config, device=resolved_device)
    reid_available = reid_model.is_available()
    reid_unavailable_reason = reid_model.unavailable_reason or ""

    actual_devices = {
        "person_detector": person_detector.device,
        "clothing_describer": clothing_describer.device,
        "pose_estimator": pose_estimator.device,
        "reid_extractor": reid_model.device,
        "body_reid": get_body_reid().device,
    }
    set_device_status(
        requested=requested_device,
        environment=env_device,
        resolved=resolved_device,
        models=actual_devices,
        models_loaded=False,
    )
    validate_model_devices(
        requested=requested_device,
        resolved=resolved_device,
        models=actual_devices,
    )
    set_device_status(
        requested=requested_device,
        environment=env_device,
        resolved=resolved_device,
        models=actual_devices,
        models_loaded=True,
    )
    print(
        "[load_models] actual devices: "
        + " ".join(f"{name}={device}" for name, device in actual_devices.items())
    )

    print("[load_models] all models ready")
    return {
        "reid_config": reid_config,
        "reid_available": reid_available,
        "reid_unavailable_reason": reid_unavailable_reason,
    }
