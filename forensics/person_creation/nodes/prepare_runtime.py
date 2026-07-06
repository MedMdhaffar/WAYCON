"""Lightweight replacement for the old `load_models` node.

It used to eagerly load every heavy model (person/face detectors, FaceNet,
InternVL, OSNet) before the graph even started, which is what pinned 6GB+ of
GPU/RAM for the whole pipeline run. Models are now loaded lazily inside the
node that actually needs them (see nodes/process_video.py, embed_faces.py,
describe_clothing_per_person.py, extract_reid.py) and released right after.

This node only validates the environment: output dir is writable, the local
YOLO weights file exists, and CUDA is reported if requested.
"""

from pathlib import Path

from forensics.person_creation import config

_YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")


def prepare_runtime(state: dict) -> dict:
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(_YOLO_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Person detector weights not found at {_YOLO_MODEL_PATH!r}. "
            "This must exist before process_video can run."
        )

    for video_path in state.get("video_paths") or []:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path!r}")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    print(
        f"[prepare_runtime] cuda_available={cuda_available} "
        f"low_memory_mode={config.LOW_MEMORY_MODE} "
        f"detector_device={config.DETECTOR_DEVICE} face_device={config.FACE_DEVICE} "
        f"vlm_device={config.VLM_DEVICE} reid_device={config.REID_DEVICE}"
    )
    print("[prepare_runtime] ready — models load lazily inside each node")
    return {}
