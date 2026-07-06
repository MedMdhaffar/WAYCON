"""Memory/device configuration for the person_creation pipeline.

Everything here can be overridden with environment variables so a given
machine's limits don't have to be hardcoded. Defaults are chosen to be safe
on a 4GB GPU: models are loaded one at a time (see nodes/*.py + models/*.py)
and batch sizes are kept small.
"""

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


def _int_env(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(env_var: str, default: str = "cuda", force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    requested = os.environ.get(env_var, default).strip().lower()
    if requested == "cuda" and not _cuda_available():
        print(f"[config] CUDA requested via {env_var} but not available — falling back to CPU")
        return "cpu"
    return requested


# Low-memory mode: sequential model loading everywhere it's an option
# (e.g. person/face detectors run in two passes instead of concurrently).
LOW_MEMORY_MODE = _bool_env("WAYCON_LOW_MEMORY", True)           #i should test if this works 1 == slow , 0 == faster (i will test both)

# Batch sizes. Small by default so a single forward pass doesn't spike VRAM.
VLM_BATCH_SIZE = _int_env("WAYCON_VLM_BATCH_SIZE", 1)
REID_BATCH_SIZE = _int_env("WAYCON_REID_BATCH_SIZE", 4)
FACE_BATCH_SIZE = _int_env("WAYCON_FACE_BATCH_SIZE", 8)

FORCE_CPU_FOR_VLM = _bool_env("WAYCON_FORCE_CPU_VLM", False)
FORCE_CPU_FOR_REID = _bool_env("WAYCON_FORCE_CPU_REID", False)

CLEAR_CUDA_AFTER_NODE = _bool_env("WAYCON_CLEAR_CUDA_AFTER_NODE", True)

DETECTOR_DEVICE = resolve_device("WAYCON_DETECTOR_DEVICE", "cuda")
# FaceNet is tiny and the original implementation always ran it on CPU;
# keep that as the default so it never competes with detectors/VLM/ReID for
# VRAM. Set WAYCON_FACE_DEVICE=cuda to opt into GPU for speed.
FACE_DEVICE = resolve_device("WAYCON_FACE_DEVICE", "cpu")
VLM_DEVICE = resolve_device("WAYCON_VLM_DEVICE", "cuda", force_cpu=FORCE_CPU_FOR_VLM)
REID_DEVICE = resolve_device("WAYCON_REID_DEVICE", "cuda", force_cpu=FORCE_CPU_FOR_REID)
