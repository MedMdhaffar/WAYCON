"""Environment-driven device and memory configuration for person_creation.

Every heavy node reads its device and batch size from here instead of
hardcoding "auto". Defaults are chosen so the pipeline survives small GPUs
(RTX 2050 4GB class): the frame detectors may use CUDA, but FaceNet, ReID
and the clothing VLM fall back to CPU unless explicitly overridden.

Environment variables:
    WAYCON_LOW_MEMORY          (default 1) sequential model loading + small batches
    WAYCON_DETECTOR_DEVICE     device for person/face detectors (default: cuda if available)
    WAYCON_FACE_DEVICE         device for FaceNet
    WAYCON_REID_DEVICE         device for OSNet ReID
    WAYCON_VLM_DEVICE          device for InternVL clothing model
    WAYCON_FORCE_CPU_FACE / WAYCON_FORCE_CPU_REID / WAYCON_FORCE_CPU_VLM
    WAYCON_FACE_BATCH_SIZE / WAYCON_REID_BATCH_SIZE / WAYCON_VLM_BATCH_SIZE

Attributes are resolved lazily (PEP 562) so importing this module never
imports torch by itself.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}

# GPUs at or below this VRAM (GB) get CPU defaults for FaceNet/ReID/VLM.
SMALL_GPU_VRAM_GB = 4.5

YOLO_MODEL_PATH = str(Path(__file__).parents[3] / "yolo26m.pt")
INTERNVL_MODEL_ID = os.getenv("PERSON_CREATION_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        return default


_cuda_info_cache: tuple[bool, str, float] | None = None


def _cuda_info() -> tuple[bool, str, float]:
    """(available, device_name, total_vram_gb) — never raises."""
    global _cuda_info_cache
    if _cuda_info_cache is None:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                _cuda_info_cache = (True, props.name, props.total_memory / 2**30)
            else:
                _cuda_info_cache = (False, "", 0.0)
        except Exception:
            _cuda_info_cache = (False, "", 0.0)
    return _cuda_info_cache


def cuda_available() -> bool:
    return _cuda_info()[0]


def total_vram_gb() -> float:
    return _cuda_info()[2]


def is_small_gpu() -> bool:
    return cuda_available() and total_vram_gb() <= SMALL_GPU_VRAM_GB


def _validate_device(device: str) -> str:
    device = device.strip().lower()
    if device.startswith("cuda") and not cuda_available():
        print(f"[config] requested device '{device}' but CUDA is unavailable — using cpu")
        return "cpu"
    return device


def _detector_device() -> str:
    explicit = os.getenv("WAYCON_DETECTOR_DEVICE")
    if explicit:
        return _validate_device(explicit)
    return "cuda" if cuda_available() else "cpu"


def _aux_device(kind: str) -> str:
    """Device for face/reid/vlm. CPU on small GPUs or in low-memory mode."""
    if _flag(f"WAYCON_FORCE_CPU_{kind.upper()}", False):
        return "cpu"
    explicit = os.getenv(f"WAYCON_{kind.upper()}_DEVICE")
    if explicit:
        return _validate_device(explicit)
    if not cuda_available():
        return "cpu"
    if is_small_gpu() or _flag("WAYCON_LOW_MEMORY", True):
        return "cpu"
    return "cuda"


def __getattr__(name: str):
    low = _flag("WAYCON_LOW_MEMORY", True)
    resolvers = {
        "LOW_MEMORY_MODE": lambda: low,
        "DETECTOR_DEVICE": _detector_device,
        "FACE_DEVICE": lambda: _aux_device("face"),
        "REID_DEVICE": lambda: _aux_device("reid"),
        "VLM_DEVICE": lambda: _aux_device("vlm"),
        "FACE_BATCH_SIZE": lambda: _int("WAYCON_FACE_BATCH_SIZE", 4 if low else 16),
        "REID_BATCH_SIZE": lambda: _int("WAYCON_REID_BATCH_SIZE", 1 if low else 8),
        "VLM_BATCH_SIZE": lambda: _int("WAYCON_VLM_BATCH_SIZE", 1 if low else 5),
    }
    if name in resolvers:
        return resolvers[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def print_memory_config() -> None:
    available, gpu_name, vram = _cuda_info()
    print("[config] memory configuration:")
    if available:
        small = " (small GPU: CPU defaults for face/reid/vlm)" if is_small_gpu() else ""
        print(f"[config]   cuda: {gpu_name} {vram:.1f}GB{small}")
    else:
        print("[config]   cuda: unavailable — all models on cpu")
    mod = __import__(__name__, fromlist=["_"])
    print(f"[config]   low_memory_mode: {mod.LOW_MEMORY_MODE}")
    print(f"[config]   detector_device: {mod.DETECTOR_DEVICE}")
    print(f"[config]   face_device: {mod.FACE_DEVICE} (batch={mod.FACE_BATCH_SIZE})")
    print(f"[config]   reid_device: {mod.REID_DEVICE} (batch={mod.REID_BATCH_SIZE})")
    print(f"[config]   vlm_device: {mod.VLM_DEVICE} (batch={mod.VLM_BATCH_SIZE})")
    print(f"[config]   vlm_model: {INTERNVL_MODEL_ID}")
