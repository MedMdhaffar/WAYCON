from __future__ import annotations

import os

_LOGGED_DEVICE_INFO = False


def resolve_device(device: str = "auto") -> str:
    """Return cuda when available, otherwise cpu.

    The project is often tested on laptops where the NVIDIA driver exists but
    PyTorch may still be CPU-only. Checking torch avoids crashing at model load.
    """
    requested = os.getenv("PERSON_CREATION_DEVICE", device).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("PERSON_CREATION_DEVICE must be one of: auto, cuda, cpu")

    try:
        import torch
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested with PERSON_CREATION_DEVICE=cuda, "
                    "but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build."
                )
            return "cuda"
        if requested == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError as exc:
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested with PERSON_CREATION_DEVICE=cuda, but torch is not installed "
                "in this Python environment."
            ) from exc
        return "cpu"


def device_info(device: str = "auto") -> dict:
    requested = os.getenv("PERSON_CREATION_DEVICE", device).strip().lower()
    info = {
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "requested_device": requested,
        "selected_device": "cpu" if requested == "auto" else requested,
        "gpu_name": None,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        selected = resolve_device(device)
        info.update({
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": cuda_available,
            "selected_device": selected,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        })
    except Exception as exc:
        info["error"] = str(exc)
    return info


def log_device_info_once(device: str = "auto") -> dict:
    global _LOGGED_DEVICE_INFO
    info = device_info(device)
    if not _LOGGED_DEVICE_INFO:
        _LOGGED_DEVICE_INFO = True
        print(
            "[device] "
            f"torch={info.get('torch_version')} "
            f"torch_cuda={info.get('torch_cuda_version')} "
            f"cuda_available={info.get('cuda_available')} "
            f"selected={info.get('selected_device')} "
            f"gpu={info.get('gpu_name')}"
        )
    return info
