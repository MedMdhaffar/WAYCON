from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Return cuda when available, otherwise cpu.

    The project is often tested on laptops where the NVIDIA driver exists but
    PyTorch may still be CPU-only. Checking torch avoids crashing at model load.
    """
    if device != "auto":
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
