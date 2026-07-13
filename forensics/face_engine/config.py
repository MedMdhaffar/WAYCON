from __future__ import annotations

import os
from pathlib import Path

from forensics.global_memory.config import SIMILARITY_THRESHOLD


PORT: int = int(os.environ.get("FACE_ENGINE_PORT", "5010"))
HOST: str = os.environ.get("FACE_ENGINE_HOST", "0.0.0.0")
BASE_URL: str = os.environ.get("FACE_ENGINE_URL", f"http://localhost:{PORT}")
MODEL_CACHE_DIR: str | None = os.environ.get("MODEL_CACHE_DIR") or None
REQUEST_TIMEOUT: float = float(os.environ.get("FACE_ENGINE_TIMEOUT", "30"))
DEVICE: str = os.environ.get(
    "FACE_ENGINE_DEVICE", os.environ.get("PERSON_CREATION_DEVICE", "auto")
)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def model_cache_path() -> Path | None:
    return Path(MODEL_CACHE_DIR).expanduser() if MODEL_CACHE_DIR else None
