from __future__ import annotations

import os
import re
import threading
from copy import deepcopy


_DEVICE_RE = re.compile(r"^(cpu|cuda(?::\d+)?)$")
_DEVICE_STATES = {"not_loaded", "disabled", "unavailable"}
_STATUS_LOCK = threading.Lock()
_DEVICE_STATUS = {
    "models_loaded": False,
    "device": {
        "requested": "auto",
        "environment": None,
        "resolved": "not_loaded",
        "models": {
            "person_detector": "not_loaded",
            "clothing_describer": "not_loaded",
            "pose_estimator": "not_loaded",
            "reid_extractor": "not_loaded",
            "body_reid": "not_loaded",
        },
    },
}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _normalize_device(value: str) -> str:
    normalized = value.strip().lower()
    if not _DEVICE_RE.fullmatch(normalized):
        raise ValueError(
            "Unsupported PERSON_CREATION device value "
            f"{value!r}; expected auto, cpu, cuda, or cuda:<index>."
        )
    if normalized == "cuda":
        return "cuda:0"
    return normalized


def is_cuda_device(device: str) -> bool:
    return device.strip().lower().startswith("cuda")


def normalize_loaded_device(value: object, default: str = "not_loaded") -> str:
    """Normalize a device reported by a loaded model or runtime library."""
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _DEVICE_STATES:
        return normalized
    if normalized == "cuda":
        return "cuda:0"
    if _DEVICE_RE.fullmatch(normalized):
        return normalized
    return default


def model_parameter_device(model: object, fallback: str = "not_loaded") -> str:
    """Read a torch-like model's actual parameter device without exposing it."""
    if model is None:
        return fallback
    candidates = (model, getattr(model, "model", None))
    for candidate in candidates:
        if candidate is None:
            continue
        device_map = getattr(candidate, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            mapped = {
                normalize_loaded_device(value, default="")
                for value in device_map.values()
            }
            mapped.discard("")
            cuda_devices = sorted(device for device in mapped if is_cuda_device(device))
            if cuda_devices:
                return cuda_devices[0]
            if len(mapped) == 1:
                return mapped.pop()
        direct = normalize_loaded_device(getattr(candidate, "device", None), default="")
        if direct:
            return direct
        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            try:
                parameter = next(iter(parameters()))
            except (StopIteration, TypeError, RuntimeError):
                continue
            actual = normalize_loaded_device(getattr(parameter, "device", None), default="")
            if actual:
                return actual
    return normalize_loaded_device(fallback)


def validate_model_devices(
    *, requested: str, resolved: str, models: dict[str, str]
) -> None:
    """Reject successfully loaded CUDA models when CPU was resolved."""
    if normalize_loaded_device(resolved) != "cpu":
        return
    for model_name, actual in models.items():
        if is_cuda_device(str(actual)):
            raise RuntimeError(
                "Person-creation device mismatch: "
                f"model={model_name!r} requested={requested!r} "
                f"resolved={resolved!r} actual={actual!r}."
            )


def set_device_status(
    *,
    requested: str,
    environment: str | None,
    resolved: str,
    models: dict[str, str],
    models_loaded: bool,
) -> None:
    normalized_models = {
        name: normalize_loaded_device(device)
        for name, device in models.items()
    }
    with _STATUS_LOCK:
        _DEVICE_STATUS["models_loaded"] = bool(models_loaded)
        _DEVICE_STATUS["device"] = {
            "requested": requested,
            "environment": environment,
            "resolved": normalize_loaded_device(resolved),
            "models": normalized_models,
        }


def get_device_status() -> dict:
    with _STATUS_LOCK:
        return deepcopy(_DEVICE_STATUS)


def resolve_device(requested: str | None = None) -> str:
    """Resolve the concrete device for person-creation model loading.

    Priority:
    1. A concrete caller-supplied device: cpu, cuda, or cuda:<index>.
    2. PERSON_CREATION_DEVICE when the caller supplies auto/None.
    3. Automatic selection: CUDA when available, otherwise CPU.
    """
    requested_value = (requested or "auto").strip().lower()
    if requested_value and requested_value != "auto":
        resolved = _normalize_device(requested_value)
    else:
        env_value = os.getenv("PERSON_CREATION_DEVICE")
        if env_value and env_value.strip().lower() != "auto":
            resolved = _normalize_device(env_value)
        elif env_value and env_value.strip().lower() == "auto":
            resolved = "cuda:0" if _cuda_available() else "cpu"
        else:
            resolved = "cuda:0" if _cuda_available() else "cpu"

    if is_cuda_device(resolved) and not _cuda_available():
        raise RuntimeError(
            f"PERSON_CREATION resolved device {resolved!r}, but CUDA is not available."
        )
    return resolved
