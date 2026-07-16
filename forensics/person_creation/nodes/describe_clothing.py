"""Per-identity clothing inference with bounded, non-overlapping VLM work."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import cv2


_FAILURE_CATEGORIES = {
    "timeout",
    "empty_output",
    "invalid_output",
    "image_decode_failed",
    "inference_error",
    "no_valid_body_crop",
}
_EMPTY_VALUES = {
    "",
    "unknown",
    "unavailable",
    "clothing description unavailable",
    "clothing description unavailable.",
}


class _ControlledVLMWorker:
    """One bounded inference lane shared by all enrollment jobs."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="person-creation-vlm",
        )
        self._lock = threading.Lock()
        self._pending = 0

    def submit(self, function, *args) -> Future | None:
        with self._lock:
            if self._pending >= 2:
                return None
            self._pending += 1
        try:
            future = self._executor.submit(function, *args)
        except Exception:
            with self._lock:
                self._pending -= 1
            raise
        future.add_done_callback(self._completed)
        return future

    def _completed(self, _future: Future) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending


_VLM_WORKER = _ControlledVLMWorker()


def _read_crops(paths: list[str]) -> tuple[list, list[dict]]:
    crops = []
    metadata = []
    for raw in paths:
        try:
            image = cv2.imread(str(Path(raw).resolve()))
        except Exception:
            image = None
        if image is None or not getattr(image, "size", 0):
            continue
        height, width = image.shape[:2]
        crops.append(image)
        metadata.append({
            "basename": Path(raw).name,
            "width": int(width),
            "height": int(height),
        })
    return crops, metadata


def _clean_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _EMPTY_VALUES else text


def _validated_result(raw, structured) -> tuple[dict | None, str | None]:
    if not str(raw or "").strip() and not structured:
        return None, "empty_output"
    if not isinstance(structured, dict):
        return None, "invalid_output"
    values = {
        "top": _clean_value(structured.get("top")),
        "bottom": _clean_value(structured.get("bottom")),
        "shoes": _clean_value(structured.get("shoes")),
        "full": _clean_value(structured.get("full")),
    }
    if not any(values[key] for key in ("top", "bottom", "shoes")):
        return None, "invalid_output"
    if values["full"] is None:
        values["full"] = ", ".join(
            value for key, value in values.items() if key != "full" and value
        )
    return values, None


def _failed_result(attempts: int, reason: str) -> dict:
    reason = reason if reason in _FAILURE_CATEGORIES else "inference_error"
    return {
        "status": "failed",
        "attempts": int(attempts),
        "top": None,
        "bottom": None,
        "shoes": None,
        "full": None,
        "failure_reason": reason,
    }


def _successful_result(attempts: int, values: dict) -> dict:
    return {
        "status": "ok",
        "attempts": int(attempts),
        **values,
        "failure_reason": None,
    }


def async_vlm_config() -> tuple[bool, float]:
    enabled = os.getenv("PERSON_CREATION_ASYNC_VLM", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    try:
        timeout = float(os.getenv("PERSON_CREATION_VLM_TIMEOUT_SECONDS", "60"))
    except ValueError:
        timeout = 60.0
    return enabled, max(1.0, timeout)


def vlm_max_retries() -> int:
    try:
        retries = int(os.getenv("PERSON_CREATION_VLM_MAX_RETRIES", "1"))
    except ValueError:
        retries = 1
    return max(0, min(retries, 3))


def _attempt_paths(paths: list[str], attempt: int) -> list[str]:
    if attempt == 1:
        return paths[:3]
    index = min(attempt - 1, len(paths) - 1)
    return [paths[index]] if paths else []


def _infer_once(describer, crops: list, timeout: float, async_enabled: bool):
    if not async_enabled:
        try:
            return describer.describe(crops), None, False
        except Exception:
            return None, "inference_error", False

    future = _VLM_WORKER.submit(describer.describe, crops)
    if future is None:
        return None, "timeout", True
    try:
        return future.result(timeout=timeout), None, False
    except FutureTimeoutError:
        # Python cannot cancel a model call that has started. A retry is safe
        # only when cancellation proves the call never began.
        cancelled = future.cancel()
        return None, "timeout", not cancelled
    except Exception:
        return None, "inference_error", False


def _describe_cluster(
    describer,
    cluster_id: int,
    paths: list[str],
    *,
    timeout: float,
    max_retries: int,
    async_enabled: bool,
) -> tuple[dict, list[dict]]:
    valid_paths = [str(path) for path in paths if path]
    if not valid_paths:
        return _failed_result(0, "no_valid_body_crop"), [{
            "cluster_id": cluster_id,
            "selected_body_crop": None,
            "width": None,
            "height": None,
            "attempt": 0,
            "status": "failed",
            "failure_reason": "no_valid_body_crop",
            "elapsed_seconds": 0.0,
        }]

    diagnostics: list[dict] = []
    last_reason = "inference_error"
    attempts = 0
    for attempt in range(1, max_retries + 2):
        attempts = attempt
        attempt_paths = _attempt_paths(valid_paths, attempt)
        started = time.monotonic()
        loaded = _read_crops(attempt_paths)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            crops, metadata = loaded
        else:
            crops = list(loaded or [])
            metadata = [
                {
                    "basename": Path(attempt_paths[index]).name,
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                }
                for index, crop in enumerate(crops)
                if index < len(attempt_paths) and hasattr(crop, "shape")
            ]
        if not crops:
            result = None
            reason = "image_decode_failed"
            still_running = False
        else:
            result, reason, still_running = _infer_once(
                describer,
                crops,
                timeout,
                async_enabled,
            )
        values = None
        if result is not None:
            raw, structured = result
            values, reason = _validated_result(raw, structured)
        elapsed = round(time.monotonic() - started, 3)
        meta = metadata[0] if metadata else {}
        status = "ok" if values is not None else "failed"
        last_reason = reason or "inference_error"
        diagnostic = {
            "cluster_id": cluster_id,
            "selected_body_crop": meta.get("basename") or (
                Path(attempt_paths[0]).name if attempt_paths else None
            ),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "attempt": attempt,
            "status": status,
            "failure_reason": None if values is not None else last_reason,
            "elapsed_seconds": elapsed,
        }
        diagnostics.append(diagnostic)
        print(
            "[describe_clothing] "
            f"cluster={cluster_id} crop={diagnostic['selected_body_crop'] or '-'} "
            f"dimensions={diagnostic['width'] or 0}x{diagnostic['height'] or 0} "
            f"attempt={attempt} status={status} "
            f"failure={diagnostic['failure_reason'] or '-'} elapsed={elapsed:.3f}s"
        )
        if values is not None:
            return _successful_result(attempt, values), diagnostics
        if still_running:
            break
    return _failed_result(attempts, last_reason), diagnostics


def describe_clothing(state: dict) -> dict:
    """Describe each identity independently and return explicit status records."""
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    describer = get_clothing_describer()
    per_cluster_best = state.get("per_cluster_best_body_crops") or {}
    best_body_crops = list(state.get("best_body_crops") or [])
    async_enabled, timeout = async_vlm_config()
    retries = vlm_max_retries()
    cluster_paths = (
        {int(cid): list(paths or []) for cid, paths in per_cluster_best.items()}
        if per_cluster_best
        else {0: best_body_crops}
    )
    print(
        f"[describe_clothing] async_mode={async_enabled} "
        f"timeout_seconds={timeout:g} max_retries={retries}"
    )

    per_cluster: dict[int, dict] = {}
    diagnostics: list[dict] = []
    for cluster_id in sorted(cluster_paths):
        result, cluster_diagnostics = _describe_cluster(
            describer,
            cluster_id,
            cluster_paths[cluster_id],
            timeout=timeout,
            max_retries=retries,
            async_enabled=async_enabled,
        )
        per_cluster[cluster_id] = result
        diagnostics.extend(cluster_diagnostics)

    first_id = sorted(per_cluster)[0] if per_cluster else None
    first = per_cluster.get(first_id, _failed_result(0, "no_valid_body_crop"))
    return {
        "per_cluster_clothing": per_cluster,
        "clothing_raw": "",
        "clothing_structured": first,
        "clothing_diagnostics": diagnostics,
    }
