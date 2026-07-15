"""Clothing description node with an optional bounded VLM worker."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import cv2


_FALLBACK = {
    "top": "unknown",
    "bottom": "unknown",
    "shoes": "unknown",
    "full": "Clothing description unavailable.",
}


def _read_crops(paths: list[str]) -> list:
    crops = []
    for p in paths:
        try:
            img = cv2.imread(str(Path(p).resolve()))
            if img is not None:
                crops.append(img)
        except Exception:
            continue
    return crops


def _describe_paths(describer, paths: list[str]) -> tuple[str, dict]:
    crops = _read_crops(paths)
    return _describe_crops(describer, crops)


def _describe_crops(describer, crops: list) -> tuple[str, dict]:
    if not crops:
        return "", dict(_FALLBACK)
    # Trust the ClothingDescriber/InternVL result directly. This node does not
    # normalize colors or rewrite the returned clothing fields.
    return describer.describe(crops)


def _prepare_async_crops(
    per_cluster_best: dict,
    best_body_crops: list[str],
) -> tuple[dict[int, list], list]:
    per_cluster_crops = {
        int(cid): _read_crops(paths)
        for cid, paths in per_cluster_best.items()
    }
    best_crops = [] if per_cluster_best else _read_crops(best_body_crops)
    return per_cluster_crops, best_crops


def _describe_preloaded(
    describer,
    per_cluster_crops: dict[int, list],
    best_crops: list,
    start_event: threading.Event,
) -> dict:
    start_event.wait()
    print("[describe_clothing] worker entered", flush=True)
    try:
        if per_cluster_crops:
            per_cluster_clothing: dict[int, dict] = {}
            first_raw = ""
            first_structured = dict(_FALLBACK)
            first = True
            for cid, crops in per_cluster_crops.items():
                raw, structured = _describe_crops(describer, crops)
                per_cluster_clothing[cid] = {"raw": raw, "structured": structured}
                if first:
                    first_raw = raw
                    first_structured = structured
                    first = False
            return {
                "per_cluster_clothing": per_cluster_clothing,
                "clothing_raw": first_raw,
                "clothing_structured": first_structured,
            }

        raw, structured = _describe_crops(describer, best_crops)
        return {"clothing_raw": raw, "clothing_structured": structured}
    finally:
        print("[describe_clothing] worker exited", flush=True)


def _describe_all(describer, per_cluster_best: dict, best_body_crops: list[str]) -> dict:
    if per_cluster_best:
        per_cluster_clothing: dict[int, dict] = {}
        first_raw = ""
        first_structured = dict(_FALLBACK)
        first = True
        for raw_cid, paths in per_cluster_best.items():
            cid = int(raw_cid)
            raw, structured = _describe_paths(describer, paths)
            per_cluster_clothing[cid] = {"raw": raw, "structured": structured}
            if first:
                first_raw = raw
                first_structured = structured
                first = False
        return {
            "per_cluster_clothing": per_cluster_clothing,
            "clothing_raw": first_raw,
            "clothing_structured": first_structured,
        }

    raw, structured = _describe_paths(describer, best_body_crops)
    return {"clothing_raw": raw, "clothing_structured": structured}


def _fallback_result(per_cluster_best: dict) -> dict:
    per_cluster_clothing = {
        int(cid): {"raw": "", "structured": dict(_FALLBACK)}
        for cid in per_cluster_best
    }
    result = {
        "clothing_raw": "",
        "clothing_structured": dict(_FALLBACK),
    }
    if per_cluster_clothing:
        result["per_cluster_clothing"] = per_cluster_clothing
    return result


def async_vlm_config() -> tuple[bool, float]:
    enabled = os.getenv("PERSON_CREATION_ASYNC_VLM", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    try:
        timeout = float(os.getenv("PERSON_CREATION_VLM_TIMEOUT_SECONDS", "60"))
    except ValueError:
        timeout = 60.0
    return enabled, max(1.0, timeout)


def describe_clothing(state: dict) -> dict:
    """Run InternVL once per identity, bounded by a configurable timeout."""
    from forensics.person_creation.models.clothing_describer import get_clothing_describer

    describer = get_clothing_describer()
    per_cluster_best = state.get("per_cluster_best_body_crops") or {}
    best_body_crops = state.get("best_body_crops", [])
    async_enabled, timeout = async_vlm_config()
    print(
        f"[describe_clothing] async_mode={async_enabled} "
        f"timeout_seconds={timeout:g}"
    )

    if not async_enabled:
        try:
            return _describe_all(describer, per_cluster_best, best_body_crops)
        except Exception as exc:
            print(f"[describe_clothing] VLM failed; using fallback ({type(exc).__name__})")
            return _fallback_result(per_cluster_best)

    # Decode selected paths before submission. A timed-out worker may finish
    # inference later, but it never retains or reopens cleanup-sensitive paths.
    per_cluster_crops, best_crops = _prepare_async_crops(
        per_cluster_best,
        best_body_crops,
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="person-creation-vlm")
    worker_start = threading.Event()
    future = executor.submit(
        _describe_preloaded,
        describer,
        per_cluster_crops,
        best_crops,
        worker_start,
    )
    try:
        print("[describe_clothing] VLM task submitted", flush=True)
    finally:
        worker_start.set()
    try:
        result = future.result(timeout=timeout)
        print("[describe_clothing] async VLM completed")
        return result
    except FutureTimeoutError:
        print("[describe_clothing] VLM timeout reached", flush=True)
        print("[describe_clothing] future cancel requested", flush=True)
        cancelled = future.cancel()
        print(
            f"[describe_clothing] future cancel result={str(cancelled).lower()}",
            flush=True,
        )
        print(f"[describe_clothing] async VLM timed out after {timeout:g}s; using fallback")
        return _fallback_result(per_cluster_best)
    except Exception as exc:
        print(f"[describe_clothing] async VLM failed; using fallback ({type(exc).__name__})")
        return _fallback_result(per_cluster_best)
    finally:
        print("[describe_clothing] executor shutdown entered", flush=True)
        executor.shutdown(wait=False, cancel_futures=True)
        print("[describe_clothing] executor shutdown completed", flush=True)
