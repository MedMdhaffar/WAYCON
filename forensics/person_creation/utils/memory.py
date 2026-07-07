"""Memory logging/cleanup helpers for the lazy-loading pipeline.

Every function here is safe to call when torch, CUDA, or psutil is
unavailable — heavy nodes call these unconditionally.
"""

from __future__ import annotations

import gc


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def log_memory(label: str) -> None:
    """Print CPU RSS and CUDA allocated/reserved memory when available."""
    parts: list[str] = []
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2**20
        parts.append(f"rss={rss:.0f}MB")
    except Exception:
        pass

    torch = _torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 2**20
                reserved = torch.cuda.memory_reserved() / 2**20
                parts.append(f"cuda_alloc={alloc:.0f}MB cuda_reserved={reserved:.0f}MB")
        except Exception:
            pass

    print(f"[memory] {label}: {' '.join(parts) if parts else 'no memory stats available'}")


def cleanup_memory(label: str = "") -> None:
    """gc.collect() + CUDA cache release, then log the result."""
    gc.collect()
    torch = _torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    log_memory(label or "cleanup")


_OOM_MARKERS = (
    "cuda out of memory",
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cudnn_status_not_supported",  # cuDNN raises this on allocation pressure
    "can't allocate memory",
)


def is_cuda_oom(exc: Exception) -> bool:
    torch = _torch()
    if torch is not None:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def clarify_oom(exc: Exception, context: str) -> Exception:
    """Wrap CUDA OOM errors with actionable low-memory guidance."""
    if not is_cuda_oom(exc):
        return exc
    return RuntimeError(
        f"[{context}] GPU ran out of memory: {exc}\n"
        "Low-memory fixes (set env vars before running):\n"
        "  WAYCON_LOW_MEMORY=1\n"
        "  WAYCON_FORCE_CPU_FACE=1  WAYCON_FORCE_CPU_REID=1  WAYCON_FORCE_CPU_VLM=1\n"
        "  WAYCON_FACE_BATCH_SIZE=1 WAYCON_REID_BATCH_SIZE=1 WAYCON_VLM_BATCH_SIZE=1\n"
        "  WAYCON_DETECTOR_DEVICE=cpu (last resort — slow detection)"
    )
