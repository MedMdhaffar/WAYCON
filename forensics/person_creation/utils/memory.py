"""Memory diagnostics + cleanup shared by every heavy-model node.

Nodes load a model, use it, then call cleanup_memory() in a `finally` block
so GPU/CPU memory is returned before the next node loads its own model.
"""

import gc


def _torch_cuda_stats() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def log_memory(label: str) -> None:
    stats = _torch_cuda_stats()
    if stats is None:
        print(f"[memory] {label}: cuda unavailable")
        return
    print(
        f"[memory] {label}: allocated={stats['allocated_mb']:.1f}MB "
        f"reserved={stats['reserved_mb']:.1f}MB "
        f"max_allocated={stats['max_allocated_mb']:.1f}MB"
    )


def clarify_oom(exc: Exception, context: str) -> Exception:
    """Wrap a possible CUDA OOM in a message that explains how to recover,
    instead of letting the process hang/crash with an opaque torch trace."""
    msg = str(exc)
    if "out of memory" in msg.lower() or "cuda error" in msg.lower():
        return RuntimeError(
            f"[{context}] CUDA appears to be out of memory: {exc}. "
            "Try WAYCON_LOW_MEMORY=1 (default), a smaller *_BATCH_SIZE env var, "
            "or WAYCON_FORCE_CPU_VLM=1 / WAYCON_FORCE_CPU_REID=1 to fall back to CPU."
        )
    return exc


def cleanup_memory(label: str = "") -> None:
    gc.collect()   #use garbage collector 
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    if label:
        log_memory(f"cleanup after {label}")
