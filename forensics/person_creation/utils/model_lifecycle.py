"""Small shared helpers for releasing torch-backed model objects.

Each model wrapper in `forensics/person_creation/models/` keeps its own
`unload()` method (the underlying objects differ: ultralytics YOLO, a plain
nn.Module, an HF AutoModel, a torchreid FeatureExtractor). This module only
holds the bit of logic that's identical across all of them.
"""


def move_to_cpu(obj) -> None:
    """Best-effort `.to("cpu")` — swallow errors from objects that don't
    support it (e.g. some device_map-loaded HF models)."""
    if obj is None:
        return
    try:
        obj.to("cpu")
    except Exception:
        pass


def delete_model(obj) -> None:
    """Drop a reference so it becomes eligible for GC / CUDA cache reclaim.
    Caller is responsible for setting its own attribute to None afterward."""
    try:
        del obj
    except Exception:
        pass
