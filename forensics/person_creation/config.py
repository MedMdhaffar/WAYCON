"""Memory/device configuration for the person_creation pipeline.

Everything here can be overridden with environment variables so a given
machine's limits don't have to be hardcoded. Defaults are chosen to be safe
on a 4GB GPU: models are loaded one at a time (see nodes/*.py + models/*.py)
and batch sizes are kept small.
"""

import os


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


def _int_env(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(env_var: str, default: str = "cuda", force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    requested = os.environ.get(env_var, default).strip().lower()
    if requested == "cuda" and not _cuda_available():
        print(f"[config] CUDA requested via {env_var} but not available — falling back to CPU")
        return "cpu"
    return requested


# Low-memory mode: sequential model loading everywhere it's an option
# (e.g. person/face detectors run in two passes instead of concurrently).
LOW_MEMORY_MODE = _bool_env("WAYCON_LOW_MEMORY", True)           #i should test if this works 1 == slow , 0 == faster (i will test both)

# Batch sizes. Small by default so a single forward pass doesn't spike VRAM.
VLM_BATCH_SIZE = _int_env("WAYCON_VLM_BATCH_SIZE", 1)
REID_BATCH_SIZE = _int_env("WAYCON_REID_BATCH_SIZE", 4)
FACE_BATCH_SIZE = _int_env("WAYCON_FACE_BATCH_SIZE", 8)

FORCE_CPU_FOR_VLM = _bool_env("WAYCON_FORCE_CPU_VLM", False)
FORCE_CPU_FOR_REID = _bool_env("WAYCON_FORCE_CPU_REID", False)

CLEAR_CUDA_AFTER_NODE = _bool_env("WAYCON_CLEAR_CUDA_AFTER_NODE", True)

# --- Identity-guard tracking thresholds -----------------------------------
# Face embedding is the strongest identity signal; geometry is a motion cue
# only. Conservative mode prefers splitting one real person into two tracks
# over merging two real people into one.
CONSERVATIVE_TRACKING = _bool_env("WAYCON_CONSERVATIVE_TRACKING", True)

# Track continuation: reject appending a detection to a track when both sides
# have a usable face embedding and cosine similarity falls below this.
FACE_CONTINUE_MIN_SIMILARITY = _float_env("WAYCON_FACE_CONTINUE_MIN_SIMILARITY", 0.72)

# Track merging: fragmented tracks are only merged when face similarity is at
# least this high (stricter than the old 0.80).
FACE_MERGE_MIN_SIMILARITY = _float_env("WAYCON_FACE_MERGE_MIN_SIMILARITY", 0.88)

# A track needs at least this many embeddable face crops before it may merge.
MIN_FACE_CROPS_FOR_MERGE = _int_env("WAYCON_MIN_FACE_CROPS_FOR_MERGE", 2)

# A merge is rejected as ambiguous when another track's similarity is within
# this margin of the best pair.
MIN_MERGE_MARGIN = _float_env("WAYCON_MIN_MERGE_MARGIN", 0.05)

# Post-tracking mixed-track detection: a track whose internal face similarity
# drops below these limits is split at the sharpest identity change.
INTERNAL_FACE_MIN_SIMILARITY = _float_env("WAYCON_INTERNAL_FACE_MIN_SIMILARITY", 0.65)
INTERNAL_FACE_MEAN_SIMILARITY = _float_env("WAYCON_INTERNAL_FACE_MEAN_SIMILARITY", 0.75)

# Face/body association: maximum geometry cost to accept a pair, and the
# minimum cost gap to the runner-up before a pair counts as ambiguous.
ASSOCIATION_MAX_COST = _float_env("WAYCON_ASSOCIATION_MAX_COST", 0.50)
ASSOCIATION_AMBIGUITY_MARGIN = _float_env("WAYCON_ASSOCIATION_AMBIGUITY_MARGIN", 0.08)

DETECTOR_DEVICE = resolve_device("WAYCON_DETECTOR_DEVICE", "cuda")
# FaceNet is tiny and the original implementation always ran it on CPU;
# keep that as the default so it never competes with detectors/VLM/ReID for
# VRAM. Set WAYCON_FACE_DEVICE=cuda to opt into GPU for speed.
FACE_DEVICE = resolve_device("WAYCON_FACE_DEVICE", "cpu")
VLM_DEVICE = resolve_device("WAYCON_VLM_DEVICE", "cuda", force_cpu=FORCE_CPU_FOR_VLM)
REID_DEVICE = resolve_device("WAYCON_REID_DEVICE", "cuda", force_cpu=FORCE_CPU_FOR_REID)
