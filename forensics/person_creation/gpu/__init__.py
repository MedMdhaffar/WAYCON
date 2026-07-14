"""GPU-first video pipeline components (NVDEC decode, CUDA preprocessing,
in-process CUDA inference, GPU crop lifecycle).

Everything in this package is optional and feature-flagged; importing it must
never break the CPU pipeline. See docs/gpu_pipeline_architecture_analysis.md.

Feature flags (environment variables, read at node dispatch time):
    PERSON_CREATION_GPU_PIPELINE    master switch for the GPU process_video node
    PERSON_CREATION_USE_NVDEC       NVDEC decode (default on when GPU pipeline on)
    PERSON_CREATION_LOCAL_FACE      in-process CUDA face detection (default on)
    PERSON_CREATION_FINAL_ONLY_CROPS  skip writing crops that filter_quality would reject
    PERSON_CREATION_GPU_JPEG        nvJPEG GPU encode for crop writing (default on)
"""

import os


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
