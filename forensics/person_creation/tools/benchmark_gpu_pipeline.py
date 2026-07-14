"""Phase 7 — benchmark the five pipeline implementations on one video.

Modes:
    A  existing CPU decode + NumPy + CUDA inference (production CPU node)
    B  NVDEC decode but frames copied back to CPU, existing interfaces
       (demonstrates why NVDEC alone is NOT the optimization)
    C  NVDEC + direct CUDA tensors + person detect_cuda; face still HTTP
    D  C + in-process CUDA face detection
    E  D + final-only crop persistence (GPU quality gate) + nvJPEG

Per mode: cold run, warm run, then one profiled warm run for the stage
breakdown and per-frame latency percentiles (frame latency := decode +
convert/upload + person + face stage time of that frame; crop handling is
reported separately since crop count varies per frame).

Measurement notes:
 - cold/warm wall times are measured with profiling DISABLED (low-overhead
   benchmark mode); the profiled run is diagnostic and reported separately;
 - GPU stage timing inside the profiled run uses the existing CUDA-event
   helper rather than global synchronize (PERSON_CREATION_PROFILE_CUDA_SYNC
   stays off);
 - models are loaded and warmed once before any mode so the mode comparison is
   inference-only; the one-time startup cost is reported separately;
 - resource sampling (CPU %, GPU util, VRAM) via the existing ResourceSampler.

Run:
    python -m forensics.person_creation.tools.benchmark_gpu_pipeline \
        [video] [--every 5] [--out /tmp/waycon_bench]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_VIDEO = "forensics/person_creation/videos/4.mp4"

FLAG_NAMES = (
    "PERSON_CREATION_GPU_PIPELINE",
    "PERSON_CREATION_USE_NVDEC",
    "PERSON_CREATION_LOCAL_FACE",
    "PERSON_CREATION_FINAL_ONLY_CROPS",
    "PERSON_CREATION_GPU_JPEG",
)

MODES = {
    "A": {"label": "CPU decode + NumPy + CUDA inference (current)", "flags": {}},
    "B": {"label": "NVDEC → copy back to CPU → existing interfaces", "flags": {}},
    "C": {
        "label": "NVDEC + CUDA tensor person; face via HTTP",
        "flags": {
            "PERSON_CREATION_GPU_PIPELINE": "1",
            "PERSON_CREATION_USE_NVDEC": "1",
            "PERSON_CREATION_LOCAL_FACE": "0",
            "PERSON_CREATION_FINAL_ONLY_CROPS": "0",
            "PERSON_CREATION_GPU_JPEG": "1",
        },
    },
    "D": {
        "label": "NVDEC + CUDA person + in-process CUDA face",
        "flags": {
            "PERSON_CREATION_GPU_PIPELINE": "1",
            "PERSON_CREATION_USE_NVDEC": "1",
            "PERSON_CREATION_LOCAL_FACE": "1",
            "PERSON_CREATION_FINAL_ONLY_CROPS": "0",
            "PERSON_CREATION_GPU_JPEG": "1",
        },
    },
    "E": {
        "label": "full GPU-first + final-only crops + nvJPEG",
        "flags": {
            "PERSON_CREATION_GPU_PIPELINE": "1",
            "PERSON_CREATION_USE_NVDEC": "1",
            "PERSON_CREATION_LOCAL_FACE": "1",
            "PERSON_CREATION_FINAL_ONLY_CROPS": "1",
            "PERSON_CREATION_GPU_JPEG": "1",
        },
    },
}

_FRAME_LATENCY_STAGES = (
    "frame.decode",
    "frame.nv12_to_rgb",
    "frame.h2d_upload",
    "frame.person_detection",
    "frame.face_detection",
)


def _set_flags(flags: dict[str, str]) -> None:
    for name in FLAG_NAMES:
        os.environ.pop(name, None)
    os.environ.update(flags)


def _mode_b_loop(video: str, every_n: int, out_dir: Path) -> dict:
    """NVDEC decode, then immediately copy each selected frame to CPU NumPy and
    feed the *existing* CPU interfaces. Intentionally naive."""
    import cv2

    from forensics.face_engine.client import FaceEngineClient
    from forensics.person_creation.gpu.decoder import NvdecDecoder
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.nodes.process_video import _crop, _sharpness

    person_det = get_person_detector()
    face_det = FaceEngineClient()
    body_dir = out_dir / "_staging" / "body_crops"
    face_dir = out_dir / "_staging" / "face_crops"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    from forensics.person_creation.utils.profiling import profile_measure

    decoder = NvdecDecoder()
    decoder.open(video)
    body_crops, face_crops = [], []
    frame_idx = 0
    stem = Path(video).stem
    while True:
        with profile_measure("frame.decode"):
            frame = decoder.read()
        if frame is None:
            break
        if frame_idx % every_n != 0:
            frame_idx += 1
            continue
        with profile_measure("frame.h2d_upload"):  # here it is actually D2H
            frame_bgr = frame.to_bgr_numpy()  # the defeating device-to-host copy
        with profile_measure("frame.person_detection", synchronize_cuda=True):
            persons = person_det.detect(frame_bgr)
        with profile_measure("frame.face_detection"):
            faces = face_det.detect(frame_bgr)
        for crop_type, detections, destination, target in (
            ("body", persons, body_dir, body_crops),
            ("face", faces, face_dir, face_crops),
        ):
            for det_idx, det in enumerate(detections):
                crop = _crop(frame_bgr, det["bbox"])
                if crop.size == 0:
                    continue
                sharp = _sharpness(crop)
                tag = "b" if crop_type == "body" else "f"
                prefix = f"{stem}_f{frame_idx:06d}" if crop_type == "body" else f"{stem}_face_f{frame_idx:06d}"
                path = str(destination / f"{prefix}_{tag}{det_idx:02d}.jpg")
                cv2.imwrite(path, crop)
                target.append({
                    "path": path, "frame_idx": frame_idx, "video": video,
                    "bbox": det["bbox"], "sharpness": sharp,
                })
        frame_idx += 1
    decoder.close()
    return {"body_crops": body_crops, "face_crops": face_crops}


def _run_mode(mode: str, video: str, every_n: int, out_dir: Path) -> dict:
    _set_flags(MODES[mode]["flags"])
    state = {"video_paths": [video], "output_dir": str(out_dir), "process_every_n": every_n}
    if mode == "A":
        from forensics.person_creation.nodes.process_video import process_video

        return process_video(state)
    if mode == "B":
        return _mode_b_loop(video, every_n, out_dir)
    from forensics.person_creation.nodes.process_video_gpu import process_video_gpu

    return process_video_gpu(state)


def _frame_latencies(records: list[dict]) -> list[float]:
    per_stage: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["name"] in _FRAME_LATENCY_STAGES:
            per_stage[record["name"]].append(float(record["elapsed_seconds"]))
    # decode fires for every frame; per-selected-frame stages define the count.
    counts = [len(v) for k, v in per_stage.items() if k != "frame.decode"]
    if not counts:
        return []
    n = min(counts)
    latencies = []
    for i in range(n):
        total = sum(per_stage[k][i] for k in per_stage if k != "frame.decode" and i < len(per_stage[k]))
        latencies.append(total)
    return latencies


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _stage_totals(records: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        if record["name"].startswith(("frame.", "video.", "model.warmup")):
            totals[record["name"]] += float(record["elapsed_seconds"])
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def benchmark(video: str, every_n: int, base_out: Path, modes: list[str] | None = None) -> dict:
    import torch

    from forensics.face_engine.client import FaceEngineClient, FaceEngineConnectionError
    from forensics.person_creation.gpu.face_local import load_local_face_detector
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.nodes.load_models import _YOLO_MODEL_PATH
    from forensics.person_creation.utils.profiling import (
        PipelineProfiler,
        ProfilingConfig,
        ResourceSampler,
        use_profiler,
    )

    try:
        FaceEngineClient().ensure_healthy()
    except FaceEngineConnectionError:
        raise SystemExit("Face Engine must be running for modes A/B/C (python -m forensics.face_engine.app)")

    print("=== one-time startup (shared by all modes) ===")
    t0 = time.perf_counter()
    detector = get_person_detector()
    if detector._model is None:
        detector.load(model_path=_YOLO_MODEL_PATH, device="auto")
    load_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    detector.warmup()
    face_local = load_local_face_detector()
    face_local.warmup()
    warmup_t = time.perf_counter() - t0
    print(f"model load {load_t:.2f}s, warm-up {warmup_t:.2f}s")

    results: dict[str, dict] = {
        "_meta": {
            "video": video,
            "every_n": every_n,
            "model_load_seconds": load_t,
            "model_warmup_seconds": warmup_t,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        }
    }

    for mode, spec in MODES.items():
        if modes and mode not in modes:
            continue
        print(f"\n=== mode {mode}: {spec['label']} ===")
        runs = {}
        for run_name in ("cold", "warm"):
            out_dir = base_out / f"mode_{mode}_{run_name}"
            shutil.rmtree(out_dir, ignore_errors=True)
            torch.cuda.reset_peak_memory_stats()
            sampler = ResourceSampler(interval_seconds=0.25, enabled=True)
            sampler.start()
            t0 = time.perf_counter()
            out = _run_mode(mode, video, every_n, out_dir)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            sampler.stop()
            samples = sampler.samples
            gpu_util = [s["gpu_utilization_percent"] for s in samples if "gpu_utilization_percent" in s]
            cpu_util = [s["process_cpu_percent"] for s in samples if "process_cpu_percent" in s]
            vram_total = [s["gpu_total_vram_used_mb"] for s in samples if "gpu_total_vram_used_mb" in s]
            runs[run_name] = {
                "wall_seconds": elapsed,
                "body_crops": len(out["body_crops"]),
                "face_crops": len(out["face_crops"]),
                "vram_peak_torch_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
                "vram_peak_system_mb": max(vram_total) if vram_total else None,
                "gpu_util_mean_percent": statistics.fmean(gpu_util) if gpu_util else None,
                "gpu_util_max_percent": max(gpu_util) if gpu_util else None,
                "cpu_util_mean_percent": statistics.fmean(cpu_util) if cpu_util else None,
            }
            print(
                f"  {run_name}: {elapsed:.3f}s  body={runs[run_name]['body_crops']} "
                f"face={runs[run_name]['face_crops']} "
                f"vram_peak(torch)={runs[run_name]['vram_peak_torch_mb']:.0f}MB "
                f"gpu_util≈{runs[run_name]['gpu_util_mean_percent']} "
                f"cpu≈{runs[run_name]['cpu_util_mean_percent']}"
            )

        # diagnostic profiled run (warm)
        out_dir = base_out / f"mode_{mode}_profiled"
        shutil.rmtree(out_dir, ignore_errors=True)
        profiler = PipelineProfiler(config=ProfilingConfig(enabled=True))
        with use_profiler(profiler):
            t0 = time.perf_counter()
            _run_mode(mode, video, every_n, out_dir)
            profiled_elapsed = time.perf_counter() - t0
        records = profiler.records()
        latencies = _frame_latencies(records)
        selected = len(latencies)
        runs["profiled"] = {
            "wall_seconds": profiled_elapsed,
            "selected_frames": selected,
            "frame_latency_mean_ms": statistics.fmean(latencies) * 1000 if latencies else None,
            "frame_latency_median_ms": _percentile(latencies, 0.5) * 1000,
            "frame_latency_p90_ms": _percentile(latencies, 0.9) * 1000,
            "frame_latency_p95_ms": _percentile(latencies, 0.95) * 1000,
            "throughput_selected_fps": selected / profiled_elapsed if profiled_elapsed else None,
            "stage_totals_seconds": _stage_totals(records),
        }
        mean_ms = runs["profiled"]["frame_latency_mean_ms"]
        print(
            f"  profiled: {profiled_elapsed:.3f}s  frame latency "
            f"mean={mean_ms:.1f}ms " if mean_ms is not None else
            f"  profiled: {profiled_elapsed:.3f}s  (no frame-level records)",
            end="",
        )
        if mean_ms is not None:
            print(
                f"p95={runs['profiled']['frame_latency_p95_ms']:.1f}ms "
                f"({runs['profiled']['throughput_selected_fps']:.1f} selected fps)"
            )
        else:
            print()
        results[mode] = {"label": spec["label"], **runs}

    _set_flags({})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--out", default="/tmp/waycon_bench")
    parser.add_argument("--json", default="/tmp/waycon_bench/results.json")
    parser.add_argument("--modes", default=None, help="comma-separated subset, e.g. A,D,E")
    args = parser.parse_args()

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)
    modes = args.modes.upper().split(",") if args.modes else None
    results = benchmark(args.video, args.every, base_out, modes=modes)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults written to {args.json}")

    print(f"\n{'mode':4s} {'cold s':>8s} {'warm s':>8s} {'lat p50 ms':>10s} {'lat p95 ms':>10s} "
          f"{'sel fps':>8s} {'VRAM MB':>8s}")
    for mode in MODES:
        if mode not in results:
            continue
        r = results[mode]
        print(
            f"{mode:4s} {r['cold']['wall_seconds']:8.2f} {r['warm']['wall_seconds']:8.2f} "
            f"{r['profiled']['frame_latency_median_ms']:10.1f} "
            f"{r['profiled']['frame_latency_p95_ms']:10.1f} "
            f"{r['profiled']['throughput_selected_fps']:8.1f} "
            f"{r['warm']['vram_peak_torch_mb']:8.0f}"
        )


if __name__ == "__main__":
    main()
