"""Phase 2 prototype — validate the NVDEC decoder in isolation.

Checks (against the CPU cv2 path on the same video):
  1. frame count, dimensions, fps metadata;
  2. color correctness of NV12→RGB (BT.601 vs BT.709, mean abs diff);
  3. decode throughput NVDEC vs cv2.VideoCapture.

Run:
    python -m forensics.person_creation.tools.gpu_decoder_prototype \
        [video_path] [--frames N]
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from forensics.person_creation.gpu.decoder import NvdecDecoder, OpenCVDecoder
from forensics.person_creation.gpu.preprocess import nv12_to_rgb

DEFAULT_VIDEO = "forensics/person_creation/videos/4.mp4"


def compare_colors(video: str, sample_indices: list[int]) -> None:
    print("\n=== color correctness (vs cv2.VideoCapture BGR) ===")
    cpu_frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video)
    idx = 0
    while len(cpu_frames) < len(sample_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_indices:
            cpu_frames[idx] = frame
        idx += 1
    cap.release()

    dec = NvdecDecoder()
    dec.open(video)
    results = {"bt601": [], "bt709": []}
    while True:
        frame = dec.read()
        if frame is None:
            break
        if frame.frame_index not in cpu_frames:
            continue
        ref_rgb = cv2.cvtColor(cpu_frames[frame.frame_index], cv2.COLOR_BGR2RGB).astype(np.float32)
        for matrix in ("bt601", "bt709"):
            got = nv12_to_rgb(frame.tensor, frame.height, frame.width, matrix)
            got_np = got.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            diff = np.abs(got_np - ref_rgb)
            results[matrix].append((frame.frame_index, float(diff.mean()), float(np.percentile(diff, 99))))
        if all(len(v) >= len(sample_indices) for v in results.values()):
            break
    dec.close()

    for matrix, rows in results.items():
        for frame_index, mean_diff, p99 in rows:
            print(f"  {matrix} frame {frame_index:4d}: mean abs diff={mean_diff:6.3f}  p99={p99:6.1f}")
        avg = sum(r[1] for r in rows) / max(len(rows), 1)
        print(f"  {matrix} average mean-abs-diff: {avg:.3f}")
    best = min(results, key=lambda m: sum(r[1] for r in results[m]))
    print(f"  selected default for this stream height: "
          f"{'bt709' if dec.height >= 720 else 'bt601'}; empirically best: {best}")


def bench_decode(video: str, limit: int | None) -> None:
    print("\n=== decode throughput ===")

    dec = OpenCVDecoder()
    dec.open(video)
    t0 = time.perf_counter()
    n_cpu = 0
    while (f := dec.read()) is not None:
        n_cpu += 1
        if limit and n_cpu >= limit:
            break
    cpu_s = time.perf_counter() - t0
    dec.close()
    print(f"  cv2.VideoCapture : {n_cpu} frames in {cpu_s:.3f}s ({n_cpu / cpu_s:.1f} fps)")

    dec = NvdecDecoder()
    dec.open(video)
    t0 = time.perf_counter()
    n_gpu = 0
    while (f := dec.read()) is not None:
        n_gpu += 1
        if limit and n_gpu >= limit:
            break
    torch.cuda.synchronize()
    nvdec_s = time.perf_counter() - t0
    dec.close()
    print(f"  NVDEC (nv12 only): {n_gpu} frames in {nvdec_s:.3f}s ({n_gpu / nvdec_s:.1f} fps)")

    dec = NvdecDecoder()
    dec.open(video)
    t0 = time.perf_counter()
    n_rgb = 0
    while (f := dec.read()) is not None:
        _ = f.to_rgb_chw()
        n_rgb += 1
        if limit and n_rgb >= limit:
            break
    torch.cuda.synchronize()
    rgb_s = time.perf_counter() - t0
    dec.close()
    print(f"  NVDEC + nv12→rgb : {n_rgb} frames in {rgb_s:.3f}s ({n_rgb / rgb_s:.1f} fps)")

    if n_cpu != n_gpu:
        print(f"  WARNING: frame count mismatch cpu={n_cpu} nvdec={n_gpu}")


def check_metadata(video: str) -> None:
    print("=== metadata ===")
    for cls in (OpenCVDecoder, NvdecDecoder):
        dec = cls()
        dec.open(video)
        first = dec.read()
        print(
            f"  {dec.backend:12s}: {dec.width}x{dec.height} fps={dec.fps:.3f} "
            f"frames={dec.frame_count} first_frame(fmt={first.pixel_format}, "
            f"device={first.device}, ts={first.timestamp_seconds:.4f}s, "
            f"tensor={tuple(first.tensor.shape)} {first.tensor.dtype} on {first.tensor.device})"
        )
        dec.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    parser.add_argument("--frames", type=int, default=None, help="limit decoded frames")
    args = parser.parse_args()

    check_metadata(args.video)
    compare_colors(args.video, sample_indices=[0, 100, 250, 400])
    bench_decode(args.video, args.frames)


if __name__ == "__main__":
    main()
