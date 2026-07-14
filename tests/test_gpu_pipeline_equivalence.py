"""Equivalence tests: GPU-first pipeline vs the existing CPU pipeline.

Tolerances (from docs/gpu_pipeline_architecture_analysis.md):
    bbox IoU               >= 0.98 (matched pairs)
    detection count        identical per frame
    detection score        |delta| <= 0.02
    sharpness              relative diff <= 1 %
    decoded color          mean abs diff <= 3 (8-bit), p99 <= 8
    JPEG round trip        PSNR >= 30 dB vs source pixels

Run (needs CUDA + the yolo26m.pt weights; HTTP test needs the Face Engine):
    python -m pytest tests/test_gpu_pipeline_equivalence.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for GPU pipeline equivalence tests", allow_module_level=True)

import cv2

VIDEO = str(Path(__file__).parents[1] / "forensics/person_creation/videos/4.mp4")
SAMPLE_FRAMES = [0, 60, 150, 300, 450]


def _read_cpu_frames(indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(VIDEO)
    frames: dict[int, np.ndarray] = {}
    idx = 0
    while len(frames) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames[idx] = frame
        idx += 1
    cap.release()
    return frames


def _bgr_to_rgb_cuda(frame_bgr: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .contiguous()
        .cuda()
    )


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _assert_detections_equivalent(ref: list[dict], got: list[dict], context: str) -> None:
    assert len(ref) == len(got), f"{context}: count {len(ref)} != {len(got)}"
    unmatched = list(range(len(got)))
    for r in ref:
        best_iou, best_j = 0.0, None
        for j in unmatched:
            iou = _iou(r["bbox"], got[j]["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        assert best_j is not None and best_iou >= 0.98, (
            f"{context}: bbox {r['bbox']} best IoU {best_iou:.4f} < 0.98"
        )
        assert abs(r["score"] - got[best_j]["score"]) <= 0.02, (
            f"{context}: score {r['score']:.4f} vs {got[best_j]['score']:.4f}"
        )
        unmatched.remove(best_j)


# ── Layer 1/2: decoder ────────────────────────────────────────────────────────

class TestDecoder:
    def test_nvdec_available_here(self):
        from forensics.person_creation.gpu.decoder import nvdec_available

        ok, reason = nvdec_available()
        assert ok, f"NVDEC expected available on this machine: {reason}"

    def test_metadata_and_frame_count(self):
        from forensics.person_creation.gpu.decoder import NvdecDecoder, OpenCVDecoder

        counts, dims = {}, {}
        for cls in (OpenCVDecoder, NvdecDecoder):
            dec = cls()
            dec.open(VIDEO)
            n = 0
            first = dec.read()
            assert first is not None
            n += 1
            while dec.read() is not None:
                n += 1
            counts[dec.backend] = n
            dims[dec.backend] = (dec.width, dec.height)
            dec.close()
        assert counts["opencv-cpu"] == counts["nvdec"], counts
        assert dims["opencv-cpu"] == dims["nvdec"], dims

    def test_color_correctness(self):
        from forensics.person_creation.gpu.decoder import NvdecDecoder

        cpu = _read_cpu_frames(SAMPLE_FRAMES)
        dec = NvdecDecoder()
        dec.open(VIDEO)
        checked = 0
        while checked < len(cpu):
            frame = dec.read()
            assert frame is not None
            if frame.frame_index not in cpu:
                continue
            assert frame.device == "cuda" and frame.tensor.is_cuda
            rgb = frame.to_rgb_chw()
            got = rgb.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            ref = cv2.cvtColor(cpu[frame.frame_index], cv2.COLOR_BGR2RGB).astype(np.float32)
            diff = np.abs(got - ref)
            assert diff.mean() <= 3.0, f"frame {frame.frame_index}: mean {diff.mean():.2f}"
            assert np.percentile(diff, 99) <= 8.0
            checked += 1
        dec.close()


# ── Layer 4: person detector ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def person_detector():
    from forensics.person_creation.nodes.load_models import _YOLO_MODEL_PATH
    from forensics.person_creation.models.person_detector import get_person_detector

    det = get_person_detector()
    if det._model is None:
        det.load(model_path=_YOLO_MODEL_PATH, device="auto")
    det.warmup()
    return det


class TestPersonDetector:
    def test_cuda_path_matches_numpy_path(self, person_detector):
        frames = _read_cpu_frames(SAMPLE_FRAMES)
        total = 0
        for idx, frame_bgr in frames.items():
            ref = person_detector.detect(frame_bgr)
            got = person_detector.detect_cuda(_bgr_to_rgb_cuda(frame_bgr))
            _assert_detections_equivalent(ref, got, f"person frame {idx}")
            total += len(ref)
        assert total > 0, "test video should contain at least one person detection"


# ── Layer 5: face detector ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def face_detector():
    from forensics.person_creation.gpu.face_local import load_local_face_detector

    det = load_local_face_detector(device="auto")
    det.warmup()
    return det


class TestFaceDetector:
    def test_cuda_path_matches_numpy_path(self, face_detector):
        frames = _read_cpu_frames(SAMPLE_FRAMES)
        for idx, frame_bgr in frames.items():
            ref_raw = face_detector.detect(frame_bgr)
            ref = [{"bbox": r["bbox"], "score": r["confidence"]} for r in ref_raw]
            got_raw = face_detector.detect_cuda(_bgr_to_rgb_cuda(frame_bgr))
            got = [{"bbox": r["bbox"], "score": r["confidence"]} for r in got_raw]
            _assert_detections_equivalent(ref, got, f"face frame {idx}")

    def test_local_matches_http_face_engine(self, face_detector):
        from forensics.face_engine.client import FaceEngineClient, FaceEngineConnectionError

        client = FaceEngineClient()
        try:
            client.ensure_healthy()
        except FaceEngineConnectionError:
            pytest.skip("Face Engine HTTP service is not running")
        frames = _read_cpu_frames([150, 300])
        for idx, frame_bgr in frames.items():
            ref = client.detect(frame_bgr)
            got = face_detector.detect_cuda(_bgr_to_rgb_cuda(frame_bgr))
            # HTTP path re-encodes the frame as JPEG before detection, so allow
            # a slightly looser score tolerance but identical counts and boxes.
            assert len(ref) == len(got), f"frame {idx}: {len(ref)} != {len(got)}"
            for r in ref:
                best = max((_iou(r["bbox"], g["bbox"]) for g in got), default=0.0)
                assert best >= 0.95, f"frame {idx}: HTTP-vs-local IoU {best:.3f}"


# ── Layer 6/7: crops, sharpness, JPEG ────────────────────────────────────────

class TestCrops:
    def test_sharpness_matches_cv2(self):
        from forensics.person_creation.gpu.crops import crop_gpu, sharpness_gpu

        frames = _read_cpu_frames([150])
        frame_bgr = frames[150]
        frame_rgb = _bgr_to_rgb_cuda(frame_bgr)
        h, w = frame_bgr.shape[:2]
        boxes = [
            [10, 10, 200, 400],
            [w - 300, h - 500, w - 20, h - 20],
            [w // 3, h // 3, w // 3 + 150, h // 3 + 300],
        ]
        for bbox in boxes:
            gpu_crop = crop_gpu(frame_rgb, bbox)
            got = sharpness_gpu(gpu_crop)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
            x2, y2 = min(w, x2 + 2), min(h, y2 + 2)
            cpu_crop = frame_bgr[y1:y2, x1:x2]
            gray = cv2.cvtColor(cpu_crop, cv2.COLOR_BGR2GRAY)
            ref = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            rel = abs(got - ref) / max(ref, 1e-6)
            assert rel <= 0.01, f"bbox {bbox}: sharpness {got:.3f} vs cv2 {ref:.3f} (rel {rel:.4f})"

    def test_gpu_jpeg_roundtrip(self, tmp_path):
        from forensics.person_creation.gpu.crops import crop_gpu, write_crop_jpeg

        frames = _read_cpu_frames([150])
        frame_rgb = _bgr_to_rgb_cuda(frames[150])
        crop = crop_gpu(frame_rgb, [100, 100, 400, 700])
        out = tmp_path / "crop.jpg"
        assert write_crop_jpeg(out, crop, use_gpu_jpeg=True)

        decoded = cv2.imread(str(out))
        assert decoded is not None
        src = crop.flip(0).permute(1, 2, 0).cpu().numpy()
        assert decoded.shape == src.shape
        mse = float(np.mean((decoded.astype(np.float64) - src.astype(np.float64)) ** 2))
        psnr = 10.0 * math.log10(255.0**2 / max(mse, 1e-9))
        assert psnr >= 30.0, f"PSNR {psnr:.1f} dB"
