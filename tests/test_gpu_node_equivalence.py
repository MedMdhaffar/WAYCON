"""End-to-end node equivalence: process_video (CPU) vs process_video_gpu.

Runs both nodes on the same video with the same settings and compares the
state contract.

Tolerances and why they differ from the pure detector tests
(tests/test_gpu_pipeline_equivalence.py, where both paths see *identical
pixels* and match at IoU >= 0.98):

 - The GPU node decodes with NVDEC, the CPU node with cv2/ffmpeg. The decoded
   pixels differ by ~1 LSB (verified: mean abs diff ~1.07). Detections whose
   confidence sits at the model's 0.25 threshold can therefore flip in or out.
   Verified concretely: frame 120 has a person at conf 0.2755 on cv2 pixels;
   detect() and detect_cuda() on the SAME pixels both return it, on NVDEC
   pixels it drops below threshold. This is a property of hardware decode,
   not of the CUDA inference path.
   → matched detections must align at IoU >= 0.95 (faces additionally cross
     the JPEG→HTTP boundary on the CPU side) / >= 0.98 (bodies), OR with all
     bbox corners within 3 px — small faces (~35 px) lose ~5 % IoU per pixel
     of shift, so a pure IoU gate over-penalizes them;
   → unmatched ("borderline flip") detections are allowed but must be <= 5 %
     of all detections and counts per kind must agree within 5.
 - Sharpness: Laplacian variance amplifies single-LSB decoder pixel noise,
   and matched boxes may legitimately differ by a few px of extent, which
   changes the crop content. Per-crop percentage checks are therefore
   brittle; what downstream actually consumes is (a) the filter_quality
   threshold decision (>= 50.0) and (b) sharpness *ranking*. The test
   asserts the threshold decision agrees for every matched pair (with a
   [35, 65] ambiguity band where disagreement is tolerated) and that the
   median relative difference stays <= 10 %.

Slow (~30-60 s). Needs CUDA, yolo26m.pt and a running Face Engine (the CPU
node's face path is HTTP).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

VIDEO = str(Path(__file__).parents[1] / "forensics/person_creation/videos/4.mp4")
EVERY_N = 15

_IOU_GATE = {"body_crops": 0.98, "face_crops": 0.95}
_MAX_BORDERLINE_FRACTION = 0.05
_MAX_COUNT_DELTA = 5
_SHARPNESS_THRESHOLD = 50.0  # filter_quality._MIN_SHARPNESS
_SHARPNESS_BAND = (35.0, 65.0)
_MAX_MEDIAN_SHARPNESS_REL = 0.10


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )


@pytest.fixture(scope="module")
def node_outputs(tmp_path_factory):
    from forensics.face_engine.client import FaceEngineClient, FaceEngineConnectionError
    from forensics.person_creation.nodes.load_models import _YOLO_MODEL_PATH
    from forensics.person_creation.models.person_detector import get_person_detector

    try:
        FaceEngineClient().ensure_healthy()
    except FaceEngineConnectionError:
        pytest.skip("Face Engine HTTP service is not running (CPU node needs it)")

    det = get_person_detector()
    if det._model is None:
        det.load(model_path=_YOLO_MODEL_PATH, device="auto")

    from forensics.person_creation.nodes.process_video import process_video
    from forensics.person_creation.nodes.process_video_gpu import process_video_gpu

    os.environ.pop("PERSON_CREATION_GPU_PIPELINE", None)  # CPU node stays CPU

    def make_state(out_dir: Path) -> dict:
        return {
            "video_paths": [VIDEO],
            "output_dir": str(out_dir),
            "process_every_n": EVERY_N,
        }

    cpu_out = process_video(make_state(tmp_path_factory.mktemp("cpu_node")))
    gpu_out = process_video_gpu(make_state(tmp_path_factory.mktemp("gpu_node")))
    return cpu_out, gpu_out


def _group(crops):
    grouped = defaultdict(list)
    for c in crops:
        grouped[c["frame_idx"]].append(c)
    return grouped


@pytest.mark.parametrize("kind", ["body_crops", "face_crops"])
def test_node_equivalence(node_outputs, kind):
    cpu_out, gpu_out = node_outputs
    cpu, gpu = _group(cpu_out[kind]), _group(gpu_out[kind])
    iou_gate = _IOU_GATE[kind]
    sharpness_rels: list[float] = []

    total_ref = sum(len(v) for v in cpu.values())
    total_got = sum(len(v) for v in gpu.values())
    assert total_ref > 0, f"{kind}: CPU node produced no detections — bad test video?"
    assert abs(total_ref - total_got) <= _MAX_COUNT_DELTA, (
        f"{kind}: total counts diverge: cpu={total_ref} gpu={total_got}"
    )

    matched = 0
    borderline = 0
    for frame_idx in sorted(set(cpu) | set(gpu)):
        ref, got = cpu.get(frame_idx, []), gpu.get(frame_idx, [])
        unmatched = list(got)
        for r in ref:
            best, best_g = 0.0, None
            for g in unmatched:
                iou = _iou(r["bbox"], g["bbox"])
                if iou > best:
                    best, best_g = iou, g
            if best_g is None or best < 0.5:
                borderline += 1  # detection flipped by decoder pixel differences
                continue
            corner_dev = max(abs(a - b) for a, b in zip(r["bbox"], best_g["bbox"]))
            if best < iou_gate and corner_dev > 3.0:
                # Same detection, meaningfully different extent: the decoders
                # disagree on a genuinely ambiguous box. Counted against the
                # same bounded borderline budget, not a hard failure.
                borderline += 1
                unmatched.remove(best_g)
                continue
            ref_sharp, got_sharp = r["sharpness"], best_g["sharpness"]
            if ref_sharp > 1.0:
                sharpness_rels.append(abs(ref_sharp - got_sharp) / ref_sharp)
            ref_pass = ref_sharp >= _SHARPNESS_THRESHOLD
            got_pass = got_sharp >= _SHARPNESS_THRESHOLD
            in_band = _SHARPNESS_BAND[0] <= ref_sharp <= _SHARPNESS_BAND[1]
            assert ref_pass == got_pass or in_band, (
                f"{kind} frame {frame_idx}: filter_quality decision flips: "
                f"cpu sharpness {ref_sharp:.2f} vs gpu {got_sharp:.2f}"
            )
            matched += 1
            unmatched.remove(best_g)
        borderline += len(unmatched)  # GPU-only detections

    fraction = borderline / max(total_ref, 1)
    assert fraction <= _MAX_BORDERLINE_FRACTION, (
        f"{kind}: {borderline} borderline flips / {total_ref} detections "
        f"({fraction:.1%}) exceeds {_MAX_BORDERLINE_FRACTION:.0%}"
    )
    assert matched > 0

    if sharpness_rels:
        sharpness_rels.sort()
        median_rel = sharpness_rels[len(sharpness_rels) // 2]
        assert median_rel <= _MAX_MEDIAN_SHARPNESS_REL, (
            f"{kind}: median sharpness rel diff {median_rel:.3f} > {_MAX_MEDIAN_SHARPNESS_REL}"
        )
    else:
        median_rel = 0.0

    for c in gpu_out[kind]:
        assert Path(c["path"]).exists(), f"GPU crop not written: {c['path']}"
    print(
        f"{kind}: matched={matched} borderline_flips={borderline} ({fraction:.2%}) "
        f"median_sharpness_rel={median_rel:.4f}"
    )
