# GPU-First Video Pipeline — Benchmark Report (Phase 7)

Measured 2026-07-14 on the dev machine: RTX 2050 (4 GB, SM 8.6), WSL2,
driver 610.43, torch 2.12.1+cu126, PyNvVideoCodec 2.1.0.
Video: `forensics/person_creation/videos/4.mp4` (1920×1080 H.264, 30 fps,
509 frames, 16.93 s), `process_every_n=5` → 102 selected frames.
Harness: `python -m forensics.person_creation.tools.benchmark_gpu_pipeline`
(raw JSON: `/tmp/waycon_bench/results.json`).

Conditions: the Face Engine and person-creation Flask services were running
and holding ~3.5 GB of the 4 GB VRAM, as in production use on this laptop.
Models were loaded (0.70 s) and warmed (1.26 s) once before all modes; cold/
warm rows measure the node body only, with profiling disabled. The "profiled"
row re-runs with the stage profiler enabled (adds ~15 % overhead) and is the
source of the per-frame latency and stage breakdowns.

## Modes

| Mode | Decode | Frame residency | Person detect | Face detect | Crop path |
|---|---|---|---|---|---|
| A (current) | cv2 CPU | CPU NumPy | Ultralytics numpy (CPU letterbox + H2D) | JPEG → HTTP → JPEG → GPU | CPU slice + cv2.imwrite |
| B | **NVDEC** | copied straight back to CPU | same as A | same as A | same as A |
| C | NVDEC | CUDA tensor | `detect_cuda` (GPU letterbox) | JPEG → HTTP (needs full-frame D2H) | GPU crop + nvJPEG |
| D | NVDEC | CUDA tensor | `detect_cuda` | **in-process CUDA** | GPU crop + nvJPEG |
| E | NVDEC | CUDA tensor | `detect_cuda` | in-process CUDA | + final-only quality gate |

## Headline results

| Mode | cold s | warm s | frame lat p50 ms | p95 ms | selected fps | proc VRAM peak MB | sys VRAM peak MB | GPU util mean/max % | CPU util mean % |
|---|---|---|---|---|---|---|---|---|---|
| A | 11.40 | 12.44 | 108.2 | 116.6–123.9 | 6.9 | 174 | 4080 | 33 / 100 | 86 |
| B | 12.98 | 12.48 | 110.7 | 119.4 | 7.6 | 208 | 3668 | 34 / 41 | 47 |
| C | 14.58 | 12.96 | 107.2 | 157.5 | 6.5 | 223 | 3720 | 34 / 44 | 47 |
| D | 6.83 | **6.19** | **41.7** | **44.1** | 13.3 | 217 | 3720 | 62 / 71 | 87 |
| E | 5.79 | **5.83** | **41.1** | **43.5** | **14.2** | 217 | 3720 | 67 / 74 | 92 |

- frame latency := decode + convert/upload + person + face stage time per
  selected frame (crop handling excluded because crop count varies per frame);
  cold/warm are full-node wall times.
- One A warm run in an earlier sweep measured 22.2 s under transient system
  contention; the table uses a clean re-measurement (11.4 / 12.4 s repro).
- Dropped frames: 0 in every mode (batch mode has no frame dropping; the
  bounded-queue drop counters apply to the Layer 8 real-time path).

**`process_video` wall time: 12.4 s → 5.8 s (2.1×). Per-frame latency:
108 ms → 41 ms (2.6×), p95 117–124 ms → 44 ms.**

## Where the time went (profiled stage totals, seconds)

| Stage | A | B | C | D | E |
|---|---|---|---|---|---|
| face detection | 7.23 | 7.00 | 7.92 | **0.91** | **0.89** |
| person detection | 3.96 | 3.60 | 3.46 | 3.31 | 3.28 |
| decode (all 509 frames) | 0.88 | 1.09 | 0.82 | 0.66 | 0.72 |
| D2H copy back to CPU | — | 0.78 | 0.74 (face fallback) | — | — |
| NV12→RGB (102 frames) | — | — | 0.05 | 0.05 | 0.05 |
| crop write | 0.28 | n/m | 1.12 | 0.64 | 0.36 |
| crop sharpness | 0.21 | n/m | 0.33 | 0.25 | 0.24 |

## Attribution — where the speedup actually comes from

1. **Killing the JPEG→HTTP→JPEG face path: −6.3 s.** Face detection drops
   from 7.2 s to 0.9 s once the same YOLOv8-face weights run in-process on
   the CUDA-resident frame. This is 90 % of the total win.
2. **GPU letterbox + tensor input for the person detector: −0.7 s.**
   Preprocessing moves off the CPU; what remains (3.3 s) is raw YOLO26-m
   inference time on an RTX 2050 — the hardware floor for this model.
3. **NVDEC decode: −0.2 s** (0.88 → 0.66+0.05 s). As predicted in the
   analysis, decode was never the bottleneck.
4. **Final-only persistence (E): −0.3 s** crop writing and 445 instead of
   575 files created (the gated crops are ones `filter_quality` would have
   deleted later anyway).

**Mode B is the proof that NVDEC alone is worthless here**: hardware decode
followed by an immediate device-to-host copy back into the old interfaces is
*not faster than mode A* (12.5 s vs 12.4 s) — the decode savings are eaten by
the D2H copy (0.78 s) and everything downstream is unchanged. The value of
NVDEC is that it lets the frame be *born on the GPU and stay there*; the
speedup is realized only when the consumers (modes D/E) accept CUDA tensors.

Mode C isolates the same effect from the other side: GPU person detection
alone doesn't help while face detection still forces a full-frame D2H + JPEG
+ HTTP round trip per frame.

## Remaining transfers in mode E (none of them full raw frames)

1. detection boxes/scores → host (~hundreds of bytes/frame);
2. compressed JPEG bytes of persisted crops (nvJPEG encodes on-GPU);
3. inside `ultralytics.predict()`: the 384×640 letterboxed input is copied to
   host once per call to build `Results.orig_img` (~0.9 MB; part of the
   3.3 s person-detect floor). Removing it means bypassing `predict()` for a
   raw `model.model()` call + manual postprocess — documented follow-up, kept
   out of scope to preserve the equivalence guarantee of Ultralytics' own
   postprocessing.

## VRAM

The GPU pipeline adds ~45 MB process-peak over mode A (174 → 217 MB: NVDEC
surface pool + in-process face model). System-wide peak stayed at ~3.7 GB
with both resident services loaded; mode A actually spiked higher (4.08 GB,
GPU util pegging 100 %) because the Face Engine process does its own GPU
inference on every HTTP request. No OOM in any run; no deep frame queue is
used anywhere (decode-ahead ≤ 1 frame).

## Correctness (see tests for gates)

- `tests/test_gpu_pipeline_equivalence.py` — 8/8 passed: decoder frame count
  and dimensions identical; color mean-abs-diff ≤ 3 (measured ~1.07, BT.709);
  person/face CUDA paths vs numpy paths on identical pixels: identical counts,
  IoU ≥ 0.98, scores ±0.02; local face vs HTTP Face Engine: identical counts,
  IoU ≥ 0.95; GPU sharpness within 1 % of cv2; nvJPEG roundtrip PSNR ≥ 30 dB.
- `tests/test_gpu_node_equivalence.py` — 2/2 passed: full CPU node vs GPU node
  on the real video: body 124 matched detections, 2 borderline flips (1.6 %);
  face 68 matched, 1 flip (1.45 %); median sharpness deviation 0.1 % / 0.25 %;
  filter_quality decisions identical outside the [35, 65] ambiguity band;
  crop files all written.
- Explained non-equivalence: NVDEC and cv2/ffmpeg decode differ by ~1 LSB per
  pixel, so detections sitting exactly at the model's 0.25 confidence
  threshold can flip (verified on frame 120: a person at conf 0.2755 on cv2
  pixels falls below threshold on NVDEC pixels; on *identical* pixels both
  detector paths agree exactly). Bounded at ≤ 5 % in the test suite, measured
  ≈ 1.5 %. These are marginal, low-confidence detections; identity assignment
  is driven by the high-confidence matches, which agree.
- Mode E writes fewer crops **by design** (the quality gate applies
  `filter_quality`'s own thresholds before persistence); the post-
  `filter_quality` state is the same set.

## Remaining bottlenecks (mode E, 5.8 s)

1. **Person inference 3.3 s** (~32 ms/frame yolo26m @ RTX 2050) — now 57 % of
   the node. Next levers: FP16 (needs an equivalence pass first), TensorRT
   export, or a smaller model — all model-accuracy decisions, not plumbing.
2. Face inference 0.9 s (~9 ms/frame).
3. Decode + conversion ~0.7 s.
4. Outside this node, the pipeline still spends ~40 s in clustering /
   embedding (per-crop HTTP!) / ReID / VLM. `embed_all_faces` re-reads crops
   from disk and pays JPEG+HTTP per crop — the same disease this node just
   cured; it is the obvious next target.

## How to run

```bash
# CPU mode (unchanged default)
python -m forensics.person_creation.run --name X --videos ... --output ...

# GPU mode
export PERSON_CREATION_GPU_PIPELINE=1     # master switch
export PERSON_CREATION_USE_NVDEC=1        # default on; 0 → CPU decode fallback
export PERSON_CREATION_LOCAL_FACE=1       # default on; 0 → HTTP Face Engine
export PERSON_CREATION_GPU_JPEG=1         # default on; 0 → cv2.imwrite
export PERSON_CREATION_FINAL_ONLY_CROPS=1 # default off
python -m forensics.person_creation.run --name X --videos ... --output ...

# equivalence tests
python -m pytest tests/test_gpu_pipeline_equivalence.py tests/test_gpu_node_equivalence.py tests/test_realtime_queue.py -v

# decoder prototype / benchmark
python -m forensics.person_creation.tools.gpu_decoder_prototype
python -m forensics.person_creation.tools.benchmark_gpu_pipeline [--modes A,D,E] [--every 5]
```

No native extensions need building — the selected stack (PyNvVideoCodec +
DLPack + torch/torchvision CUDA ops) covers every boundary that native
C++/CUDA modules would have addressed (analysis §6).
