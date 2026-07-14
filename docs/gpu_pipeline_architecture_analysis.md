# GPU-First Video Pipeline — Architecture Analysis (Phase 1)

Status: analysis complete, verified against the live environment on 2026-07-14.
Companion doc: `crop_writing_architecture_analysis.md` (crop lifecycle deep-dive).
Benchmarks: `gpu_pipeline_benchmark_report.md` (Phase 7).

---

## 1. Executive summary

The current `process_video` node decodes on CPU, uploads every selected frame
to the GPU **twice** (once inside Ultralytics for person detection, once inside
the Face Engine after a JPEG→HTTP→JPEG round trip), extracts crops as CPU NumPy
slices and writes every candidate crop to `_staging/` as JPEG.

Profiling of one 1920×1080, 16.97 s video (509 frames, 102 selected):

| Stage | Time | Share of `process_video` (20.51 s) |
|---|---|---|
| person detection | 7.64 s | 37 % |
| face detection (JPEG + HTTP + remote infer) | 8.80 s | 43 % |
| video decoding (all frames) | 1.01 s | 5 % |
| crop writing | 0.51 s | 2 % |

**Decode is ~1 s of a 62 s pipeline. NVDEC by itself is not the win.** The win
is eliminating, per selected frame:

1. CPU BGR→letterbox→normalize preprocessing inside Ultralytics (CPU) followed
   by a host→device upload of the full frame — happens once per frame today;
2. a full-frame JPEG encode (CPU), an HTTP POST, a JPEG decode (CPU), and a
   *second* host→device upload of the same frame inside the Face Engine;
3. CPU crop slicing + CPU Laplacian sharpness + synchronous `cv2.imwrite`
   of every candidate crop (~5 000 writes per longer job, most later deleted).

NVDEC matters *architecturally*: it is what makes it possible for the frame to
be **born on the GPU and never visit CPU RAM at all**, which is the property
the future RTSP real-time pipeline needs.

Verified feasibility on this machine (WSL2, RTX 2050 4 GB, driver 610.43):

- **NVDEC works in WSL2 here.** `libnvcuvid.so` is present in `/usr/lib/wsl/lib`
  and `PyNvVideoCodec 2.1.0` (pip) decodes the test video **GPU-resident at
  ~531 fps** (509 frames in 0.958 s), exposing frames via **DLPack** directly
  as CUDA `torch.Tensor` (NV12). Zero host copies.
- The installed OpenCV 5.0.0 is the pip build: **no CUDA module, no
  `cudacodec`** → OpenCV `cudacodec::VideoReader` is not an option.
- No system `ffmpeg`, no PyAV, no VPF, no TensorRT, no CuPy. `onnxruntime`
  is CPU-only. CUDA toolkit 12.3 (`nvcc`) is installed → native extensions are
  buildable, but **none are required** (see §6).
- torchvision 0.27.1+cu126 provides CUDA `roi_align`, CUDA `nms` and **nvJPEG
  GPU JPEG encode/decode** (verified working) → GPU crop extraction, GPU
  sharpness and GPU JPEG encode need no custom CUDA kernels.

Selected backend: **PyNvVideoCodec (NVDEC) + DLPack → torch, with pure-torch
NV12→RGB conversion, Ultralytics tensor-input inference, in-process CUDA face
detection, GPU crops + nvJPEG encode.** Everything behind feature flags with
the existing CPU path untouched as fallback.

---

## 2. Environment capability matrix (all probed, not assumed)

| Capability | Status | Evidence |
|---|---|---|
| GPU | RTX 2050, 4 GB, SM 8.6 | `torch.cuda.get_device_properties` |
| Driver / CUDA | 610.43.02 / UMD 13.3 (WSL2) | `nvidia-smi` |
| PyTorch | 2.12.1+cu126, CUDA available | probe |
| torchvision | 0.27.1+cu126; CUDA `nms`, `roi_align`, nvJPEG `encode_jpeg`/`decode_jpeg` on `cuda:0` all verified | probe |
| NVDEC in WSL2 | **Available** — `/usr/lib/wsl/lib/libnvcuvid.so`, `libnvidia-encode.so` present | `ldconfig -p` |
| PyNvVideoCodec | **2.1.0 installed (pip), decode verified**: 509 frames GPU-resident in 0.958 s (~531 fps); `SimpleDecoder` supports indexed access; DLPack → `torch.Size([1620,1920]) uint8 cuda:0` (NV12) | probe on `videos/4.mp4` |
| OpenCV | 5.0.0 pip build; FFMPEG=YES, **CUDA devices = 0, no `cv2.cudacodec`** | `cv2.getBuildInformation()` |
| ffmpeg / ffprobe binaries | not installed | `which` |
| PyAV / VPF / TensorRT / CuPy / pycuda / DALI / decord | not installed | import probe |
| onnxruntime | 1.27.0, CPU + Azure providers only | `get_available_providers()` |
| nvcc | CUDA 12.3 at `/usr/local/cuda` | `nvcc --version` |
| ultralytics | 8.4.86; person model `yolo26m.pt` (YOLO26 = NMS-free, end-to-end) | repo + pip |
| Face detector | YOLOv8-Face (`arnabdhar/YOLOv8-Face-Detection` via HF hub), loaded inside the Face Engine Flask process | `face_engine/models/detector.py` |
| Face embedder | facenet-pytorch InceptionResnetV1 (vggface2), 512-d | `face_engine/models/embedder.py` |
| VRAM pressure | **3.6 / 4.0 GB already used** by the resident Face Engine + person-creation service processes | `nvidia-smi` |

---

## 3. Current data flow — every transfer, encode and disk I/O

Files traced: `nodes/process_video.py`, `models/person_detector.py`,
`face_engine/client.py`, `face_engine/routes/{detect,embed}.py`,
`face_engine/models/{detector,embedder}.py`, `nodes/{filter_quality,
embed_all_faces,cluster_identities,assign_bodies_to_clusters,auto_pair,
promote_crops,select_best,compute_reid,describe_clothing,build_profile,
finalize}.py`, `models/{reid_extractor,pose_estimator,clothing_describer}.py`,
`graph.py`, `state.py`, `service.py`, `run.py`, `utils/profiling.py`.

Per **selected** frame (102× for the test video):

```
cv2.VideoCapture.read()                      CPU decode → CPU BGR ndarray (6.2 MB)
│
├─ person_det.detect(frame)                  [Ultralytics predict]
│     letterbox+normalize                    CPU
│     H2D upload #1                          CPU→GPU (full frame)
│     YOLO26 forward                         GPU
│     boxes .cpu().numpy()                   GPU→CPU (tiny, unavoidable)
│
├─ face_det.detect(frame)                    [FaceEngineClient]
│     cv2.imencode('.jpg', frame)            CPU JPEG encode  (full frame!)
│     HTTP POST localhost:5010/detect        loopback copy ×2
│     cv2.imdecode                           CPU JPEG decode  (Face Engine proc)
│     Ultralytics predict (YOLOv8-face)      CPU letterbox → H2D upload #2 → GPU
│     boxes → JSON                           GPU→CPU + serialize
│
└─ per detection (body + face):
      frame[y1:y2, x1:x2]                    CPU slice
      cv2.Laplacian(gray).var()              CPU sharpness
      cv2.imwrite(_staging/... .jpg)         CPU JPEG encode + disk write
                                             (every candidate, most deleted later)
```

Downstream (outside `process_video`, unchanged by this project but recorded):

- `embed_all_faces`: `cv2.imread` **from disk** per quality face crop → JPEG →
  HTTP `/embed` → JPEG decode → H2D upload #3 of the same pixels.
- `auto_pair` / `profile_signals` / `reid_extractor` / `describe_clothing`:
  each `cv2.imread`s crops back from disk.
- `promote_crops` renames confirmed crops out of `_staging/`; `finalize`
  deletes the rejects. **Downstream nodes consume crops by disk path** — this
  is the state contract (`{path, frame_idx, video, bbox, sharpness}` with
  `path` readable from disk) and must be preserved.

Count of avoidable boundaries per selected frame today: **2 full-frame H2D
uploads, 1 full-frame JPEG encode + decode, 1 HTTP round trip, N crop JPEG
writes** — plus a 3rd H2D upload per face crop later in `embed_all_faces`.

---

## 4. NVDEC options evaluated (in the required order)

| Option | Verdict | Reason |
|---|---|---|
| **PyNvVideoCodec** | **SELECTED** | NVIDIA's officially maintained pip wheel (2.1.0, cp311, manylinux). Installed and *tested on this machine*: GPU-resident NV12 frames at ~531 fps from the real test video, DLPack zero-copy into torch. `SimpleDecoder` gives metadata (w/h/fps/frame-count/codec), indexed + sequential access, `use_device_memory=True`. |
| NVIDIA VPF (PyNvCodec) | rejected | Archived/superseded by PyNvVideoCodec; requires source build (ffmpeg dev headers, not present). No advantage over the tested option. |
| FFmpeg NVDEC + native C++ wrapper | rejected (for now) | No ffmpeg binary or dev libs installed; would require building ffmpeg with `--enable-cuvid` plus a pybind11 wrapper — large build surface for zero functional gain over PyNvVideoCodec. Reconsider only if exotic containers/RTSP demuxing exceeds PyNvVideoCodec's demuxer. |
| OpenCV `cudacodec::VideoReader` | **impossible** | pip OpenCV 5.0.0 has no CUDA module (`cv2.cuda.getCudaEnabledDeviceCount() == 0`, no `cv2.cudacodec`). Would require a custom OpenCV CUDA build. |

Fallback decoder: existing `cv2.VideoCapture` (CPU) — kept verbatim; selected
automatically with a clearly logged reason when NVDEC is unavailable
(no `libnvcuvid`, import failure, unsupported codec, or flag off).

---

## 5. Zero-/minimal-copy interop design

- `PyNvVideoCodec.DecodedFrame` implements **DLPack** (`torch.from_dlpack`
  verified). It has no `__cuda_array_interface__`; DLPack is the mechanism.
- The DLPack tensor is a **view of a decoder-owned NV12 surface** from a fixed
  pool. It must be consumed before the decoder recycles the surface. We
  therefore perform NV12→RGB conversion **immediately**, producing a new
  GPU-owned tensor. This is one **device-to-device** copy (unavoidable and
  cheap, ~µs); **no host copy occurs at any point**.
- NV12→RGB: pure torch ops on GPU (split Y and interleaved UV planes,
  bilinear-upsample UV ×2, BT.601/BT.709 matrix). ~5 kernel launches per
  frame; no custom CUDA needed. Color matrix choice is validated against
  `cv2.VideoCapture` output in the Phase 2 prototype (mean abs diff must be
  small; H.264 1080p is typically BT.709 but cv2/ffmpeg uses the stream's VUI).
- Unavoidable transfer boundaries in the new path (each recorded by profiling):
  1. **D2H: final detection metadata** (boxes/scores — a few hundred bytes/frame);
  2. **D2H: encoded JPEG bytes** of crops that are actually persisted
     (compressed, ~30–100 kB each instead of raw pixels);
  3. (fallback paths only) full-frame D2H when the CPU path is selected.

---

## 6. Native C++/CUDA modules — decision

`nvcc` 12.3 is available, so `native/video_decoder`, `native/cuda_preprocess`,
`native/cuda_roi`, `native/inference_bridge` are all *buildable*. But every
boundary they would remove is already removed by maintained libraries:

| Candidate module | Covered by | Custom code needed? |
|---|---|---|
| `native/video_decoder` | PyNvVideoCodec (NVDEC, DLPack) | No |
| `native/cuda_preprocess` (NV12→RGB, letterbox, normalize) | torch CUDA ops (verified) | No — a fused kernel would save ~µs/frame; not the bottleneck |
| `native/cuda_roi` | GPU tensor slicing + `torchvision.ops.roi_align` (verified) | No |
| `native/inference_bridge` | Ultralytics accepts CUDA tensors; face model called in-process | No |
| GPU JPEG | torchvision nvJPEG `encode_jpeg` on cuda (verified) | No |

Writing C++ here would be rewriting fast library code, which the requirements
explicitly forbid. **Decision: no native modules in this phase.** The decoder
abstraction (§7 Layer 1) keeps the seam so an FFmpeg/C++ backend can be added
behind the same interface if PyNvVideoCodec's demuxer ever falls short (e.g.
exotic RTSP servers).

---

## 7. Target architecture (layers → concrete modules)

New package: `forensics/person_creation/gpu/` — importing it never breaks the
CPU path; every entry point is capability-checked and logged.

```
video file / RTSP
   │  Layer 1  gpu/decoder.py      VideoDecoder ABC; NvdecDecoder (PyNvVideoCodec),
   │                               OpenCVDecoder fallback; create_decoder() logs choice
   ▼
DecodedFrame(frame_index, timestamp_seconds, width, height,
             pixel_format, device, tensor)          # NV12 or BGR, GPU or CPU
   │  Layer 2/3  gpu/preprocess.py nv12→rgb (torch), letterbox_gpu, uint8→fp
   ▼
   ├─ Layer 4  PersonDetector.detect_cuda(frame_rgb_u8)   models/person_detector.py
   │           GPU letterbox → ultralytics predict(tensor) → boxes rescaled;
   │           torch.inference_mode(), warm-up, same output dicts
   ├─ Layer 5  gpu/face_local.py   LocalFaceDetector: same YOLOv8-face weights
   │           loaded in-process; detect_cuda(frame) full-frame or ROI-batched
   │           inside person boxes; FaceEngineClient kept as fallback
   ├─ Layer 6  gpu/crops.py        GPU slice crops, GPU Laplacian sharpness
   │           (conv2d, replicates cv2 kernel), stays CUDA until accepted
   └─ Layer 7  gpu/crops.py        nvJPEG encode on GPU → D2H of compressed
               bytes only → write. Optional quality gate (same thresholds as
               filter_quality) skips writing crops that would be filtered.
   Layer 8  gpu/realtime.py        LatestFrameQueue (bounded, drop-oldest,
            dropped counters), RTSP reconnect loop, async VLM stays out of loop
   Layer 9  model lifecycle        singletons already exist; add warm-up in
            load_models; models loaded once per process (unchanged)
   Layer 10 native modules         none needed (§6)
```

Integration: `nodes/process_video.py` dispatches to
`nodes/process_video_gpu.py` when `PERSON_CREATION_GPU_PIPELINE=1`; the GPU
node returns the **identical state contract** (`body_crops`/`face_crops`
dicts with `path` on disk in `_staging/`, `frame_idx`, `video`, `bbox`,
`sharpness`) so `filter_quality` → … → `finalize` are untouched.

### Feature flags

| Flag | Default | Effect |
|---|---|---|
| `PERSON_CREATION_GPU_PIPELINE` | 0 | master switch: GPU process_video node |
| `PERSON_CREATION_USE_NVDEC` | 1 (when GPU pipeline on) | NVDEC decode; falls back to CPU decode + H2D upload with logged reason |
| `PERSON_CREATION_LOCAL_FACE` | 1 (when GPU pipeline on) | in-process CUDA face detection; 0 = HTTP Face Engine |
| `PERSON_CREATION_FINAL_ONLY_CROPS` | 0 | apply filter_quality thresholds on GPU before writing; rejects are never written to `_staging/` |
| `PERSON_CREATION_GPU_JPEG` | 1 (when GPU pipeline on) | nvJPEG GPU encode; 0 = D2H + cv2.imwrite |

### CUDA streams and synchronization

- Single default stream for correctness first; the dependency chain
  decode→convert→infer is naturally ordered. No `torch.cuda.synchronize()` in
  the steady state; the only implicit syncs are the D2H copies of detection
  metadata (already minimal) and JPEG bytes.
- CUDA events (existing `cuda_event_measure`) time GPU stages without global
  syncs; diagnostic mode uses `PERSON_CREATION_PROFILE_CUDA_SYNC=1` as today.
- Double buffering / decode-ahead of more than 2 frames is **rejected**: 4 GB
  VRAM with 3.6 GB already resident leaves no headroom (one 1080p RGB float
  frame ≈ 25 MB; NV12 pool of 4 surfaces ≈ 12 MB is fine).

---

## 8. Model input analysis

**Person detector** (`yolo26m.pt`, YOLO26-m, end-to-end/NMS-free):
Ultralytics `predict()` accepts a `torch.Tensor` source (BCHW, RGB, float
0–1, already on device). Tensor input **bypasses Ultralytics' CPU letterbox**,
so we letterbox on GPU ourselves and rescale output boxes back to original
coordinates. This reuses Ultralytics' own postprocessing (keeps equivalence)
while eliminating the CPU preprocess + H2D upload. If a future Ultralytics
version breaks tensor input, the fallback is calling `model.model(im)`
directly + the library's `non_max_suppression`/E2E postprocess — noted, not
needed today. FP16 only after the Phase 3 equivalence test passes at FP32.

**Face detector** (YOLOv8-face): identical mechanism; the weights file is
small and already cached by the Face Engine's `hf_hub_download`. Running it
in-process adds one model instance (~50 MB VRAM). ROI-batched detection
(crops of person boxes, letterboxed to one batch) is implemented but
**full-frame is the default** for strict equivalence with the current
behavior (the current path detects faces on the full frame; ROI mode changes
recall characteristics and needs its own validation).

**Face embedder** (InceptionResnetV1): stays in the Face Engine for this
project — it runs in `embed_all_faces`, outside `process_video`, on ~tens of
crops. Migrating it in-process is a follow-up (it would remove H2D upload #3
and the per-crop HTTP round trips; recorded as future work).

---

## 9. Expected gains (honest attribution)

Per-frame costs eliminated (102 selected frames, measured today ≈ 20.5 s node):

| Boundary removed | Today | Mechanism |
|---|---|---|
| Face: JPEG encode + HTTP + decode + 2nd H2D | ~8.8 s − remote inference (~2–3 s) ≈ 5–6 s | in-process CUDA-tensor face detection |
| Person: CPU letterbox + H2D upload | ~2–4 s of the 7.64 s | GPU letterbox + tensor predict |
| Decode | 1.0 s → ~0.3 s (and off the CPU) | NVDEC |
| Crop slice + CPU sharpness + write of rejects | ~0.5–1 s | GPU crops, nvJPEG, quality gate |

Realistic target for `process_video`: **20.5 s → 6–9 s** (2.3–3.4×) on this
hardware, dominated afterwards by raw YOLO26-m + YOLOv8-face inference time on
an RTX 2050. The remaining 41.9 s of the 62.4 s pipeline (clustering, VLM,
ReID, per-crop HTTP embeds) is out of scope and will then dominate — stated
plainly so nobody expects a 62 s → 10 s miracle from this change.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **VRAM: 3.6/4.0 GB already used** by resident services; adding an in-process face model + decode surfaces may spill to shared memory (WDDM) and slow down, or OOM | small NV12 pool (≤4 surfaces); no frame queue >2; batch=1; FP32 first; peak-VRAM recorded in benchmarks; flags allow partial rollback (e.g. NVDEC on, local-face off) |
| NV12→RGB color matrix mismatch (BT.601 vs 709) vs cv2 output | Phase 2 prototype compares frames pixel-wise against cv2 and selects the matrix; equivalence gate: detections IoU ≥ 0.98 downstream |
| Ultralytics tensor input semantics change between versions | version pinned (8.4.86); equivalence test in CI catches drift; documented fallback to direct `model.model()` call |
| PyNvVideoCodec surface lifetime (DLPack view recycled) | convert to owned tensor immediately after decode (D2D) |
| JPEG bytes differ between nvJPEG and libjpeg (file hashes will differ) | equivalence is defined on *decoded* content + downstream results (bbox IoU, embedding cosine ≥ 0.999), not file bytes |
| Sharpness threshold boundary effects (GPU float vs cv2 float64 Laplacian) | GPU Laplacian uses the same 3×3 kernel in float64-equivalent math; tolerance asserted in tests; `FINAL_ONLY_CROPS` stays default-off |
| Behind-flag drift (two implementations) | equivalence test runs both paths on `videos/4.mp4` and diffs the state contract |
| RTSP reconnect semantics untestable locally (no live camera) | Layer 8 ships as a bounded-queue + reconnect skeleton with unit tests around the queue/drop logic; live-camera soak test deferred |

## 11. Rollout

1. Phase 2 — `gpu/decoder.py` + standalone prototype: dimensions, timestamps,
   color correctness vs cv2, decode benchmark. No LangGraph integration.
2. Phase 3 — `PersonDetector.detect_cuda` + equivalence vs `detect`.
3. Phase 4 — `gpu/face_local.py` + equivalence vs HTTP Face Engine.
4. Phase 5 — GPU crops/sharpness/nvJPEG + `FINAL_ONLY_CROPS` flag.
5. Phase 6 — `process_video_gpu.py` node behind `PERSON_CREATION_GPU_PIPELINE`,
   `gpu/realtime.py` skeleton.
6. Phase 7 — benchmark modes A–E, `gpu_pipeline_benchmark_report.md`.
