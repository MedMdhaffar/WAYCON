# Crop-Writing Architecture Analysis

**Scope:** the crop extraction → staging → promotion → profile path of the WAYCON
person-creation pipeline, centred on `forensics/person_creation/nodes/process_video.py`.
**Basis:** direct code inspection of every pipeline node plus the measured profiling run in
`forensics/person_db/waycon/profiling/` (16.97 s, 1920×1080 @ 30 FPS, `process_every_n=5`,
RTX 2050 4 GB, run under **WSL2** with the repo on `/mnt/d`).
**Status:** analysis only — no code changes proposed here are implemented.

---

## 1. Executive summary

The pipeline writes **578 JPEG files (373 body + 205 face)** during `process_video`, costing
**9.49 s of the node's 31.0 s (30.6 %)**. The measured funnel then shows that only
**77 body + 80 face crops (157 files, 27 %)** are ever promoted; **421 files (73 %) are
written, never read, and deleted** by `finalize`. The single largest amplifier is the
filesystem: the run executed inside WSL2 with output on `/mnt/d` (drvfs/9P), where each
small-file create costs ~16 ms average / 52 ms p95 and each rename ~19 ms — one to two
orders of magnitude above native ext4 or a local NTFS process.

Three further measured problems compound this:

1. **`finalize` costs 22.9 s**, of which ~5.3 s is 200 `db.crop_reference.insert` calls that
   each run a full `cv2.imread` just to record width/height, executed **twice per profile**
   (`register` + `update_crop_paths`).
2. **The remote Face Engine** JPEG-encodes the **full 1080p frame** per selected frame
   (102 × ≈ 0.68 s encode + 6.45 s HTTP round-trips), and face crops that were already
   lossy-saved to disk are **re-encoded a second time** for the `/embed` HTTP request —
   double generation loss on exactly the images used for identity embeddings.
3. **Every kept "best" body crop is decoded from disk up to five separate times**
   (ReID, VLM, colour signals, and twice for DB gallery registration).

The evidence does **not** support jumping to NVDEC or C++. The decode cost is only 3.0 s;
detection (8.3 s) and face-engine transport (7.4 s) matter, but the cheapest large win is
**not writing 73 % of the files at all** and **not putting staging on `/mnt/d`**.

---

## 2. Complete crop lifecycle (traced from code)

### 2.1 One body crop, end to end

| Step | File / function | Medium | Operation |
|---|---|---|---|
| Decode frame | `process_video.py:105` `cap.read()` | CPU RAM | H.264 → BGR NumPy (CPU decode) |
| Person detection | `person_detector.py:37` YOLO `predict` | CPU→GPU→CPU | letterbox on CPU, tensor upload, inference, bbox download |
| Crop extraction | `process_video.py:149` `_crop()` | CPU RAM | NumPy **view** (no copy) |
| Sharpness | `process_video.py:153` `_sharpness()` | CPU | `cvtColor` + `Laplacian(CV_64F)` + `.var()` |
| **Staging write** | `process_video.py:160` `cv2.imwrite` | CPU→disk | JPEG encode + file create in `_staging/body_crops/` |
| Quality filter | `filter_quality.py` | metadata only | bbox size + sharpness threshold — **no disk read** |
| Association | `assign_bodies_to_clusters.py` | metadata only | geometry + conf/sharp scores from dicts — **no disk read** |
| **Promotion** | `promote_crops.py:31` `shutil.move` | disk | rename `_staging/…` → `cluster_N/body_crops/…` |
| Best-crop selection | `select_best.py:7` | disk stat | `Path.exists()` per association |
| ReID | `reid_extractor.py:139` | disk→CPU→GPU | **`cv2.imread` (decode #1)**, resize 256×128, upload, OSNet |
| Clothing VLM | `describe_clothing.py:41` | disk→CPU→GPU | **`cv2.imread` (decode #2)**, pad/resize 448², InternVL |
| Colour signals | `profile_signals.py:78` | disk→CPU | **`cv2.imread` (decode #3)** |
| **Final move** | `finalize.py:52` `shutil.move` | disk | rename `cluster_N/…` → `person_db/person_XXX/body_crops/…` |
| DB gallery insert | `store.py:675` | disk→CPU + DB | **`cv2.imread` (decode #4)** for width/height, SQLite insert — called from `register` **and again (#5)** from `update_crop_paths` |
| Frontend | `service.py:270` `/api/images` | disk→HTTP | `send_file` on demand |

### 2.2 One face crop, end to end

Same decode/detect/crop/sharpness/write path, except detection is the **remote Face Engine**
(`client.py:32`): the full frame is JPEG-encoded and POSTed to `/detect`. Then:

| Step | File / function | Operation |
|---|---|---|
| Quality filter | `filter_quality.py` | metadata only (60×60 min, sharpness ≥ 50) — **123 of 205 faces rejected here, after being written** |
| Embedding | `embed_all_faces.py:18` | `cv2.imread` (decode #1) → `client.embed()` → **second JPEG encode** (`client.py:96`) → HTTP → server `imdecode` → FaceNet |
| Clustering | `cluster_identities.py` | embeddings only, keyed by `crop_path` string |
| Promotion | `promote_crops.py` | rename to `cluster_N/face_crops/` (via `face_records` and `associations`; the remap dict prevents double moves) |
| Final move | `finalize.py` | second rename to `person_db/person_XXX/face_crops/` |
| DB gallery | `store.py:675` | `cv2.imread` ×2 (register + update_crop_paths) |
| Profile image | `store.py:840` `_best_face_crop` | `Path.exists()` per crop, sharpest kept |

### 2.3 Lifecycle diagram with transfer annotations

```
compressed video (disk)
    ↓  [DISK READ + CPU decode — cv2.VideoCapture, 510 reads, 2.98 s]
decoded 1080p BGR frame (CPU RAM, ~6 MB)
    ├──────────────► person detection
    │                [CPU preprocess → CPU→GPU upload → GPU YOLO → GPU→CPU bboxes]
    │                102 calls, 8.33 s
    └──────────────► face detection (REMOTE)
                     [CPU JPEG ENCODE full frame 0.68 s → HTTP 6.45 s → server decode → GPU MTCNN → JSON]
    ↓
crop extraction (CPU, NumPy view — 578×, 0.011 s total)
    ↓
sharpness (CPU cvtColor+Laplacian — 578×, 0.26 s total)
    ↓
staging JPEG write  ◄◄◄ BOTTLENECK
    [CPU JPEG ENCODE + DISK WRITE on /mnt/d — 578×, 9.49 s, avg 16.4 ms, p95 51.6 ms]
    ↓
quality filtering (metadata only — no I/O)          373→364 body, 205→82 face
    ↓
face embedding [DISK READ ×82 → CPU JPEG ENCODE #2 → HTTP → server decode → GPU]  6.9 s
    ↓
clustering (CPU, in-memory)                          4 clusters
    ↓
association / Hungarian matching (metadata only)     77 pairs
    ↓
promotion  [DISK RENAME ×157 — 3.0 s, ~19 ms each]
    ↓
best-crop selection (disk stat only)                 5 per cluster (20 total)
    ↓
ReID      [DISK READ ×20 → CPU→GPU → OSNet]          0.7 s
VLM       [DISK READ ×20 → CPU→GPU → InternVL]       17.3 s (inference-dominated)
colour    [DISK READ ×20 → CPU]
    ↓
finalize  [DISK RENAME ×157 again → DB inserts with DISK READ ×≤200 → staging rmtree ×421 files]
    node total 22.9 s; db.crop_reference.insert 200×, 5.3 s
    ↓
database crop references (SQLite person_gallery, pruned to 10/type)
frontend  [DISK READ on demand — send_file by absolute path]
```

### 2.4 Lifecycle facts (all code-verified)

- **Created / written:** all 578 detections, immediately, in `process_video`, into
  `<output_dir>/_staging/{body,face}_crops/`.
- **Kept:** 77 body + 80 face (this run). Everything else stays in `_staging` and is
  `shutil.rmtree`-deleted by `finalize` — **on success only**.
- **Moved twice:** every kept crop is renamed in `promote_crops` and renamed again in
  `finalize._move_or_merge_dir`. Both are same-filesystem `shutil.move` → `os.rename`
  (output_dir defaults to `forensics/person_db/<name>`, base_db_dir is its parent), so no
  data copy — but ~19 ms per rename on /mnt/d, measured 3.0 s in `promote_crops`.
- **Never copied or re-encoded on disk** — files keep their original bytes from the single
  `imwrite`. The only *re-encode* is in-memory: `client.embed()` JPEG-encodes the
  already-JPEG-decoded face crop for HTTP.
- **Reopened from disk:** face crops 1× (embed) + ≤2× (gallery); best body crops up to 5×
  (ReID, VLM, colour, gallery ×2); non-best promoted body crops ≤2× (gallery only inserts
  `face_crops` + `best_body_crops`, so plain associated body crops are reopened 0×).
- **Nodes requiring a real file path today:** `embed_all_faces`, `select_best`
  (`Path.exists`), `promote_crops`, `compute_reid`, `describe_clothing`,
  `profile_signals.color_signals_from_crops`, `finalize`, `GlobalMemory.update_gallery` /
  `_best_face_crop`, Flask `/api/images`, `/api/person/crop` (delete-by-path), frontend.
- **Nodes that could consume in-memory images with trivial changes:** `embed_all_faces`
  (already passes a decoded array to `client.embed`), `compute_reid._embed_rgb`,
  `describe_clothing` (`describer.describe(crops)` takes arrays), colour signals. All of
  them wrap `cv2.imread` around array-based cores.
- **Path is the identity key everywhere:** `cluster_lookup[path]`, association dedupe,
  the `remap` dict in promotion, sharpness maps `{path: sharpness}`, `person_gallery.path`,
  frontend delete/serve. Metadata does **not** require the file to exist at append time
  (nothing stats the path until `embed_all_faces`/`select_best`), but every later stage
  assumes the string is a real file.
- **Database registration only covers final crops** (`face_crops` + `best_body_crops` per
  profile, pruned to 10 per type) — the DB is *not* coupled to the 578 staged files. The DB
  cost problem is the per-insert `cv2.imread` and the duplicated `update_gallery` pass.
- **Same crop opened multiple times:** yes — verified for best body crops (up to 5 decodes)
  and final face crops (up to 3 decodes including the embed read).

---

## 3. Verified problems (A–Q)

### A. Every detection is materialised immediately — CONFIRMED, measured
All 373+205 detections are written before quality filtering, association, dedupe, or
selection. Measured waste: 421/578 files (73 %) are deleted unread; ≈ 6.9 s of the 9.49 s
write cost bought nothing. 123 of 205 face crops fail the *very next node's* threshold
check (60×60 px, sharpness 50) that needs only the bbox and the already-computed sharpness
— both available **before** the write.

### B. Hundreds of small files on the worst possible filesystem — CONFIRMED, measured
578 `imwrite` calls at 16.4 ms avg / 51.6 ms p95. The profiling metadata proves the run
executed under WSL2 (`Linux-…-microsoft-standard-WSL2`) with the repo on `D:\` → output on
`/mnt/d` via drvfs/9P, where every open/create/close is a cross-VM round trip and Windows
Defender scans each new file on the host side. Native ext4 (inside the WSL VM disk) or a
native Windows process on NTFS typically does a 100 KB JPEG write in well under 1 ms.
Filename f-strings, `Python→OpenCV` call overhead, and JPEG encoding itself
(~1–3 ms for a ~100 KB crop) are minor next to this. **Renames confirm it:** 157 renames
took 3.0 s in `promote_crops` (~19 ms each) — a rename does no encoding at all.

### C. JPEG compression before selection — CONFIRMED
`imwrite` encodes all 578 crops at OpenCV's default JPEG quality (95). Only ~157 encodes
were needed. Selecting in memory first would also let final crops be encoded once at a
deliberate quality level.

### D. Repeated disk read/write cycles — CONFIRMED, enumerated
The RAM→JPEG→disk→JPEG-decode→RAM→tensor→GPU cycle occurs at:
- `embed_all_faces`: 82 reads (plus a **second JPEG encode** each for HTTP);
- `compute_reid`: ≤20 reads; `describe_clothing`: ≤20; colour signals: ≤20;
- `store._insert_gallery_crop`: ≤200 reads across `register` + `update_crop_paths`
  (measured 5.31 s total, 26.6 ms avg — read + insert).
`filter_quality` and `assign_bodies_to_clusters` are metadata-only (no reads) — the crops
sit untouched on disk between the write and `embed_all_faces`.
Total ≈ **340 JPEG decodes** of files whose pixels existed in RAM a few nodes earlier.

### E. Lossy intermediates — CONFIRMED for faces, single-encode elsewhere
Face crops used for identity embeddings suffer **two JPEG generations**: staging `imwrite`,
then `client._image_files` re-encode for `/embed`. Faces are the smallest crops
(≥60 px threshold) where 8×8 block artefacts matter most relative to feature size.
Body crops are encoded once; ReID/VLM/colour all read the same single-generation file.
Sharpness is computed on the raw crop *before* encoding, so stored sharpness slightly
overstates the on-disk image — harmless for ranking (monotone), but a real caveat if
sharpness values are ever compared against thresholds derived from re-read files.
No file on disk is ever re-encoded (moves are renames).

### F. `_crop` returns a view — CONFIRMED, currently safe, dangerous for caching
`frame[y1:y2, x1:x2]` is a non-contiguous NumPy view. `cv2.imwrite` handles strided input
(OpenCV wraps it as a `Mat` with a step, copying internally only if the encoder needs it) —
correctness is fine and measured extraction cost is 0.011 s total. Today the view dies at
the end of each loop iteration and `cap.read()` allocates a fresh frame, so no frame is
retained. **But** any design that stores crops in memory (Options 3/4/5) must call
`.copy()` — a retained view keeps the entire ~6 MB parent frame alive; 100 retained views
from 100 different frames would silently pin ~600 MB.

### G. Sharpness for every crop — CONFIRMED but NOT a bottleneck
Measured: 578 calls, 0.258 s total (0.45 ms avg) — 2.7 % of the write cost. It runs
`cvtColor` + `Laplacian(CV_64F)` + variance on the full crop. It must remain *before*
selection because filtering and top-K ranking consume it. Optimisations (downscale, CV_16S,
GPU) are available but low-value; replacing the metric would invalidate `_MIN_SHARPNESS=50`
and every stored sharpness map, so it should be kept unless benchmarked end-to-end.

### H. Synchronous writes block the loop — CONFIRMED
`cv2.imwrite` runs on the main thread between the face-detection HTTP call of frame *n* and
the decode of frame *n+1*. The 9.49 s of write time is strictly serialized with 8.3 s of
GPU detection, 7.4 s of remote face detection, and 3.0 s of decode, none of which need the
CPU while the file is being created. A bounded writer queue would overlap them; risks
(queue growth, RAM, shutdown, error propagation, ordering, path availability before
downstream nodes run, thread safety of `counters`) are analysed under Option 2.

### I. Staging + promotion — CONFIRMED, rename not copy, cleanup gap on failure
- `promote_crops` uses `shutil.move`; staging and destination share `output_dir`, and
  `finalize`'s second move goes to `output_dir.parent` — same filesystem → `os.rename`
  both times. No byte copying, but two metadata operations per kept file (~19 ms each
  on /mnt/d, measured).
- Every kept crop therefore has **three path identities** over its life
  (`_staging/…` → `cluster_N/…` → `person_XXX/…`), each requiring remap bookkeeping
  (`promote_crops.remap`, `finalize._remap_crop_paths/_remap_sharpness_map/_remap_color_sample_paths`)
  — this remap code exists *only* because files are written before identity is known.
- **Failed / crashed / cancelled runs never clean `_staging`** — `finalize` is the only
  deleter and it runs last. `_run_pipeline` catches exceptions but does no staging cleanup.
  Reruns with the same `output_dir` reuse deterministic filenames (`stem_f000123_b00.jpg`)
  so they overwrite rather than accumulate, but a crashed run leaves up to the full staging
  set (~42 MB here, more for longer videos) orphaned; `tools/cleanup_orphan_crops.py`
  exists but is manual/endpoint-driven.
- `_move_path` handles the duplicate-promotion case (dst exists → unlink src), so repeated
  promotion passes are safe.

### J. Path-based metadata too early — CONFIRMED
The dict appended at `process_video.py:166` stores `path` as the crop's primary key, and
clustering (`crop_path`), association dedupe, promotion remap, sharpness maps, the DB
gallery, and the frontend all key on it. This is the coupling that makes "don't write yet"
non-trivial: deferring the write means either (a) keeping the same *planned* path string as
the key and writing later (Options 2/3/4 — modest change), or (b) introducing a
frame/detection ID key (larger refactor). Nothing between the write and `embed_all_faces`
dereferences the path, so a "write before `embed_all_faces`" contract is the minimal
invariant that preserves all downstream behaviour.

### K. Full frame lives only in CPU RAM — CONFIRMED
`cv2.VideoCapture` decodes on CPU (FFmpeg). Per selected frame: ~6 MB BGR in RAM; YOLO
letterboxes on CPU and uploads ~a 640×640 tensor (≈4.9 MB fp32) to GPU; the face engine
gets a full-frame JPEG over HTTP (second full-frame consumption); crops and JPEG encoding
are pure CPU. GPU→CPU traffic is only bounding boxes. Estimated upload volume for the run:
102 × ~5 MB ≈ 0.5 GB CPU→GPU (estimate; ultralytics internals not measured separately).

### L. NVDEC does not solve crop writing — ANALYSIS
NVDEC would: cut the 2.98 s CPU decode, remove decode from the CPU thread, and (with a
GPU-resident detector path) eliminate the per-frame CPU→GPU upload. It would **not**:
reduce the number of crops written, make `cv2.imwrite` faster, or feed the remote face
engine (which needs a CPU JPEG). Worse, with GPU-resident frames every crop that must
reach disk or the face engine needs a **GPU→CPU download first** — today that copy is free
because frames are already in RAM. On this profile, NVDEC attacks 3.0 s of a 31 s node
while adding transfer complexity; it only becomes compelling after crop-writing is fixed
and if detection moves fully GPU-side.

### M. Remote Face Engine breaks zero-copy — CONFIRMED, measured
Per selected frame: full 1080p JPEG encode (6.7 ms; 0.68 s total) → HTTP POST → server
`imdecode` → MTCNN → JSON. Measured 7.34 s for detection + 5.08 s for 82 embeds ≈ **12.4 s
of pipeline time in face-engine transport+inference**, all localhost HTTP. Any future
NVDEC/GPU-resident frame would still need GPU→CPU→JPEG→HTTP for this service unless the
face engine becomes in-process, accepts person-ROI crops instead of full frames, batches,
or uses shared memory / CUDA IPC. ROI-only requests are the cheapest fix: person boxes are
already available before the face call, and body crops are ~5–10 % of frame pixels.

### N. RTX 2050 4 GB VRAM — CONFIRMED tight
Measured peak: 2.40 GB allocated / 2.98 GB reserved with YOLO + InternVL (1B/2B, bf16) +
OSNet + FaceNet(remote process) resident. Free headroom ≈ 1.0–1.6 GB and fragmentation-
sensitive. A raw 1080p ring buffer costs 5.93 MB/frame → 300 frames ≈ 1.78 GB: **does not
fit** alongside the current models. Any GPU-resident design on this card must hold at most
a few frames (≤10, ~60 MB) plus crops, or evict the VLM while ingesting.

### O. Profiling overhead — PARTIALLY CONFIRMED, needs a Phase-0 measurement
Each `profile_measure` takes **two resource snapshots** (fresh `psutil.Process()`,
`memory_info`, `cpu_times`, `num_threads`, plus three `torch.cuda.memory_*` queries).
Snapshots fall **outside** each block's `perf_counter` window (before start / after end),
so the 16.4 ms crop-write average is *not* inflated by its own snapshots — but the run
recorded 5,114 records ≈ 10,200 snapshots of inter-block wall time that **do** inflate
`node.process_video` and `video.complete` totals (estimate 0.5–3 s; unknown until measured).
**CUDA sync around crop writes: does not occur.** `frame.crop_write` passes no
`synchronize_cuda`, so even with `cuda_sync=true` (which this run had) no
`torch.cuda.synchronize()` brackets writes. `torch.cuda.memory_allocated()` is a CPU-side
allocator query, not a sync. Detection blocks *did* sync (accurate GPU timing, slightly
slower wall time).

### P. Database crop-reference overhead — CONFIRMED but bounded
Only final crops are registered (≤ face_crops + 5 best bodies per cluster, gallery pruned
to 10/type). The measured 5.31 s / 200 inserts comes from: `cv2.imread` per insert for
width/height (store.py:675), `update_gallery` running **twice** per profile (once in
`register`, once in `update_crop_paths` after the finalize move changes every path —
the first pass's rows then point at moved files and are purged by
`_remove_missing_gallery_paths`), and per-statement SQLite work under `isolation_level=None`.
DB cost scales with *final* crop count, not the 578 staged — the fix is decoupling
dimensions from insert (store them in the crop metadata) and registering once, after paths
are final.

### Q. Cleanup and failed runs — CONFIRMED gaps
`_staging` removal happens only in `finalize` on a fully successful run. Failure, crash,
or cancellation at any earlier node leaves the full staging tree. `finalize._prune_orphans`
protects final person dirs; `rejected_detections.json` records rejects but their files are
already gone (staging) or never existed. Disk-growth risk is modest per run (~42 MB) but
unbounded across repeated failed runs with varying output dirs.

---

## 4. Quantified costs

### Measured (from `pipeline_profile.json` / `profiling_summary.txt`)

| Metric | Value |
|---|---|
| Pipeline total | 102.66 s (RTF 6.05 vs 16.97 s video) |
| `node.process_video` | 31.04 s (30.2 %) |
| Crop writes | 578 calls, **9.49 s**, avg 16.42 ms, p95 51.6 ms → **30.6 % of process_video** |
| Frame decode | 510 calls, 2.98 s |
| Person detection | 102 calls, 8.33 s |
| Face detection (remote) | 102 calls, 7.41 s (0.68 s JPEG encode + 6.45 s HTTP) |
| Face embedding (remote) | 82 calls, 5.08 s |
| Sharpness | 578 calls, 0.258 s |
| Crop extraction | 578 calls, 0.011 s |
| `node.promote_crops` | 3.05 s (157 renames ≈ 19 ms each) |
| `node.finalize` | 22.85 s (register 10.6 s, update_crop_paths 9.2 s, gallery inserts 5.3 s/200 calls) |
| Funnel | 578 written → 364+82 pass quality → 77 associations → **77 body + 80 face promoted** → 5 best/cluster (4 clusters) |
| Wasted writes | **421 files = 73 %** (296 body + 125 face) |
| Peak RSS / CUDA | 3.46 GB RAM; 2.40 GB alloc / 2.98 GB reserved VRAM of 4 GB |
| Crops per selected frame | 578 / 102 = 5.7 |

### Code-derived

| Metric | Value |
|---|---|
| Later disk reads (decodes) | 82 (embed) + 20 (ReID) + 20 (VLM) + 20 (colour) + ≤200 (gallery) ≈ **340** |
| JPEG encode events | 578 (staging) + 102 (full-frame detect) + 82 (embed re-encode) = **762** |
| Double-JPEG'd images | all embedded face crops (82) |
| Renames per kept crop | 2 (promote + finalize) |
| DB inserts | final crops only; gallery pruned to 10/type/person |

### Estimates

| Metric | Estimate | Basis |
|---|---|---|
| Staging disk per run | ≈ 42 MB (373 × ~110 KB + 205 × ~9 KB) | measured avg file sizes in `person_db/person_00*` |
| Wasted write time | ≈ 6.9 s (73 % of 9.49 s) | proportional |
| Native-FS write cost for 578 files | < 0.5–1 s | typical ext4/NTFS small-file throughput; **must be benchmarked (Phase 0/1)** |
| CPU→GPU upload volume | ≈ 0.5 GB (102 × ~5 MB tensors) | YOLO 640² fp32; internals not separately measured |
| GPU→CPU volume under NVDEC + top-K | ≈ 80 MB (157 kept crops × ~0.5 MB raw) vs today's 0 extra | back-of-envelope |
| Profiler overhead in process_video | 0.5–3 s (10,200 snapshots) | unknown until Phase 0 A/B |

### Unknowns (require Phase 0)

- Exact `/mnt/d` vs ext4 vs native-Windows write cost on this machine.
- Profiler-off vs profiler-on process_video delta.
- Whether Windows Defender real-time scanning contributes to the 51.6 ms p95.
- Face-engine server-side split (decode vs MTCNN vs NMS) — not instrumentable from client.

---

## 5. Solution options

> Ratings: **Speedup** = expected reduction of the measured 9.5 s write cost + related I/O;
> nothing is claimed as an exact number without benchmarking.

### Option 1 — Keep design, optimize JPEG settings and filesystem placement
Write staging to the WSL-native filesystem (e.g. `~/waycon_staging` on ext4, or set
`output_dir` off `/mnt/*`), optionally lower staging JPEG quality to ~85, drop the
`frame.metadata_construction` probe (measures 12 µs at snapshot cost), and gate per-crop
probes behind `frame_level`.
- **Files:** `service.py` / `run.py` (output_dir default or a `STAGING_DIR` env),
  `process_video.py` (imwrite params), `profiling.py` (probe gating). Note `finalize`
  renames become cross-filesystem **copies** if staging and person_db split volumes —
  either keep both on ext4 or accept 157 copies of ~40 MB (cheap on ext4).
- **Benefit:** potentially most of the 9.5 s if drvfs is the dominant cost (likely);
  zero behavioural change. **Complexity:** very low. **Risk:** low (path remap only).
- **RAM/VRAM/disk:** unchanged / unchanged / same bytes, better-placed.
- **Compatibility:** full (downstream, Flask, Global Memory — paths still real files).
- **RTX 2050:** fine. **Multi-camera future:** insufficient alone (still writes 73 % waste).
- **Why limited:** doesn't touch wasted writes, double-encoding, or re-read cycles.

### Option 2 — Asynchronous bounded crop writer
Main loop enqueues `(path, crop.copy())` on a `queue.Queue(maxsize≈64)`; one writer thread
does `imencode`+write; `process_video` joins the queue before returning so downstream nodes
still see files on disk (preserves the path contract at node boundaries).
- **Files:** `process_video.py` only (+ a small writer utility).
- **Benefit:** overlaps ~9.5 s of I/O with GPU/HTTP waits; wall-clock win bounded by the
  loop's other 21 s — expect writes to mostly vanish from the critical path.
- **Complexity:** low-medium. **Risk:** medium — must handle: bounded queue backpressure
  (block on put, never unbounded), `.copy()` mandatory (F), error propagation (collect
  writer exceptions, surface failed paths so `crop_write_failures` stays truthful),
  clean shutdown on exception (`finally: queue.join()`), counters updated from writer
  results not enqueue, and ordering (irrelevant for correctness — filenames deterministic).
- **RAM:** +≤64 crops ≈ 35 MB. **VRAM/disk:** unchanged.
- **Compatibility:** full. **RTX 2050:** fine. **Future:** good building block, still
  writes everything.

### Option 3 — Hold crops in memory until selection, write only survivors
Keep `{planned_path, frame_idx, bbox, sharpness, crop: np.ndarray (copied)}` in RAM;
`filter_quality` filters in memory; write to disk only what proceeds (either after quality
filtering — simple, writes 364+82; or after association — writes 157 but requires
`embed_all_faces` to consume arrays).
- **Files:** `process_video.py`, `filter_quality.py`, `embed_all_faces.py` (accept
  in-memory image), `select_best.py` (drop `Path.exists` for in-memory entries), state
  plumbing in `graph.py`/`state.py`; `service.py` snapshot (arrays must not be jsonified).
- **Benefit:** eliminates 73 % of writes *and* the 82 embed re-reads; single JPEG
  generation for faces (embed from the raw array).
- **Complexity:** medium-high (touches the state contract; LangGraph state now carries
  arrays — snapshot/serialization code must exclude them). **Risk:** medium-high.
- **RAM:** all 578 raw crops ≈ 0.2–0.3 GB (measured avg crop ~0.5 MB raw) on top of a
  3.5 GB peak — acceptable for one video, **unbounded for long/multi videos** unless
  combined with Option 4's cap. **Views danger:** every stored crop must be `.copy()`d.
- **Compatibility:** frontend/status endpoints can't show candidate crops until the write
  point; delete-by-path endpoint unaffected after writes.
- **RTX 2050:** fine (CPU RAM). **Future:** needs bounding → really wants Option 4.

### Option 4 — Per-track / per-identity top-K retention
Add lightweight IoU tracking across selected frames (or reuse cluster/frame grouping);
keep only top-K crops per track ranked by sharpness×size×confidence with temporal spread;
evict losers immediately. K≈10 bodies/track and K≈10 faces/identity would cover
`select_best`'s top-5 and the gallery's 10-cap with margin.
- **Files:** `process_video.py` (tracker + ranked buffer), `filter_quality.py`,
  possibly new `tracker.py`; downstream unchanged if writes happen before
  `embed_all_faces`.
- **Benefit:** 578 → order-of-100 writes even before association; bounds RAM by design;
  also reduces embed calls (fewer faces to embed) — compounding savings.
- **Complexity:** medium-high (tracking is new behaviour). **Risk:** **high on
  correctness**: clustering currently benefits from many face samples (82 embeddings → 4
  clusters); aggressive K could change cluster shapes, association counts, and evidence
  retention. Retention policy must be explicit (this is forensic evidence).
- **RAM:** bounded (K × tracks × ~0.5 MB). **Disk:** ~5–10× fewer files.
- **Compatibility:** output contract preserved if K ≥ downstream needs; profile counts
  will legitimately differ (fewer `face_crops` in profiles).
- **RTX 2050:** fine. **Future multi-camera:** this is the standard real-time pattern —
  the best long-term shape.

### Option 5 — In-memory *encoded* crop cache
Like Option 3 but store `cv2.imencode(".jpg", crop)` bytes instead of raw arrays; flush
selected ones to disk with a plain `write_bytes`.
- **Files:** as Option 3.
- **Benefit:** removes filesystem cost for rejects; RAM ≈ 42 MB (encoded) instead of
  ~300 MB raw; encoded bytes are directly reusable for the face-engine HTTP body
  (removes the double-encode: send the same JPEG that will be saved).
- **Cost:** still pays 578 JPEG encodes (~1–2 s CPU, can sit in the Option-2 worker);
  still lossy; embeds from single-generation JPEG (same as today's disk file — no quality
  regression, one generation *better* than today's double encode if bytes are reused).
- **Complexity:** medium. **Risk:** medium (same state-contract issues as Option 3, minus
  the view/RAM danger). **RTX 2050:** fine. **Future:** good; pairs naturally with 4.

### Option 6 — Temporary container (LMDB / SQLite BLOBs / HDF5 / tar-append)
Stage all candidates as blobs in one file; materialise only final crops as JPEGs.
- **Files:** `process_video.py`, `embed_all_faces.py`, `promote_crops.py` (becomes
  "extract from container"), cleanup logic.
- **Benefit:** one file handle, sequential appends — kills the many-small-files problem
  even on /mnt/d. **But:** Option 1 (move staging off /mnt/d) buys most of this with 1 %
  of the effort, and Options 3/5 avoid persisting rejects entirely.
- **Complexity:** medium-high; new dependency; new failure modes (container corruption,
  cleanup). **Risk:** medium. **Verdict:** **excessive for this project** — justified only
  if candidates must survive process restarts (they don't; staging is deleted on success
  and orphaned on failure anyway).

### Option 7 — GPU crop extraction + selective host transfer
Keep the decoded frame's tensor on GPU after detection, crop on GPU (PyTorch slicing /
`cv::cuda::GpuMat`), score quality on GPU, download only retained crops (pinned memory).
- **Files:** `person_detector.py`, `process_video.py`, new GPU utility module.
- **Benefit today: small.** The frame is *already on CPU* (cv2 decode), so GPU cropping
  requires uploading the full frame anyway (YOLO's letterboxed upload is smaller than the
  raw frame). Crop extraction costs 0.011 s — there is nothing to win until decode is also
  GPU-side. Only meaningful as a layer on Option 8.
- **Complexity:** high. **Risk:** medium. **VRAM:** +frames on GPU (tight on 4 GB).
- **RTX 2050:** marginal. **Future:** yes, as part of a GPU-resident stack.

### Option 8 — NVDEC + GPU-resident inference
FFmpeg/GStreamer or PyNvVideoCodec → CUDA surfaces → GPU preprocess → detector (TensorRT
or PyTorch) → GPU crop/selection → download selected.
- **Files:** new ingest module replacing `cv2.VideoCapture` in `process_video.py`;
  detector preprocessing; device management.
- **Improves:** decode (2.98 s), CPU→GPU uploads, CPU load, frees CPU for encoding/writes.
- **Does not improve:** crop writing (still CPU JPEG unless Option 9), the remote face
  engine (still needs CPU JPEG — see M), finalize/DB, VLM inference (17 s).
- **VRAM:** decode surfaces + frame queue on a card with ~1 GB headroom — must keep the
  queue ≤ ~10 frames; full ring buffers are unsafe (N).
- **Complexity:** high. **Risk:** medium-high (codec coverage, WSL2 NVDEC support must be
  verified). **RTX 2050:** possible but tight. **Future:** core of a production ingest.

### Option 9 — nvJPEG / GPU JPEG encoding
- **For all intermediate crops:** pointless once Options 3/4 stop writing intermediates.
- **For final selected crops (~157):** CPU encode of 157 crops is ~0.3–0.5 s — not worth a
  new dependency and GPU→CPU staging on a 4 GB card.
- **Verdict:** not justified at this scale; revisit only in a multi-camera GPU pipeline
  where encoded evidence frames are produced continuously.

### Option 10 — Native C++ ingest/detection service
C++ + FFmpeg/NVDEC + CUDA + TensorRT + nvJPEG, exposed via shared memory/pybind11/gRPC.
- **Justified when:** many cameras, 24/7 ingest, Python GIL/serialization measurably the
  ceiling, and the Python pipeline already optimal. **None of that holds here** — the
  measured bottlenecks are wasted writes, a slow mounted filesystem, HTTP transport, and
  duplicated DB work, all fixable in Python. Rewriting `cv2.imwrite` in C++ changes
  nothing: the syscall/9P cost dominates, not the language.
- **Complexity:** very high. **Risk:** high. **Verdict:** premature; keep as the Phase-6
  end-state only if Phases 0–5 evidence demands it.

### Option 11 — No intermediate files: store (video, frame_idx, bbox), re-decode on demand
- **Cost analysis:** each later access = seek + decode. With keyframes every 1–10 s,
  seeking to an arbitrary frame decodes up to hundreds of frames; ReID+VLM+colour+embed
  accesses (~340 today) would each pay it, from Python, serially. Random access to 82 face
  frames spread over the video ≈ re-decoding a large fraction of the video several times.
- **Verdict:** impractical as the primary mechanism; **valuable as a fallback** (profiles
  already store `video_sources`, `frame_idx`, `bbox` — enough to regenerate any crop
  offline for forensic review).

### Option 12 — Recommended hybrid (this project)
```
cv2.VideoCapture decode (keep)                    ← decode is only 3 s; NVDEC later
  → detect every N frames (keep)
  → quality-gate BEFORE any write (bbox size + sharpness — kills 123 faces + tiny bodies for free)
  → bounded in-memory candidate buffer per identity/track, encoded bytes (Opt 5) with
    top-K cap (Opt 4, generous K to protect clustering)
  → embed faces from memory (reuse encoded bytes for HTTP; single JPEG generation)
  → after association: write only promoted crops, once, directly to their
    cluster_N/ destination (skip _staging entirely; promotion becomes bookkeeping)
  → async single writer thread (Opt 2) for those writes
  → staging dir & double-rename removed; finalize does one rename set
  → gallery insert takes width/height from crop metadata (no imread), update_gallery
    runs once, after final paths exist
  → all output dirs on a native filesystem (Opt 1) regardless
```
Preserves: filenames, sharpness values, association behaviour, profile schema, DB
registration, frontend serving (files exist from promotion onward). Removes: ~421 wasted
writes, 82 re-reads + re-encodes, 157 redundant renames, ~200 DB-side imreads, drvfs tax.

---

## 6. Decision matrix

| Option | Likely speedup | Effort | RAM | VRAM | Disk reduction | Code risk | Current-pipeline compat | Future real-time suitability |
|---|---|---|---|---|---|---|---|---|
| 1 Filesystem + JPEG settings | high (if drvfs confirmed) | very low | none | none | none (better placed) | very low | full | low |
| 2 Async bounded writer | medium | low | low (+35 MB) | none | none | medium | full | medium |
| 3 Raw crops in RAM until selection | high | medium-high | high (~0.3 GB, unbounded w/o cap) | none | high (73 %) | medium-high | high (state contract change) | medium |
| 4 Top-K per track | high | medium-high | low (bounded) | none | very high | high (behaviour) | medium (fewer crops in outputs) | very high |
| 5 Encoded-bytes cache | high | medium | medium (~42 MB) | none | high (73 %) | medium | high | high |
| 6 Container staging | medium | medium-high | low | none | high (file count) | medium | medium | low-medium |
| 7 GPU crop extraction | low (today) | high | low | medium | none | medium | high | high |
| 8 NVDEC GPU-resident | low-medium (today) | high | low | high (tight) | none | medium-high | medium | very high |
| 9 nvJPEG | very low | medium | low | low | none | low | high | medium |
| 10 C++ service | unknown (today: low) | very high | low | medium | none | high | low | very high |
| 11 No intermediates, re-decode | negative (read cost explodes) | medium | low | none | very high | high | low | low |
| 12 Hybrid (1+2+4+5 + DB fix) | high | medium | medium (bounded) | none | very high | medium | high | high |

---

## 7. Phased plan

**Phase 0 — Verify measurements** (no behaviour change)
1. Same video, profiling **disabled** vs **enabled** (and `cuda_sync` off vs on) →
   quantify profiler + sync overhead in `node.process_video`.
2. Confirm crop-write blocks never CUDA-sync (code-verified; confirm empirically that
   removing the wrappers doesn't change write timings).
3. Microbenchmark: write the same 578 crops to `/mnt/d/...` vs `~/` (ext4) vs (if the
   service ever runs natively) `D:\` → attribute the 16.4 ms average.
4. Record Defender/real-time-scan status during the native-Windows test.

**Phase 1 — Low-risk experiment**
Add `PERSON_CREATION_DISABLE_CROP_WRITES=1` benchmark flag: skip `imwrite`, keep
detections/sharpness/counters/metadata. Compare `node.process_video`. Output is
non-functional (downstream will find no files) — benchmark only. Files: `process_video.py`.

**Phase 2 — Selective persistence**
(a) Move the quality gate before the write (inline `_body_ok`/`_face_ok` thresholds) —
saves 123 face + 9 body writes with **zero** downstream change (those crops were dropped
by the very next node). (b) Then defer body-crop writes to post-association per Option 12,
writing straight to final cluster dirs. Preserve filenames and the "files exist before
`embed_all_faces`/frontend display" contract or move that contract to "before promotion".
Files: `process_video.py`, `filter_quality.py`, `embed_all_faces.py`, `promote_crops.py`.

**Phase 3 — Async writer**
Bounded queue (≤64), one writer thread, mandatory `.copy()`, error propagation into
`crop_write_failures`, join-before-return, clean shutdown on exception. Files:
`process_video.py`.

**Phase 4 — Face Engine transport**
Send person-ROI crops (or a downscaled frame) to `/detect` instead of full 1080p; reuse
already-encoded JPEG bytes for `/embed` (single generation); optionally batch embeds into
one request; measure local-inference (in-process) vs HTTP. Files: `client.py`,
`face_engine/routes/*`, `process_video.py`, `embed_all_faces.py`.

**Phase 5 — NVDEC prototype**
Side-by-side decoder benchmark (cv2 vs NVDEC via PyNvVideoCodec/ffmpeg-cuda) with identical
detector behaviour; measure decode, transfer, total, and VRAM. Verify NVDEC availability
under WSL2 first. Go/no-go on data.

**Phase 6 — Native GPU-resident path**
Only if Phase 5 shows decode/transfer as the new ceiling *and* multi-camera requirements
materialise. Otherwise skip.

Also fold in (any phase, independent): `update_gallery` once instead of twice; width/height
from metadata instead of `cv2.imread` (store.py:675) — this alone is ~5 s of the 22.9 s
finalize; staging cleanup on failed runs (try/finally in `_run_pipeline` or startup sweep).

---

## 8. Benchmark design

Fixed across all runs: same video file, `process_every_n`, model weights, thresholds,
device, and (where the change allows) identical selected frames and detections. Report:
`node.process_video`, `frame.crop_write` total, selected FPS, pipeline RTF, CPU %, peak
RSS, GPU util, peak VRAM (allocated+reserved), bytes written, file count, and **output
equivalence**: identical detection count, frame indices, bboxes, sharpness values, cluster
count, association count, final profile count, and final crop path sets (allowing for
intentional retention-policy differences, which must be listed explicitly).

| # | Benchmark | Variable |
|---|---|---|
| B1 | Current full pipeline | baseline (profiler on/off pair) |
| B2 | Crop writes disabled | Phase-1 flag |
| B3 | JPEG quality 95 vs 85 vs 75 | imwrite params; also compare embedding drift on same crops |
| B4 | `/mnt/d` vs ext4 vs native NTFS | staging location |
| B5 | Sync vs async writer | Phase 3 |
| B6 | Write-all vs quality-gated vs top-K | Phase 2 / Option 4 (report crop-count deltas) |
| B7 | Raw-array cache vs encoded-bytes cache | Options 3 vs 5 (peak RSS focus) |
| B8 | cv2 decode vs NVDEC | Phase 5 (decode ms, transfer ms, VRAM) |
| B9 | Remote vs in-process face engine | Phase 4 |
| B10 | Full-frame vs person-ROI `/detect` | Phase 4 (must also compare face recall!) |

B3 and B10 need a quality check, not just speed: face-embedding cosine drift (B3) and
detection recall on a labelled subset (B10).

---

## 9. Safety and correctness requirements

Any optimisation must preserve: detection results, frame indices, bboxes, sharpness values
(or a validated replacement with re-derived thresholds), association behaviour, profile
outputs and schema, final crop paths/filenames, Global Memory registration, frontend
serving (`/api/images` by absolute path, `/api/person/crop` delete-by-path), cleanup
behaviour, reproducibility (deterministic filenames), and error handling.

Hard rules:
- **No silent evidence loss** — dropping candidate crops (top-K, quality gating) is a
  retention-policy decision; make K and thresholds explicit, logged, and configurable.
- **No unbounded queues or RAM** — every buffer bounded with backpressure.
- **No retained NumPy views** — `.copy()` anything stored beyond the loop iteration
  (a view pins its ~6 MB parent frame).
- **No `torch.cuda.empty_cache()`** inside benchmark comparisons.
- **No full raw-frame VRAM ring buffer** on the RTX 2050 (1.78 GB/10 s does not fit next
  to a 2.4–3.0 GB model footprint).
- Writer-thread failures must surface in `crop_write_failures` and job status, not vanish.
- Downstream path contract: files must exist on disk before the first node that
  dereferences them (`embed_all_faces` today; promotion under Option 12).

---

## 10. Recommendations

**Three safest improvements**
1. Move staging/output off `/mnt/*` onto the WSL-native filesystem (Option 1 / B4).
2. Quality-gate before writing (inline `filter_quality` thresholds) — removes 132 writes
   this run with zero downstream difference.
3. Stop the double `update_gallery` pass and take width/height from metadata instead of
   `cv2.imread` in `store._insert_gallery_crop` (~5 s of finalize).

**Three highest-impact improvements**
1. Defer body-crop persistence until after association; write only promoted crops directly
   to final dirs (Option 12 core) — eliminates ~73 % of writes and both renames.
2. Face-engine transport fix: ROI-based `/detect`, reuse encoded bytes for `/embed`,
   batch — attacks 12.4 s and the double-JPEG problem simultaneously.
3. Async bounded writer for whatever is still written (Option 2).

**Three most risky improvements**
1. Per-track top-K eviction (Option 4) — changes clustering evidence; needs careful K and
   output-equivalence testing.
2. NVDEC/GPU-resident ingest (Options 7/8) — VRAM-tight on 4 GB, WSL2 codec-path risk,
   attacks the smallest measured cost today.
3. C++ native service (Option 10) — very high effort against bottlenecks that are
   filesystem- and architecture-shaped, not language-shaped.

**Single recommended next experiment**
Phase 0/1 combined: profiler-off baseline, then the same run with (a) crop writes disabled
and (b) staging on ext4. Three runs isolate profiler overhead, write cost, and filesystem
tax — the three numbers every later decision depends on.

**Recommended architecture — current project (RTX 2050, single video)**
Option 12 hybrid: keep cv2 decode and current models; quality-gate pre-write; bounded
in-memory encoded-bytes candidates; embed faces from memory; write only promoted crops once,
asynchronously, to a native filesystem; single gallery registration without imreads; staging
directory and double-rename removed. Expected result from measured data: `process_video`
≈ 31 s → ~21–23 s, `finalize` ≈ 23 s → ~8–12 s, `promote_crops` ≈ 3 s → ~0 s — roughly
25–30 s off a 103 s pipeline without touching any model.

**Recommended architecture — future production multi-camera**
Per-camera ingest workers: NVDEC (or camera-side) decode → GPU-resident detector (TensorRT)
→ lightweight tracker → top-K per track in bounded GPU/CPU memory → selective host transfer
→ async evidence writer to local NVMe → central association + Global Memory service; face
engine either in-process per worker or a batched gRPC/shared-memory service fed ROIs, never
full frames. Adopt only with per-phase measurements justifying each step; the current 4 GB
card is not the target hardware for that design.
