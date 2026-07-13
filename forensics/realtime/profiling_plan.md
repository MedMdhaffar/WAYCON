# Profiling & Measurement Plan — person_creation → realtime

Goal: put a number on every stage before optimizing anything (grab(), NVDEC,
TensorRT, batching...). Every decision from the last few discussions hangs on
these numbers.

---

## 0. Test conditions (do this first, or the numbers are worthless)

- **Fixed inputs**: pick ONE pair of clips (e.g. `malek_clips.mp4` + `malek_clips2.mp4`)
  and use them for every run. Never change inputs between comparison runs.
- **Record the environment** once per machine:
  ```bash
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  python -c "import torch, cv2; print(torch.__version__, torch.version.cuda, cv2.__version__)"
  ```
- **Cold vs warm**: run each experiment 3 times. Run #1 = cold (includes model
  download/load, disk cache empty). Runs #2–3 = warm. Report warm numbers,
  note the cold one separately.
- **One variable at a time**: change `every_n` OR grab() OR sharpness-downscale
  per experiment — never two at once.
- **Log to CSV**, not to your eyes. Template at the bottom.

---

## 1. Level 0 — per-node wall time (the big map)

Wrap every graph node with one decorator. This single table tells you where
the 60–120s of a batch run actually goes.

```python
# forensics/person_creation/profiling.py
import time, csv, functools, os
from pathlib import Path

_CSV = Path(os.environ.get("PROFILE_CSV", "profile_nodes.csv"))

def timed_node(fn):
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state)
        dt = time.perf_counter() - t0
        new = not _CSV.exists()
        with _CSV.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["node", "seconds", "run_id"])
            w.writerow([fn.__name__, f"{dt:.3f}", os.environ.get("RUN_ID", "0")])
        print(f"[profile] {fn.__name__}: {dt:.2f}s")
        return out
    return wrapper
```

In `graph.py`:
```python
from forensics.person_creation.profiling import timed_node
builder.add_node("process_video", timed_node(process_video))
# ... same for all 12 nodes
```

**Measurements to collect:**

| # | Metric | Expectation to test |
|---|--------|---------------------|
| 1.1 | `load_models` seconds | the "~90s" claim from service.py — verify it |
| 1.2 | `process_video` seconds | probably #1 or #2 cost |
| 1.3 | `embed_all_faces` seconds | scales with face count |
| 1.4 | `cluster_identities` seconds | expected negligible — verify |
| 1.5 | `compute_reid` seconds | |
| 1.6 | `describe_clothing` seconds | probably #1 or #2 cost |
| 1.7 | everything else | expected < 1s each — verify |
| 1.8 | total end-to-end | sum check |

---

## 2. Level 1 — inside process_video (decode vs detect vs write)

This is the table that decides grab(), NVDEC, and write-filtering. Accumulate
per-category time across the whole run:

```python
import time
from collections import defaultdict

T = defaultdict(float)
N = defaultdict(int)

def tick(key, t0):
    T[key] += time.perf_counter() - t0
    N[key] += 1

# in the loop:
t0 = time.perf_counter(); ret, frame = cap.read();            tick("decode", t0)
t0 = time.perf_counter(); persons = person_det.detect(frame); tick("person_det", t0)
t0 = time.perf_counter(); faces = face_det.detect(frame);     tick("face_det", t0)
t0 = time.perf_counter(); sharp = _sharpness(crop);           tick("sharpness", t0)
t0 = time.perf_counter(); cv2.imwrite(path, crop);            tick("imwrite", t0)

# at the end:
for k in T:
    print(f"{k:12s} total={T[k]:7.2f}s  n={N[k]:5d}  avg={1000*T[k]/N[k]:7.2f}ms")
```

**Measurements:**

| # | Metric | Decision it feeds |
|---|--------|-------------------|
| 2.1 | total decode s + avg ms/frame | grab() and NVDEC worth it? |
| 2.2 | person_det avg ms/frame | TensorRT worth it? |
| 2.3 | face_det avg ms/frame (incl. HTTP if face_engine is remote!) | in-process vs API overhead |
| 2.4 | sharpness avg ms/crop | downscale worth it? |
| 2.5 | imwrite total s + count | filter-before-write worth it? |
| 2.6 | % of decode spent on frames that are skipped (every_n) | grab() expected gain |

> **2.3 note**: if `FaceEngineClient` goes over HTTP, time it separately from
> model inference (add a timer server-side too). HTTP + JPEG encode per frame
> can quietly cost more than the model itself — this decides whether the
> realtime fast tier uses the client or in-process models.

---

## 3. Level 2 — model-level measurements

### 3.1 Load times & VRAM per model
```python
import torch, time

def measure_load(name, load_fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    m = load_fn()
    torch.cuda.synchronize()
    print(f"{name}: load={time.perf_counter()-t0:.1f}s  vram={(torch.cuda.memory_allocated()-before)/1e9:.2f}GB")
    return m
```
Collect for: yolo26m, YOLOv8-Face, FaceNet, InternVL, ReID, pose.
**Feeds**: the realtime VRAM budget ("can everything stay resident?") and
Docker healthcheck startup timeout.

### 3.2 Inference latency vs batch size (FaceNet + InternVL)
For batch sizes 1, 2, 4, 8, 16 on real crops:
```python
for bs in [1, 2, 4, 8, 16]:
    batch = crops[:bs]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    embedder.embed(batch)
    torch.cuda.synchronize()
    print(f"bs={bs}: {(time.perf_counter()-t0)*1000:.1f}ms total, {(time.perf_counter()-t0)*1000/bs:.1f}ms/item")
```
**Feeds**: fast-tier batching (FaceNet) and slow-tier VLM batching. Expect
per-item cost to drop steeply until the GPU saturates.

### 3.3 GPU utilization during a full run
In a second terminal while the pipeline runs:
```bash
nvidia-smi dmon -s um -d 1 -o T > gpu_trace.txt
```
**Feeds**: if GPU util is low (<40%) during detection, the bottleneck is
CPU-side (decode, preprocessing, Python glue) → batching/NVDEC helps. If util
is pinned at ~100%, the models themselves are the wall → TensorRT/quantization.

---

## 4. CPU, memory, disk

| # | Tool | Command | What to record |
|---|------|---------|----------------|
| 4.1 | py-spy flamegraph (no code changes) | `py-spy record -o flame.svg -- python run.py --name malek ...` | top 5 functions by CPU time |
| 4.2 | RAM peak | `psutil.Process().memory_info().rss` sampled per node | peak RSS; catches frame/crop lists growing |
| 4.3 | staging disk volume | `du -sh person_db/malek/_staging/` after process_video | MB written; ties to 2.5 |
| 4.4 | crop counts | already printed by your nodes | crops written vs crops surviving filter_quality → % wasted writes |

---

## 5. A/B experiments (run after baseline exists)

Each = baseline vs variant on the same clips, 3 warm runs each, compare medians.

| # | Experiment | Variant | Success criterion |
|---|-----------|---------|-------------------|
| 5.1 | grab() skipping | `grab()` for non-processed frames | decode total (2.1) drops ≥20%, identical crop count |
| 5.2 | filter-before-write | inline quality gate in process_video | imwrite total (2.5) drops ~= wasted-write %, identical surviving crops |
| 5.3 | sharpness downscale | Laplacian on 200px-wide gray | sharpness total (2.4) drops; **recalibrate threshold**: log (full_res_sharp, downscaled_sharp) pairs for ~200 crops, fit the mapping, pick new threshold that keeps the same accept set within ±2 crops |
| 5.4 | every_n sweep | every_n ∈ {3, 5, 8, 10} | runtime vs confirmed-pair count vs final embedding quality (5.6) |
| 5.5 | TensorRT export | `model.export(format="engine")` for yolo26m + YOLOv8-Face | 2.2/2.3 avg ms drop; detection boxes IoU ≥ 0.9 vs PyTorch on 50 sample frames |
| 5.6 | embedding stability check | for any variant: cosine similarity between baseline final profile embedding and variant embedding | ≥ 0.99 (the optimization changed speed, not identity) |

> 5.6 is your regression test for ALL of these: fast but wrong = failed.

---

## 6. Realtime-specific measurements (before/while building forensics/realtime)

Use the looped-video-as-fake-RTSP setup from your design doc's build order.

| # | Metric | How | Target from design doc |
|---|--------|-----|------------------------|
| 6.1 | decode ms/frame: CPU (`cv2.VideoCapture`) vs NVDEC (PyAV hwaccel) | timer around read/decode for 1000 frames each | NVDEC meaningfully cheaper per frame + CPU core freed (watch `top`) |
| 6.2 | CPU % of ingestion thread | `top -H` / psutil per-thread | < 1 core with NVDEC |
| 6.3 | fast-tier latency: frame timestamp → recognition log write | `time.time()` stamped at capture, diff at log write; collect p50/p95 over 5 min | **p95 < 1s** |
| 6.4 | ingestion fps vs fast-tier processed fps | counters, log every 10s | processed fps stable, no drift |
| 6.5 | ring buffer fill % | `len(buf)/buf.maxlen` sampled | not pinned at 100% (would mean constant drops) |
| 6.6 | slow-tier queue depth + VLM s/track | counters | queue doesn't grow unbounded during a 2-person 5-min test |
| 6.7 | GlobalMemory write latency + `query_by_face` latency vs profile count | timer; test with 10 / 100 / 1000 synthetic profiles | query stays « fast-tier budget; no `database is locked` during concurrent fast+slow writes (this validates the WAL + writer-thread fix) |
| 6.8 | 30-min soak: RSS + VRAM over time | psutil + `torch.cuda.memory_allocated()` sampled/min | both flat after warmup (no leak, ring buffer actually bounded) |
| 6.9 | RTSP drop/reconnect | kill the fake stream mid-run | reconnect works, backoff logged, pipeline resumes |

---

## 7. Results template (one row per run)

```csv
run_id,date,experiment,every_n,decode_s,person_det_s,face_det_s,sharpness_s,imwrite_s,
embed_s,vlm_s,total_s,crops_written,crops_kept,peak_rss_gb,peak_vram_gb,
gpu_util_avg_pct,embedding_cos_vs_baseline,notes
```

## 8. Order of execution

1. **Day 1**: sections 1 + 2 + 3.3 on the fixed clips → baseline table (3 warm runs).
2. **Day 1–2**: section 3.1/3.2 (loads, VRAM, batch curves) → realtime budget facts.
3. **Day 2–3**: experiments 5.1–5.3 (cheap wins), keep what passes 5.6.
4. **Later**: 5.4/5.5 if the baseline says detection dominates.
5. **When starting realtime**: all of section 6 against the fake RTSP loop.

Rule of thumb throughout: if a proposed optimization targets a stage that is
< 5% of total time in the baseline table, skip it — no matter how fun it is.