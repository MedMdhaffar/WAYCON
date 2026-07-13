# Real-Time Pipeline Design

## Goal

Turn `person_creation` from an offline, per-file batch job into a real-time
pipeline: models stay loaded in GPU memory permanently, a live RTSP camera
feed is buffered in a ring buffer, and new/updated person info is written
into `GlobalMemory` incrementally as people appear on camera.

Target environment for this design: production machine with a real GPU, one
RTSP camera to start, near-instant face recognition separate from slower
clothing/VLM description (fast/slow tier split).

## Current State (why real-time needs new plumbing)

- **`person_creation` graph** ([graph.py](../../forensics/person_creation/graph.py))
  is a file-based batch job: `load_models` -> `process_video` (reads a whole
  `.mp4` with `cv2.VideoCapture`) -> clustering -> VLM description ->
  `finalize` (writes to `GlobalMemory`).
- **The killer problem**: [load_models.py](../../forensics/person_creation/nodes/load_models.py)
  calls `.load()` on YOLO, the InternVL clothing VLM, pose estimator, and
  ReID extractor **on every single job** (`/api/person/start` spins a new
  thread that reruns the whole graph, models included). That's the "~90s"
  cost mentioned in [service.py](../../forensics/person_creation/service.py#L130)
  — no model is resident between requests.
- **`face_engine`** is the one part of the codebase that already does this
  right: [app.py](../../forensics/face_engine/app.py) loads the face
  detector/embedder once at process startup and serves them over Flask for
  the life of the process. This is the pattern to replicate.
- There is no camera ingestion code anywhere yet — this part is greenfield.

## Architecture: Two-Tier Real-Time Pipeline

```
RTSP stream
   |
   v
[Ingestion thread]  cv2.VideoCapture(rtsp_url), reconnect/backoff on drop
   |  pushes (frame, timestamp)
   v
[Ring buffer]  bounded deque, ~10s @ capture fps, drop-oldest on overflow
   |
   |--------------> [Fast tier]  every frame or every_n (e.g. every 3rd frame)
   |                    YOLO person detect -> crop -> face detect/embed (face_engine, in-process)
   |                    -> GlobalMemory.query_by_face() -> if match: log "recognized" event now
   |                    -> lightweight tracker (IOU or ByteTrack) assigns/updates track_id
   |                    latency target: <1s from frame capture to recognition_log write
   |
   +--------------> [Slow tier]  triggered per-track, not purely on a timer
                        when a track has been stable >= N frames (new identity candidate)
                        or a track closes (person left frame) or every 5-10s for tracks
                        still active:
                          - pick best face crop + best body crop(s) from that track's buffer window
                          - run VLM clothing description (batched, InternVL already supports
                            multi-image describe() - describe_clothing.py can process several
                            crops in one forward pass)
                          - run ReID embedding
                          - GlobalMemory.register() / update_crop_paths() for that person
```

**Why track-lifecycle triggers instead of a pure fixed timer:** a fixed
5-10s tumbling window can split one person's walk-through across two
windows, causing duplicate/fragmented enrollments and wasted VLM calls on
the same person twice. Keying the slow tier off the tracker's state (track
confirmed / track closed / periodic refresh while still active) means the
ring buffer becomes the *lookback window a track pulls its best frames
from*, not the sole scheduling clock. The ring buffer stays, but slow-tier
work is driven by tracking events, with the buffer duration as a
ceiling/fallback (e.g. "if a track has been open >10s with no close, force
a refresh").

## Components and Where They'd Live

New package `forensics/realtime/`, reusing existing singletons rather than
rewriting them:

| Component | Reuses |
|---|---|
| Model service | `get_person_detector()`, `get_clothing_describer()`, `get_reid_extractor()`, `get_pose_estimator()`, `FaceEngineClient` — all already lazy singletons in [person_creation/models/](../../forensics/person_creation/models/), just call `.load()` **once** at process start instead of per job |
| Ingestion | new `RtspStream` wrapper: `cv2.VideoCapture(url)`, watchdog thread, exponential-backoff reconnect |
| Ring buffer | new `FrameRingBuffer`: `collections.deque(maxlen=fps*10)`, thread-safe, timestamped |
| Tracker | new lightweight IOU tracker (or `bytetrack`/`supervision` dep) associating YOLO detections frame-to-frame into track_ids |
| Fast tier | new `fast_worker.py`: per-frame face match, writes straight to `GlobalMemory` |
| Slow tier | new `slow_worker.py`: adapts `describe_clothing.py`/`compute_reid.py`/`build_profile.py` node logic (already pure functions operating on crop lists) to run per-track instead of per-cluster-in-a-batch-job |
| GlobalMemory writer | single dedicated writer thread consuming a queue — see concurrency note below |

## GlobalMemory Concurrency (needed regardless of approach)

Right now every request does `GlobalMemory()` -> open connection -> close
([store.py](../../forensics/global_memory/store.py#L16-L28)). That's fine
for occasional batch jobs but under continuous real-time writes (fast tier
logging every second, slow tier writing every few seconds) it will produce
`sqlite3.OperationalError: database is locked` under concurrent writers.
Two changes needed either way:

1. Add `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000` in the
   schema/`__init__`.
2. Route all writes through one long-lived `GlobalMemory` instance + a
   single writer thread/queue (fast tier and slow tier both enqueue write
   ops rather than opening their own connections), so SQLite only ever sees
   one writer at a time.

## GPU/VRAM Budget

Since this is a production box with a real GPU: keep everything resident
simultaneously — YOLO (~50MB), face detector+embedder (small), InternVL
3.5-1B/2B (few GB in bf16), ReID, pose. That's the entire point of fixing
`load_models` — load once at service startup, never again. Batch the VLM
call across multiple tracks' best-crops in one `describe()` invocation when
several tracks close around the same time, since
`ClothingDescriber.describe()` already accepts a list of crops
([clothing_describer.py](../../forensics/person_creation/models/clothing_describer.py#L83-L106)).

## Other Approaches Considered

1. **Pure fixed-window batch (no tracker)** — simplest to build, but
   re-detects/re-embeds the same person every window and risks
   double-enrollment for anyone visible across a window boundary. Good as a
   v0 to validate the pipeline before adding tracking.
2. **Motion/presence-gated processing** — run a cheap frame-diff or low-res
   person check before waking the expensive models, to save GPU when the
   scene is empty. Worth layering on top of either design once real footage
   shows how often the camera is empty.
3. **Message-queue microservices (Redis Streams/Kafka + worker pool)** — the
   right answer at 5+ cameras/multi-GPU, overkill for 1 camera today.
   Worth naming because the in-process design should still keep ingestion
   and inference cleanly separated (via an in-memory queue) so migrating to
   Redis later is a swap, not a rewrite.
4. **Single-tier (no fast/slow split)** — simpler, one cadence for
   everything. Fallback if the tracker/two-tier complexity isn't worth it
   initially.

## Suggested Build Order

1. Fix `load_models` reload problem: turn model loading into a one-time
   startup step (new `forensics/realtime/model_service.py`, mirrors
   `face_engine/app.py`'s pattern) — this alone is valuable even before
   real-time capture exists.
2. Add WAL mode + single-writer queue to `GlobalMemory`.
3. Build ingestion + ring buffer against a **pre-recorded video looped as a
   fake RTSP-paced stream** (no real camera needed to develop/test).
4. Add fast tier (per-frame face match + log) — testable against step 3
   immediately.
5. Add tracker + slow tier (VLM/ReID per track).
6. Swap in a real RTSP URL, add reconnect/backoff, tune
   `every_n`/window sizes against real footage.
