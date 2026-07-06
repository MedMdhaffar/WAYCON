# Person Creation — Step 1 Documentation

## Goal

Build the **identity infrastructure** for the forensics system. Before any real-time recognition can happen ("Malek is looking at laptop"), we need a trusted ground-truth profile per person: a verified face embedding, confirmed body crops, and a structured clothing description for that day. This step is the enrollment pipeline — it runs once per person per day and produces a `profile.json` that all downstream modules consume.

The output answers: **who is this person, what do they look like today, and what is their biometric signature?**

---

## Architecture

```
forensics/
└── person_creation/
    ├── graph.py                    ← LangGraph StateGraph (9 nodes)
    ├── state.py                    ← PersonCreationState TypedDict
    ├── run.py                      ← CLI entry point
    ├── service.py                  ← Flask REST API (port 5009)
    ├── models/
    │   ├── person_detector.py      ← yolo26m.pt wrapper (class_id==0)
    │   ├── face_detector.py        ← YOLOv8-Face (HF Hub) wrapper
    │   ├── face_embedder.py        ← FaceNet InceptionResnetV1 vggface2
    │   └── clothing_describer.py   ← InternVL3.5-2B wrapper
    ├── nodes/
    │   ├── load_models.py
    │   ├── process_video.py
    │   ├── filter_quality.py
    │   ├── embed_faces.py
    │   ├── human_in_the_loop.py    ← interrupt() #1
    │   ├── select_best.py
    │   ├── describe_clothing.py
    │   ├── build_profile.py        ← interrupt() #2
    │   └── finalize.py
    ├── prompts/
    │   └── clothing.yaml
    └── frontend/                   ← React + Vite (port 5175)
        └── src/components/
            ├── StartForm.jsx
            ├── ProgressTracker.jsx
            ├── CropsGrid.jsx
            ├── FramePairingPanel.jsx
            ├── AssociationsView.jsx
            ├── ClothingPanel.jsx
            └── ReviewPanel.jsx
```

### Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| `yolo26m.pt` | Person detection (class_id == 0 filter) | Local file |
| YOLOv8-Face | Face detection | HF Hub `arnabdhar/YOLOv8-Face-Detection` |
| InceptionResnetV1 (vggface2) | Face embedding → 512-d L2-normalized | `facenet_pytorch` |
| InternVL3.5-2B | Clothing description (structured JSON) | HF Hub `OpenGVLab/InternVL3_5-2B` |
| OSNet_x1_0 | Person ReID / body appearance embedding | `torchreid` / deep-person-reid |
| DominantColorExtractor_v1 | Color signal extraction | Deterministic OpenCV/numpy extractor |

---

## Pipeline Flow

```
START
  │
  ▼
load_models          Load all 4 model singletons
  │
  ▼
process_video        Frame loop: every 5 frames, run person + face detect,
                     save crops to disk with detection index in filename
  │
  ▼
filter_quality       Body: h≥80px, area≥3000px², sharpness≥50.0
                     Face: w≥60px, h≥60px, sharpness≥50.0
  │
  ▼
embed_faces          FaceNet embed all quality face crops → mean L2-normalized
                     512-d embedding
  │
  ▼
human_in_the_loop    ← interrupt() #1
                     Group detections by frame (only frames with ≥1 face AND ≥1 body)
                     Present frame groups to UI for manual pairing
                     User assigns face → body, deletes wrong crops
                     Save pairing_feedback.json to disk
  │
  ▼
select_best          From confirmed pairs, select top-5 body crops temporally
                     spread (one per video segment, sharpest per segment)
  │
  ▼
describe_clothing    InternVL3.5-2B on top-5 body crops →
                     {top, bottom, shoes, full} structured JSON
  │
  ▼
build_profile        Assemble profile dict
                     ← interrupt() #2 (final review)
                     Apply user clothing corrections if edited in UI
  │
  ▼
finalize             Write profile.json to person_db/
  │
  ▼
END
```

### Two Interrupt Points (LangGraph HITL)

**Interrupt #1 — `human_in_the_loop`:**
The pipeline pauses after all detections are extracted. The UI presents frame groups (all faces + all bodies from the same frame side by side). The user manually pairs the correct face with the correct body, deletes noise (background people, wrong detections). Only confirmed pairs continue to the VLM. This is the identity assignment step — the human IS the identity filter.

**Interrupt #2 — `build_profile`:**
The pipeline pauses again after VLM description. The user reviews the assembled profile (face count, clothing description), edits any wrong clothing fields directly in the UI, and approves. Edited clothing values override the VLM output in the saved profile.

---

## What We Built — Step by Step

### 1. Detection & Crop Extraction (`process_video.py`)

- Opens each video with `cv2.VideoCapture`, processes every `N=5` frames
- Runs person detector and face detector on each processed frame simultaneously
- **Critical**: uses `enumerate()` on detection loops to add detection index to filenames:
  - `{stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg` for bodies
  - `{stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg` for faces
- Saves sharpness (Laplacian variance) per crop as metadata

### 2. Quality Filtering (`filter_quality.py`)

Removes unusable crops before any expensive model runs:
- Body minimum: height 80px, area 3000px², sharpness 50.0
- Face minimum: 60×60px, sharpness 50.0

### 3. Face Embedding (`embed_faces.py`)

- Runs FaceNet (InceptionResnetV1, VGGFace2) on all quality face crops
- Preprocessing: BGR→RGB → PIL → Resize(160,160) → ToTensor → ×255 → standardize((x-127.5)/128)
- Computes mean embedding across all crops, L2-normalizes → final 512-d profile embedding

### 4. Human-in-the-Loop Pairing (`human_in_the_loop.py`)

- Groups all detections by `(frame_idx, video)` key
- Filters to frames with ≥1 face AND ≥1 body (frames with only bodies or only faces are useless for pairing)
- Calls `interrupt()` — pipeline freezes, UI takes over
- On resume: receives `human_pairs` + `deleted_paths`
- **Returns updated `quality_body_crops` and `quality_face_crops`** with deleted files removed (propagates deletions into LangGraph's checkpointed state — critical for downstream nodes)
- Saves `pairing_feedback.json` with full bbox data for every confirmed and deleted crop

### 5. Best Crop Selection (`select_best.py`)

- Divides confirmed pairs' frame range into 5 temporal segments
- Picks the sharpest body crop from each segment
- Fills remaining slots with globally sharpest unused crops
- Checks `Path(p).exists()` before using any path (defensive against UI deletions)

### 6. Clothing Description (`describe_clothing.py`)

- InternVL3.5-2B on up to 5 body crops
- Preprocessing: pad-to-square gray(114,114,114) → BGR→RGB → BICUBIC resize 448×448 → ToTensor → ImageNet normalize → bfloat16
- Structured prompt in `clothing.yaml` returns JSON: `{top, bottom, shoes, full}`
- imread wrapped in try/except (ultralytics patches cv2.imread to raise instead of return None)

### 7. Profile Assembly & Review (`build_profile.py`)

- Assembles full profile dict
- `interrupt()` #2 — shows profile preview to user
- After resume: applies `clothing_override` from user's UI edits before returning
- Clothing corrections flow: ClothingPanel → App.jsx state → ReviewPanel → approve POST body → service resume_value → build_profile applies override

### 8. Finalize (`finalize.py`)

- Writes `profile.json` to `output_dir/`

---

## Service Architecture (Flask + LangGraph Threading)

```
Flask main thread
  └── POST /api/person/start
        └── spawns background thread: _run_pipeline()
              ├── streams graph until interrupt #1
              │     sets job.status = "awaiting_pairing"
              │     waits on resume_event
              │
              ├── POST /api/person/confirm-pairs → sets resume_value, fires event
              │
              ├── streams graph until interrupt #2
              │     sets job.status = "awaiting_review"
              │     waits on resume_event
              │
              ├── POST /api/person/approve → sets resume_value, fires event
              │
              └── streams to END → job.status = "done"
```

**Key design**: `job.snapshot` (Flask-level dict) is separate from LangGraph's MemorySaver checkpointed state. Deletions via `DELETE /api/person/crop` update `job.snapshot` for UI display. The `human_in_the_loop` node propagates these deletions into LangGraph state by returning updated crop lists.

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/person/start` | Start pipeline job |
| GET | `/api/person/status/<job_id>` | Poll status + snapshot |
| POST | `/api/person/confirm-pairs/<job_id>` | Submit human pairs (interrupt #1 resume) |
| POST | `/api/person/approve/<job_id>` | Approve profile (interrupt #2 resume) |
| DELETE | `/api/person/crop/<job_id>` | Delete crop from disk + snapshot |
| GET | `/api/images?path=...` | Serve crop image files |

---

## UI Flow (React + Vite, port 5175)

**4 tabs, auto-switch on pipeline status:**

```
Tab 1: Setup
  → Name, video paths, every_n slider (default 5), Start

Tab 2: Progress & Crops
  → Live pipeline step tracker (9 nodes, color-coded)
  → Body/Face crop grids with sharpness badges
  → × button per crop for immediate deletion
  Auto-switches to Tab 3 when status = "awaiting_pairing"

Tab 3: Pair Faces & Bodies
  → Frame group cards (one per frame with ≥1 face + ≥1 body)
  → Faces row (90×90px) + Bodies row (80×120px)
  → Click face → purple ring → click body → confirmed pair
  → Confirmed pairs shown below with unlink button
  → × deletes crop permanently from disk
  → Skip button per frame
  → Confirm button (disabled if 0 pairs)
  Auto-switches to Tab 4 when status = "awaiting_review"

Tab 4: Review & Approve
  → Associations view: all confirmed pairs with full metadata
    (frame, video, IoU, face/body ratio, dimensions, sharpness)
  → Clothing panel: best 5 body crops + editable top/bottom/shoes/full fields
  → Profile summary card
  → Approve & Save button
```

---

## Issues Encountered & Fixes

### 1. Filename Collision (Critical)

**Problem**: All detections in the same frame got the same filename (`{stem}_f000045.jpg`). Second person overwrote first person's crop. Only the last detection per frame survived on disk.

**Fix**: Added `det_idx` from `enumerate()` to filename: `{stem}_f000045_b01.jpg`, `_b02.jpg`, etc.

### 2. LangGraph State vs Flask Snapshot Desync

**Problem**: `DELETE /api/person/crop` updated `job.snapshot` (Flask dict) but not LangGraph's internal MemorySaver state. When pipeline resumed, `select_best` read stale `quality_body_crops` from LangGraph — including paths to files the user deleted. `describe_clothing` then crashed with FileNotFoundError (ultralytics patches `cv2.imread` to raise, not return None).

**Fix**: `human_in_the_loop` node now filters `quality_body_crops` and `quality_face_crops` using `deleted_paths` from the resume payload, and **returns these filtered lists** — pushing the deletions into LangGraph's checkpointed state before downstream nodes run.

### 3. ultralytics cv2.imread Patch

**Problem**: ultralytics monkey-patches `cv2.imread` via `np.fromfile`, which raises `FileNotFoundError` instead of returning `None`. All our `if img is not None` guards were ineffective.

**Fix**: Wrapped `cv2.imread` calls in `try/except Exception` in `describe_clothing.py`. Added `Path(p).exists()` check in `select_best.py` before using paths.

### 4. Relative Path Resolution

**Problem**: Crop paths stored as relative strings (e.g. `forensics\person_db\malek\body_crops\...`). When service runs from project root, `cv2.imread(relative_path)` works. But `np.fromfile` (ultralytics patch) fails on relative paths in some contexts.

**Fix**: `Path(p).resolve()` converts to absolute before all `cv2.imread` calls.

### 5. 0-Pair Submission

**Problem**: User accidentally clicked Confirm with 0 pairs. Pipeline resumed, `select_best` fell back to stale (deleted) body crops, crashed. Second confirm-pairs attempt returned 400 because pipeline had already moved past the interrupt.

**Fix**: Disabled Confirm button in `FramePairingPanel.jsx` when `totalPairs === 0`. Hard block with red error message.

### 6. Clothing Corrections Not Applied

**Problem**: `ClothingPanel.jsx` had local state for edits and called `onChange?.(next)` — but `App.jsx` never passed the `onChange` prop. Edits were lost on Approve.

**Fix**: Full wiring: `App.jsx` captures `clothingOverride` state → passes `onChange={setClothingOverride}` to `ClothingPanel` → passes `clothingOverride` to `ReviewPanel` → includes in approve POST body → `service.py` passes in `resume_value` → `build_profile.py` applies override to profile before returning.

### 7. VLM Describes Held Objects

**Problem**: Prompt said "Ignore background, floor, and furniture" but VLM still described held objects: *"holding a water bottle and a cup"*.

**Fix**: Tightened `clothing.yaml` constraints: *"Describe ONLY clothing items worn on the body. Do NOT mention held objects, bags, accessories, body movements, or actions."*

### 8. Wrong Face-Body Association (Original Algorithm)

**Problem**: The original automated `associate.py` node used only geometric containment (face center inside body bbox). When 2 people stood close in frame, the wrong body was picked. VLM then described the wrong person's clothes.

**Fix**: Replaced `associate.py` + `select_best.py` + `validate_clothing.py` with a single `human_in_the_loop` node. Human visual assignment within the same frame is 100% accurate and takes ~2 seconds per frame group.

---

## Files Reference

| File | Role |
|------|------|
| `nodes/process_video.py` | Frame loop, detection, crop save |
| `nodes/filter_quality.py` | Min size/sharpness filter |
| `nodes/embed_faces.py` | FaceNet mean embedding |
| `nodes/human_in_the_loop.py` | Frame grouping + interrupt + feedback save |
| `nodes/select_best.py` | Top-5 temporal crop selection |
| `nodes/describe_clothing.py` | InternVL structured clothing description |
| `nodes/build_profile.py` | Profile assembly + interrupt + corrections apply |
| `nodes/finalize.py` | profile.json save |
| `nodes/load_models.py` | All 4 model singletons |
| `models/person_detector.py` | yolo26m.pt, class_id==0 filter |
| `models/face_detector.py` | YOLOv8-Face + Conv/BN fuse patch |
| `models/face_embedder.py` | InceptionResnetV1 vggface2, 512-d L2 |
| `models/clothing_describer.py` | InternVL3.5-2B, 448×448 pad+resize |
| `prompts/clothing.yaml` | VLM clothing + validation prompts |
| `graph.py` | LangGraph StateGraph 9-node linear graph |
| `state.py` | PersonCreationState TypedDict |
| `service.py` | Flask API + 2-interrupt threading |
| `run.py` | CLI entry (--name, --videos, --output, --every) |
| `frontend/src/components/FramePairingPanel.jsx` | Core HITL pairing UI |
| `frontend/src/components/ClothingPanel.jsx` | Editable clothing fields |
| `frontend/src/components/ReviewPanel.jsx` | Final approve |

---

## Feedback Data — First Run on Malek

**Session**: 2026-05-13  
**Videos**: `malek_clips.mp4`, `malek_clips2.mp4`  
**Frame groups presented**: 25 (frames with ≥1 face + ≥1 body)  
**Confirmed pairs**: 24  
**Deleted crops**: 17  

### Deletion Pattern Analysis

| Frame | Bodies detected | Kept | Deleted | Reason |
|-------|----------------|------|---------|--------|
| 115–130 | 2 | b00 | b01 | Second person in frame |
| 135, 140 | 2 | b01 | b00 | Malek was second detected (b01) |
| 145 | 3 | b00 | b01, b02 | Two other people |
| 150, 180 | 2 | b01 | b00 | Malek was second detected |
| **160** | **4** | **none** | **b00–b03 + f00** | **Malek not identifiable — full frame discarded** |
| 175 | 4 | b00 | b01, b02, b03 | Three other people |

**Frame 160** is the most informative: 4 body detections, 1 face — all deleted. Malek either left the frame, was fully occluded, or the detected face was someone else.

### Geometric Observations from Confirmed Pairs

```
face_height_ratio:  0.129 – 0.247  (mean ~0.17)
body_area:          163,012 – 317,608 px²  (always >> 8000 threshold)
face_h:             81 – 186 px
body_h:             591 – 806 px
```

**Key insight**: Malek was always close to the camera (body_area consistently > 150,000 px²). His face height is reliably 13–25% of his body height. These ranges are now known ground truth for this camera setup and can be used to calibrate the future automated association algorithm.

**Malek was sometimes b00, sometimes b01** — YOLO's detection order (left-to-right) does not reliably identify the target person. Position alone is not a discriminative signal.

---

## Profile Created — Malek (2026-05-13)

```json
{
  "id": "malek",
  "name": "Malek",
  "created_at": "2026-05-13",
  "face_crop_count": 24,
  "appearance": {
    "date": "2026-05-13",
    "top": "white polo shirt",
    "bottom": "dark pants",
    "shoes": "black shoes",
    "full": "The person is wearing a white polo shirt, dark pants, and white shoes."
  },
  "face_embedding": [ ... 512 floats, L2-norm ≈ 1.0 ... ],
  "body_crops": [ 24 paths ],
  "best_body_crops": [ 5 paths — temporally spread ],
  "video_sources": [ "malek_clips.mp4", "malek_clips2.mp4" ]
}
```

**Note**: The `full` field contains "white shoes" which contradicts the `shoes` field ("black shoes") — a known VLM inconsistency. The per-field values (`top`, `bottom`, `shoes`) are more reliable than the `full` sentence and should be used as the authoritative clothing description in downstream matching. The clothing correction flow (editable UI → override in profile) was built precisely to handle this.

---

## Observations & Lessons

1. **Human-in-the-loop is the right design for enrollment**. Automated geometric association fails when multiple people are in frame. Visual assignment by the human within a frame takes 1–2 seconds and is 100% accurate.

2. **The feedback file is the real asset**. `pairing_feedback.json` contains labeled ground truth: correct face-body pairs with full bbox data, and confirmed deletions with their context. This data will train the future automated algorithm.

3. **Detection order (b00, b01...) does not equal identity**. Malek was b00 in some frames and b01 in others. Any automated algorithm must not assume the target is always the first detected person.

4. **Frame 160 pattern** (all deleted): when a frame has N detections but zero valid pairs, the human skips it. The automated algorithm should do the same — if no face-body pair scores above a confident threshold, discard the frame entirely.

5. **VLM clothing description quality is good but noisy at the sentence level**. Per-field outputs (top/bottom/shoes) are more reliable than the generated `full` sentence. Always validate `full` against the individual fields.

6. **every_n=5** (process every 5th frame at 30fps) gives sufficient temporal coverage without redundancy. Two 30-second clips yield ~360 candidate frames → ~25–30 frame groups with face+body detections.

7. **The profile is daily**. Clothing description must be regenerated each day. Face embedding is stable (run once, reuse). The architecture correctly separates these: face embedding goes in `face_embedding` (permanent), clothing goes in `appearance.date` (daily).
