# Live Camera URI Ingestion

## Overview

Person creation accepts either existing video paths or a bounded live camera session. Live RTSP, HTTP, HTTPS, and `file://` sources use the same person detector, Face Engine face detector, staging crop format, automatic face/body assignment, profile builder, and Global Memory finalization as video files.

The automatic graph has no review interruption or approve endpoint:

```text
START → load_models → process_video | process_live_stream
      → filter_quality → embed_all_faces → cluster_identities
      → assign_bodies_to_clusters → promote_crops → select_best
      → compute_reid → describe_clothing → build_profile → finalize → END
```

## UI fields

The setup form has an input-source selector:

- **Video file/path** shows the existing repeatable video path fields.
- **Live camera URI** shows camera URI, optional camera ID, and duration in seconds.

Both modes retain the output directory and process-every-N-frames controls. Live progress/results show whether the stream opened, frames read and processed, dropped frames, requested duration, warnings, and the stream report path.

## API requests

Video mode remains backward compatible; `input_type` may be omitted:

```json
{
  "name": "malek",
  "video_paths": ["/home/sesterce/WAYCON_live_camera/videos/ak.mp4"],
  "every_n": 5,
  "output_dir": "forensics/person_db/runs/malek_video_test"
}
```

Live mode:

```json
{
  "name": "malek",
  "input_type": "camera_uri",
  "camera_uri": "rtsp://user:password@camera_ip:554/Streaming/Channels/101",
  "camera_id": "103",
  "duration_seconds": 30,
  "every_n": 5,
  "output_dir": "forensics/person_db/runs/malek_live_test"
}
```

Live duration defaults to 30 seconds and is clamped to 5–300 seconds. Camera IDs are optional. Invalid schemes and missing camera URIs are rejected before model loading.

## Buffer and crop behavior

`LiveFrameBuffer` owns an OpenCV reader thread and a bounded FIFO with a default maximum of 30 frames. The reader continuously captures frames. When the buffer is full, it discards the oldest frame so inference works on recent footage instead of accumulating an unbounded delay. The consumer samples by the original stream frame index using `every_n`.

Open/read timeout properties are passed to the OpenCV FFmpeg backend when supported. Capture always stops at the configured duration; a five-second no-frame interval ends the stream with a warning, while a stream that opens but yields no frames raises a clear error. The reader releases `VideoCapture` during shutdown.

Both video and live ingestion call the same crop writer. Live crops use names such as:

```text
_staging/body_crops/live103_f000120_b00.jpg
_staging/face_crops/live103_face_f000120_f00.jpg
```

Live crop metadata contains the camera ID, original frame index, UTC timestamp, bounding box, confidence, sharpness, source type, and only the masked source URI. Downstream nodes receive the same `body_crops` and `face_crops` record lists used by video mode.

Successful capture writes `<output_dir>/stream_report.json`. Its statistics are also copied into `session_report.json` by finalization.

## Asynchronous VLM behavior

Clothing description can run in a single-worker `ThreadPoolExecutor`:

```bash
export PERSON_CREATION_ASYNC_VLM=1
export PERSON_CREATION_VLM_TIMEOUT_SECONDS=60
```

The node waits at most the configured timeout. A timeout or VLM exception lets the graph continue with:

```json
{
  "top": "unknown",
  "bottom": "unknown",
  "shoes": "unknown",
  "full": "Clothing description unavailable."
}
```

Successful InternVL structured output is used directly by the node without clothing/color consistency rewriting. Setting `PERSON_CREATION_ASYNC_VLM=0` restores synchronous execution; exceptions still use the fallback.

Python cannot forcibly stop model inference running inside a thread. On timeout the graph stops waiting and finalizes, but the overdue worker may remain alive until the underlying model call returns.

## Credential security

The raw camera URI exists only in in-memory graph state while OpenCV connects. User-info passwords are replaced with `****` in crop metadata, logs produced by this feature, status responses, error tracebacks, stream reports, session reports, and profiles. Common secret query fields such as `password` and `token` are masked too.

For example:

```text
rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101
→ rtsp://admin:****@192.168.1.64:554/Streaming/Channels/101
```

## Run the project

### Terminal 1 — Face Engine

```bash
cd /home/sesterce/WAYCON_live_camera
export PERSON_CREATION_DEVICE=cuda
python3 -m forensics.face_engine.service
```

Verify:

```bash
curl http://127.0.0.1:5010/health
```

The response identifies `waycon-face-engine`, API version `1`, and reports loaded models.

### Terminal 2 — person creation backend

```bash
cd /home/sesterce/WAYCON_live_camera
export PERSON_CREATION_AUTO_MODE=1
export PERSON_CREATION_DEVICE=cuda
export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
export PERSON_CREATION_ASYNC_VLM=1
export PERSON_CREATION_VLM_TIMEOUT_SECONDS=60
python3 -m forensics.person_creation.service
```

Verify:

```bash
curl http://127.0.0.1:5009/api/health
```

### Terminal 3 — frontend

```bash
cd /home/sesterce/WAYCON_live_camera/forensics/person_creation/frontend
npm run dev
```

## API test commands

Video mode:

```bash
curl -X POST http://127.0.0.1:5009/api/person/start \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"malek",
    "video_paths":["/home/sesterce/WAYCON_live_camera/videos/ak.mp4"],
    "every_n":5,
    "output_dir":"forensics/person_db/runs/malek_video_test"
  }'
```

Live camera mode:

```bash
curl -X POST http://127.0.0.1:5009/api/person/start \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"malek",
    "input_type":"camera_uri",
    "camera_uri":"rtsp://user:password@camera_ip:554/Streaming/Channels/101",
    "camera_id":"103",
    "duration_seconds":30,
    "every_n":5,
    "output_dir":"forensics/person_db/runs/malek_live_test"
  }'
```

Poll either returned job ID:

```bash
curl http://127.0.0.1:5009/api/person/status/JOB_ID
```

## Lightweight verification

No camera or model inference is required:

```bash
cd /home/sesterce/WAYCON_live_camera
python3 -m forensics.person_creation.tools.smoke_live_camera_config
python3 -m py_compile \
  forensics/person_creation/live_stream.py \
  forensics/person_creation/service.py \
  forensics/person_creation/graph.py \
  forensics/person_creation/nodes/process_video.py \
  forensics/person_creation/nodes/process_live_stream.py \
  forensics/person_creation/nodes/describe_clothing.py
```

Frontend production build:

```bash
cd /home/sesterce/WAYCON_live_camera/forensics/person_creation/frontend
npm run build
```
