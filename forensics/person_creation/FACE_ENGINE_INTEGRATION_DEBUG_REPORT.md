# Face Engine Integration Debug Report

## Exact problem

The traceback did not come from the Face Engine implementation in this checkout. The current client and Flask service have compatible APIs, while `request body must be a JSON object` does not occur anywhere in this repository or in Face Engine's introducing commit (`2972819`). A different or stale process at `FACE_ENGINE_URL` therefore answered the request.

There was also a real launch mismatch: the implementation was `forensics.face_engine.app`, but the required command names `forensics.face_engine.service`. Before this fix that command failed because `service.py` did not exist, making it easy to leave an unrelated process on port 5010.

At investigation time port 5010 was not listening, so the historical process cannot be identified retroactively. The new health fingerprint below identifies future port conflicts.

## Client and server contracts

For `/detect`, `forensics/face_engine/client.py` calls `_request(..., files=...)`. `_image_files()` JPEG-encodes the BGR array and sends multipart field `image` with content type `image/jpeg`. `forensics/face_engine/routes/detect.py` calls `image_from_request()` in `routes/utils.py`, which accepts either multipart field `image` or JSON `{"image_b64":"<base64 image bytes>"}`.

For `/embed`, the client sends the same multipart field. `routes/embed.py` uses the same decoder and therefore accepts both formats, returning a normalized 512-dimensional embedding. No image-path or raw-body API exists.

The exact client call sites are `FaceEngineClient.detect()` and `FaceEngineClient.embed()` in `client.py`; the exact server decoding is `image_from_request()` in `routes/utils.py`. Thus the checked-out client and server are compatible. The current run failed because the service actually reached accepted only JSON, whereas the WAYCON Flask service checks `request.files` and cannot emit the reported error.

## Configuration and branch consistency

`FACE_ENGINE_URL` selects the client address and defaults to `http://localhost:5010`. Face Engine's server port is `FACE_ENGINE_PORT` (default 5010), and its host is now `FACE_ENGINE_HOST` (default `0.0.0.0`). Its device is `FACE_ENGINE_DEVICE`, falling back to `PERSON_CREATION_DEVICE` after this fix.

This branch contains no readers for `PERSON_CREATION_USE_FACE_ENGINE` or `FACE_ENGINE_FALLBACK_LOCAL`. Face Engine is used unconditionally by `load_models.py`, `process_video.py`, and `embed_all_faces.py`; those variables document intent but do not switch implementations. The API contract is unchanged since commit `2972819`, so no incompatible Face Engine versions are present in this branch.

## Fixes implemented

- Added `forensics/face_engine/service.py`, which launches the actual Face Engine app.
- Added `service=waycon-face-engine` and `api_version=1` to `/health`.
- Added a person_creation startup log with Face Engine URL, identity, API version, and device.
- Added `FACE_ENGINE_HOST` and the `PERSON_CREATION_DEVICE` fallback.
- Retained multipart and JSON-base64 support and added tests for both endpoints/formats.
- Added `forensics/face_engine/tools/smoke_face_engine_api.py`. It contacts only an already-running service and never starts/downloads models.
- Added person_creation `/api/health`.

No local fallback or Goal 5 automation was changed.

## Packages and checks

Runtime imports passed for Flask, Flask-CORS, Requests, OpenCV, NumPy, Torch, and facenet-pytorch. Runtime packages are not missing and did not cause fallback. `pytest` is missing from this interpreter, so pytest collection could not run; this is only a test-runner dependency.

The requested files plus the new files compiled. Service/client imports passed. A direct Flask check passed `/detect` and `/embed` with multipart and JSON-base64, and verified the health fingerprint. The live smoke tool correctly reported that port 5010 was offline without starting models.

Test a running Face Engine:

```bash
cd /home/sesterce/WAYCON_live_camera
python3 -m forensics.face_engine.tools.smoke_face_engine_api --url http://127.0.0.1:5010
curl -X POST -F "image=@/path/to/face.jpg" http://127.0.0.1:5010/embed
```

The dummy image may contain no face; then detection succeeds and embedding is clearly skipped. The curl command exercises embedding with a real face.

## Full project commands

Terminal 1 — Face Engine:

```bash
cd /home/sesterce/WAYCON_live_camera
export PERSON_CREATION_DEVICE=cuda
python3 -m forensics.face_engine.service
```

Terminal 2 — person_creation backend using Face Engine:

```bash
cd /home/sesterce/WAYCON_live_camera
export PERSON_CREATION_AUTO_MODE=1
export PERSON_CREATION_DEVICE=cuda
export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
python3 -m forensics.person_creation.service
```

Terminal 3 — frontend:

```bash
cd /home/sesterce/WAYCON_live_camera/forensics/person_creation/frontend
npm run dev
```

Use `/home/sesterce/WAYCON_live_camera/videos/ak.mp4` in the UI.

Verify both services:

```bash
curl http://127.0.0.1:5010/health
curl http://127.0.0.1:5009/api/health
```

Face Engine health must contain `"service":"waycon-face-engine"`, `"api_version":1`, and eventually `"models_loaded":true`. The backend must log:

```text
[face_engine] using http://127.0.0.1:5010 (service=waycon-face-engine, api_version=1, device=cuda)
```

Face Engine access logs then show `POST /detect` and `POST /embed`.

## Remaining risks

- The old JSON-only process was gone, so current evidence cannot name its executable.
- Full GPU/model inference was intentionally not run; startup may require cached assets or downloads.
- Install `pytest` to run the repository suite in addition to the completed direct contract check.
