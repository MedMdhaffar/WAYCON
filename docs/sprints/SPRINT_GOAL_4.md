# Goal 4 Sprint - Standalone Face Engine

## Objective

Move face detection and face embedding behind a standalone HTTP service in `forensics/face_engine/`, then make `person_creation` and the global-memory recognition test use the service through `FaceEngineClient`.

## Planned Module

- `forensics/face_engine/app.py` - Flask app and startup model load
- `forensics/face_engine/config.py` - environment-based config
- `forensics/face_engine/models/detector.py` - YOLOv8-Face wrapper copied from person creation
- `forensics/face_engine/models/embedder.py` - FaceNet wrapper with unchanged preprocessing
- `forensics/face_engine/routes/` - health, detect, embed, recognize
- `forensics/face_engine/client.py` - thin HTTP client for callers
- `forensics/face_engine/Dockerfile` - Goal 6 stub
- `forensics/face_engine/tests/test_face_engine.py` - route/client tests where practical

## Planned Refactor Touches

- `forensics/person_creation/nodes/load_models.py`
- `forensics/person_creation/nodes/process_video.py`
- `forensics/person_creation/nodes/embed_all_faces.py`
- `forensics/person_creation/nodes/embed_faces.py`
- `forensics/person_creation/nodes/auto_pair.py`
- `forensics/person_creation/tools/add_face_photos.py`
- `forensics/global_memory/test_recognition.py`

## Constraints

- Keep person creation state keys unchanged.
- Do not change body detection, InternVL, ReID, pose, graph order, or final review interrupt.
- `face_engine` may import `global_memory`; it must not import `person_creation`.
- All direct face model use moves out of `person_creation`.

## Verification

- `python -m py_compile` for changed Python files.
- `python -m pytest tests/test_global_memory.py -q` where the local environment supports pytest.
- `python -m pytest forensics/face_engine/tests/test_face_engine.py -q` where dependencies are installed.
- Manual engine commands:
  - `python -m forensics.face_engine.app`
  - `curl http://localhost:5010/health`
  - `curl -X POST -F "image=@<crop.jpg>" http://localhost:5010/embed`

## Results

Implemented.

### Created

- `forensics/face_engine/__init__.py`
- `forensics/face_engine/app.py`
- `forensics/face_engine/client.py`
- `forensics/face_engine/config.py`
- `forensics/face_engine/Dockerfile`
- `forensics/face_engine/models/__init__.py`
- `forensics/face_engine/models/detector.py`
- `forensics/face_engine/models/embedder.py`
- `forensics/face_engine/routes/__init__.py`
- `forensics/face_engine/routes/utils.py`
- `forensics/face_engine/routes/health.py`
- `forensics/face_engine/routes/detect.py`
- `forensics/face_engine/routes/embed.py`
- `forensics/face_engine/routes/recognize.py`
- `forensics/face_engine/tests/__init__.py`
- `forensics/face_engine/tests/test_face_engine.py`

### Modified

- `forensics/person_creation/nodes/load_models.py`
- `forensics/person_creation/nodes/process_video.py`
- `forensics/person_creation/nodes/embed_all_faces.py`
- `forensics/person_creation/nodes/embed_faces.py`
- `forensics/person_creation/nodes/auto_pair.py`
- `forensics/person_creation/tools/add_face_photos.py`
- `forensics/global_memory/test_recognition.py`

### Deleted

- `forensics/person_creation/models/face_detector.py`
- `forensics/person_creation/models/face_embedder.py`

### Verification Run

- `python -m py_compile ...` for all changed Python files: passed.
- Direct Flask route smoke with fake detector/embedder:
  - `/health` returned 200.
  - `/detect` returned one face.
  - `/embed` returned 512-d embedding.
  - `/recognize` rejected 128-d embedding with 422.
  - `/recognize` returned `recognized: false` against an empty temp Global Memory DB.
- Runtime grep for direct `person_creation.models.face_*` imports: clean. Remaining hits are documentation and metadata strings only.

### Not Run Here

- Full model startup and real `/detect`/`/embed` inference, because this sandbox has no network/GPU model cache confirmation.
- Pytest suite, because this PowerShell environment has no pytest and the available `.venv` is WSL-style. Run from WSL:
  - `python3 -m pytest forensics/face_engine/tests/test_face_engine.py -q`
  - `python3 -m pytest tests/test_global_memory.py -q`
