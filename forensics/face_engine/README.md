# Face Engine

Standalone local service for face detection, face embedding, and read-only
Global Memory face recognition.

Phase 1 only adds this service. It does not rewire `person_creation`.

## Start In WSL

```bash
cd /mnt/c/Users/aziza/Documents/GitHub/WAYCON
python3 -m forensics.face_engine.service
```

Defaults:

```bash
FACE_ENGINE_HOST=127.0.0.1
FACE_ENGINE_PORT=5010
```

Useful environment variables:

```bash
export FACE_ENGINE_HOST=127.0.0.1
export FACE_ENGINE_PORT=5010
export PERSON_CREATION_DEVICE=auto
export PERSON_GLOBAL_MEMORY_DB=/mnt/c/Users/aziza/Documents/GitHub/WAYCON/forensics/person_db/global_memory.sqlite
```

Use `PERSON_CREATION_DEVICE=cuda` when you want startup to fail clearly if CUDA
is not available.

## Health

```bash
curl http://127.0.0.1:5010/health
```

## Detect

```bash
curl -X POST http://127.0.0.1:5010/detect \
  -H "Content-Type: application/json" \
  -d '{"image_path":"forensics/person_creation/videos/test.jpg"}'
```

## Embed

```bash
curl -X POST http://127.0.0.1:5010/embed \
  -H "Content-Type: application/json" \
  -d '{"face_crop_path":"forensics/person_db/session/cluster_0/face_crops/face.jpg"}'
```

Python norm check:

```bash
python3 - <<'PY'
import json
import math
import urllib.request

payload = {"face_crop_path": "forensics/person_db/session/cluster_0/face_crops/face.jpg"}
req = urllib.request.Request(
    "http://127.0.0.1:5010/embed",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))
embedding = data["embedding"]
print("embedding length:", len(embedding))
print("embedding norm:", math.sqrt(sum(v * v for v in embedding)))
PY
```

## Detect And Embed

```bash
curl -X POST http://127.0.0.1:5010/detect-and-embed \
  -H "Content-Type: application/json" \
  -d '{"image_path":"forensics/person_creation/videos/test.jpg","save_crops_dir":"forensics/person_db/face_engine_crops"}'
```

## Recognize

By crop path:

```bash
curl -X POST http://127.0.0.1:5010/recognize \
  -H "Content-Type: application/json" \
  -d '{"face_crop_path":"forensics/person_db/session/cluster_0/face_crops/face.jpg","top_k":5}'
```

By embedding:

```bash
curl -X POST http://127.0.0.1:5010/recognize \
  -H "Content-Type: application/json" \
  -d @embedding_payload.json
```

Where `embedding_payload.json` is a valid JSON object with an `embedding`
array of exactly 512 numeric values and an optional `top_k`.

`/recognize` is read-only. It calls `GlobalMemoryStore.search_by_face()` and
does not register people, merge people, or create suggestions.

## Path Handling

The service accepts repo-relative paths and absolute WSL paths. Windows drive
paths such as `C:\Users\...` are converted to `/mnt/c/Users/...` for local WSL
use. Relative paths are resolved from the repo root and cannot escape it with
`..`.
