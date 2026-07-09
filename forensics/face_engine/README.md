# Face Engine

Standalone local service for face detection, face embedding, and read-only
Global Memory face recognition.

The service is optional. `person_creation` still uses local face models unless
phone-photo registration, video face crop embedding, or video face detection is
explicitly configured to call `face_engine`.

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
export PERSON_CREATION_USE_FACE_ENGINE=0
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
export PERSON_GLOBAL_MEMORY_DB=/mnt/c/Users/aziza/Documents/GitHub/WAYCON/forensics/person_db/global_memory.sqlite
```

Use `PERSON_CREATION_DEVICE=cuda` when you want startup to fail clearly if CUDA
is not available.

## Optional Phone-Photo Registration

Phone-photo registration can use this service for face detection and embedding:

```bash
export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
python3 -m forensics.person_creation.tools.register_face_photos_to_memory \
  --name TestService \
  --images /mnt/c/path/to/photo.jpg
```

Set `FACE_ENGINE_FALLBACK_LOCAL=1` to fall back to local models if the service
is unavailable.

## Optional Video Face Detection

`process_video.py` can use `face_engine` for video face detection while keeping
body detection, crop saving, crop names, and metadata local:

```bash
export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
```

Unset `PERSON_CREATION_USE_FACE_ENGINE` or set it to `0` to use the local face
detector. Set `FACE_ENGINE_FALLBACK_LOCAL=1` to fall back to the local detector
if service detection fails.

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

## Detect Bytes

`/detect-bytes` accepts a PNG or JPG encoded image using multipart form field
`image`. It detects faces only; it does not save crops or embed faces.

```bash
curl -X POST http://127.0.0.1:5010/detect-bytes \
  -F "image=@/mnt/c/path/to/frame.png;type=image/png"
```

Python standard-library example:

```bash
python3 - <<'PY'
import urllib.request
import uuid
from pathlib import Path

image_path = Path("/mnt/c/path/to/frame.png")
boundary = "----WAYCON" + uuid.uuid4().hex
body = b""
body += f"--{boundary}\r\n".encode()
body += b'Content-Disposition: form-data; name="image"; filename="frame.png"\r\n'
body += b"Content-Type: image/png\r\n\r\n"
body += image_path.read_bytes()
body += f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:5010/detect-bytes",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(resp.status)
    print(resp.read().decode())
PY
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
