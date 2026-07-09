# Global Memory

Global Memory is the persistent identity store for `person_creation`. It keeps
the permanent identity key (FaceNet face embedding) separate from daily
appearance signals such as clothing, color summaries, and optional ReID data.

The default database is:

```text
forensics/person_db/global_memory.sqlite
```

Override it with:

```powershell
$env:PERSON_GLOBAL_MEMORY_DB = "path\to\global_memory.sqlite"
```

## Register Phone Photos

Use clear phone or camera photos where the person has a visible face:

```powershell
python -m forensics.person_creation.tools.register_face_photos_to_memory `
  --name Malek `
  --images path\to\photo1.jpg path\to\photo2.jpg
```

The tool detects the face with the existing YOLO face detector, embeds the crop
with the existing FaceNet embedder, averages all valid photo embeddings, and
stores or updates the matching person.

Phone-photo registration can optionally use the standalone local face engine for
detection and embedding:

```bash
cd /mnt/c/Users/aziza/Documents/GitHub/WAYCON
PERSON_CREATION_DEVICE=cuda python3 -m forensics.face_engine.service
```

In another shell:

```bash
export PERSON_CREATION_USE_FACE_ENGINE=1
export FACE_ENGINE_URL=http://127.0.0.1:5010
export FACE_ENGINE_FALLBACK_LOCAL=0
python3 -m forensics.person_creation.tools.register_face_photos_to_memory \
  --name Malek \
  --images /mnt/c/path/to/photo1.jpg /mnt/c/path/to/photo2.jpg
```

Defaults are unchanged: if `PERSON_CREATION_USE_FACE_ENGINE` is unset or `0`,
registration uses the local detector and embedder. If `FACE_ENGINE_FALLBACK_LOCAL=1`,
a service outage falls back to local models with a warning.

Compare local and service embeddings for the same photo set:

```bash
python3 -m forensics.person_creation.tools.compare_face_engine_photo_embedding \
  --images /mnt/c/path/to/photo1.jpg
```

## Run Video Enrollment

Run the current enrollment UI/backend as usual. After the profile approval
interrupt is approved, `finalize.py` writes `cluster_<id>/profile.json` and then
registers each approved profile into Global Memory.

## Inspect Memory

```powershell
python -m forensics.person_creation.tools.inspect_global_memory
```

Show appearances for one date:

```powershell
python -m forensics.person_creation.tools.inspect_global_memory --date 2026-07-07
```

You can also inspect the SQLite file with any SQLite browser. The core tables
are `persons`, `appearances`, `profile_runs`, `face_photo_sources`, and
`crop_references`.

## Verify Phone Photo And Video Matched

1. Register phone photos first.
2. Inspect memory and note the `person_id` and `identity_source=phone_photo`.
3. Run one video enrollment and approve the generated profile.
4. Inspect memory again.

If the video face embedding matched the phone-photo identity, the same
`person_id` remains and `identity_source` becomes `mixed`. A new duplicate
person means the face similarity was below the current threshold.

## Identity And Appearance

Face embedding is the permanent identity key. All matching compares normalized
FaceNet embeddings with cosine similarity.

Clothing and body description are daily appearance signals. They are useful for
review and downstream context, but they are not the permanent identity key.

Body/ReID embedding is optional and nullable for now. If a finalized profile
does not contain a real body/ReID embedding, `body_reid_embedding_json` stays
`NULL`.
