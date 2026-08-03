# Profile Schema v2

## Why It Changed

The person creation pipeline now produces a multi-person session profile. The older profile shape duplicated `person_1` into top-level fields such as `face_embedding`, `appearance`, `reid`, `color_signals`, and crop lists. That was dangerous because downstream modules could accidentally read only `person_1` and ignore the rest of the session.

## Core Rule

Top level is session metadata only. Each `people[i]` owns its identity, appearance signals, tracks, and crops.

## Signal Semantics

- `identity.face` is the permanent biometric identity anchor.
- `appearance.clothing`, `appearance.reid`, and `appearance.colors` are same-day supporting appearance signals.
- Debug data such as associations, frame groups, per-crop ReID embeddings, and color palettes should stay out of `profile.json`.

## Shape

```json
{
  "schema_version": "2.0",
  "profile_type": "multi_person_session",
  "session": {
    "id": "session_1",
    "name": "session_1",
    "created_at": "YYYY-MM-DD",
    "video_sources": ["..."],
    "process_every_n": 5,
    "people_count": 2
  },
  "models": {
    "person_detector": "yolo26m.pt",
    "face_detector": "YOLOv8-Face",
    "face_embedder": "FaceNet_InceptionResnetV1_VGGFace2",
    "clothing_describer": "InternVL3.5-2B",
    "reid": "OSNet_x1_0",
    "color_extractor": "DominantColorExtractor_v1",
    "association": "auto_associate"
  },
  "people": [
    {
      "person_id": "person_1",
      "track": {
        "frame_range": [55, 475],
        "num_observations": 26,
        "avg_center_movement": 0.1234,
        "avg_iou": 0.4567
      },
      "identity": {
        "face": {
          "model": "FaceNet_InceptionResnetV1_VGGFace2",
          "embedding_dim": 512,
          "embedding": [],
          "source_crops": ["..."],
          "crop_count": 26,
          "signal_type": "permanent_biometric_identity"
        }
      },
      "appearance": {
        "date": "YYYY-MM-DD",
        "clothing": {
          "model": "InternVL3.5-2B",
          "top": "unknown",
          "bottom": "unknown",
          "shoes": "unknown",
          "full": "unknown",
          "source_crops": ["..."]
        },
        "reid": {
          "model": "OSNet_x1_0",
          "embedding_dim": 512,
          "embedding": [],
          "source_crops": ["..."],
          "per_crop_count": 5,
          "signal_type": "same_day_supporting_appearance"
        },
        "colors": {
          "extractor": "DominantColorExtractor_v1",
          "signal_type": "same_day_supporting_appearance",
          "source_crops": ["..."],
          "per_crop_count": 5,
          "top": null,
          "bottom": null,
          "shoes": null
        }
      },
      "crops": {
        "faces": ["..."],
        "bodies": ["..."],
        "best_bodies": ["..."]
      }
    }
  ]
}
```

## Validation

Run:

```bash
python -m forensics.person_creation.tools.validate_profile_schema path/to/profile.json
```

The validator fails if legacy top-level person fields return, if counts mismatch, if embeddings are not L2-normalized, or if a person references another person crop.
