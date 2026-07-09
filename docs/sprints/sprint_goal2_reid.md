# Goal 2 ReID Integration

## What Was Added

- `OSNet_x1_0` body appearance embedding through `models/reid_embedder.py`.
- `nodes/extract_reid.py` to compute per-crop ReID embeddings, average them, and L2-normalize the final vector.
- Profile output now includes a compact `reid` block for each person and a legacy top-level first-person `reid` block.

## Why OSNet_x1_0

`OSNet_x1_0` is a standard person ReID backbone with a practical accuracy/speed tradeoff for appearance matching from body crops. In this project it is only a same-day supporting appearance signal. Face embedding remains the permanent biometric identity anchor.

## Graph Insertion

Current graph source of truth:

`select_best_per_person -> describe_clothing_per_person -> extract_reid -> build_multi_profile`

ReID and clothing both depend on selected body crops. Profile assembly consumes both outputs.

## Profile Output

Each ReID block stores:

- `model`: `OSNet_x1_0`
- `embedding_dim`: embedding length, usually 512
- `embedding`: final averaged, L2-normalized vector
- `source_crops`: body crop paths used
- `per_crop_count`: number of per-crop embeddings used
- `signal_type`: `same_day_supporting_appearance`
- `error`: only present when extraction failed

Per-crop embeddings are retained in LangGraph state for debugging, but are intentionally not written to `profile.json`.

## Model List

| Model | Role | Purpose |
|-------|------|---------|
| `yolo26m.pt` | Person detection | Detect body crops |
| YOLOv8-Face | Face detection | Detect face crops |
| FaceNet - InceptionResnetV1 VGGFace2 | Face embedding | Permanent biometric identity anchor |
| InternVL3.5-2B | Clothing description | Daily appearance description |
| OSNet_x1_0 | Person ReID / body appearance embedding | Same-day supporting signal from body crops |

## Dependency Note

No Python dependency manifest exists in this workspace. Install ReID support with one of:

```bash
pip install torchreid
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Do not reinstall or pin `torch` just for this integration; keep the existing CUDA/PyTorch setup.

## Intentionally Not Done Yet

- Full automated face-to-body assignment.
- Color extraction.
- Removing or changing HITL nodes.
- Treating ReID as a permanent identity key.
