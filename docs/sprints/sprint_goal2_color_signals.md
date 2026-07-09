# Goal 2 Color Signals

## What Color Signals Are

Color signals are deterministic summaries of selected body crops. The extractor estimates dominant colors for the approximate top, bottom, and shoes regions of a person crop.

Colors are same-day supporting appearance signals. They complement InternVL3.5-2B clothing description and OSNet_x1_0 ReID, but they are not identity. Face embedding remains the permanent biometric identity anchor.

## Graph Insertion

Current graph source of truth:

`select_best_per_person -> describe_clothing_per_person -> extract_reid -> extract_colors -> build_multi_profile`

Color extraction depends on selected body crops and is independent of clothing and ReID.

## Profile Schema

`profile.json` includes a compact `color_signals` block:

```json
{
  "extractor": "DominantColorExtractor_v1",
  "signal_type": "same_day_supporting_appearance",
  "source_crops": ["..."],
  "per_crop_count": 5,
  "top": {
    "dominant": "white",
    "confidence": 0.82,
    "rgb_mean": [232, 232, 225],
    "votes": {"white": 4, "gray": 1}
  },
  "bottom": {
    "dominant": "navy",
    "confidence": 0.76,
    "rgb_mean": [22, 27, 39],
    "votes": {"navy": 3, "black": 2}
  },
  "shoes": {
    "dominant": "black",
    "confidence": 0.69,
    "rgb_mean": [18, 18, 18],
    "votes": {"black": 4, "gray": 1}
  }
}
```

Per-crop palettes are retained in LangGraph state for debugging but are intentionally not written to `profile.json`.

## Validation

Run from the package root:

```bash
python -m forensics.person_creation.tools.test_color_extractor --crops path/to/body1.jpg path/to/body2.jpg
```

The tool prints per-crop top/bottom/shoes dominant colors and the aggregated color signal with confidence values.

## Signal List

| Signal | Role | Source | Purpose |
|--------|------|--------|---------|
| DominantColorExtractor_v1 | Color signal extraction | Deterministic OpenCV/numpy extractor | Extract dominant top/bottom/shoes colors from selected body crops |

## Limitations

- Region split is heuristic.
- Background may affect color estimates.
- Occlusion can affect bottom and shoes.
- Lighting and shadows can shift color names.
- Colors are not identity.
- Full automatic face-to-body assignment is not implemented in this task.
