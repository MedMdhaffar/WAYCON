from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np


_COLOR_NAMES = [
    ("black", (20, 20, 20)),
    ("white", (235, 235, 235)),
    ("gray", (128, 128, 128)),
    ("red", (200, 45, 45)),
    ("orange", (220, 120, 35)),
    ("yellow", (220, 200, 45)),
    ("green", (50, 150, 70)),
    ("blue", (45, 95, 190)),
    ("purple", (120, 75, 170)),
    ("pink", (220, 110, 155)),
    ("brown", (115, 75, 45)),
]


def _nearest_color_name(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    best_name = "unknown"
    best_dist = float("inf")
    for name, ref in _COLOR_NAMES:
        rr, gg, bb = ref
        dist = (r - rr) ** 2 + (g - gg) ** 2 + (b - bb) ** 2
        if dist < best_dist:
            best_name = name
            best_dist = dist
    return best_name


def _dominant_rgb(region_bgr: np.ndarray) -> tuple[int, int, int] | None:
    if region_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    if len(pixels) == 0:
        return None

    # Drop very dark shadows and very bright background highlights when enough
    # pixels remain. This keeps the signal clothing-oriented without segmentation.
    brightness = pixels.mean(axis=1)
    mask = (brightness > 25) & (brightness < 245)
    if int(mask.sum()) > 100:
        pixels = pixels[mask]

    quantized = (pixels // 32) * 32 + 16
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    return int(dominant[0]), int(dominant[1]), int(dominant[2])


def _region_signal(img_bgr: np.ndarray, y1_frac: float, y2_frac: float) -> dict:
    h = img_bgr.shape[0]
    y1 = max(0, min(h, int(h * y1_frac)))
    y2 = max(0, min(h, int(h * y2_frac)))
    rgb = _dominant_rgb(img_bgr[y1:y2, :])
    if rgb is None:
        return {"name": "unknown", "rgb": None}
    return {"name": _nearest_color_name(rgb), "rgb": list(rgb)}


def color_signals_from_crops(paths: list[str]) -> dict:
    top_names: list[str] = []
    bottom_names: list[str] = []
    samples: list[dict] = []

    for raw in paths:
        p = Path(raw)
        if not p.exists():
            continue
        img = cv2.imread(str(p.resolve()))
        if img is None:
            continue
        top = _region_signal(img, 0.18, 0.55)
        bottom = _region_signal(img, 0.55, 0.90)
        top_names.append(top["name"])
        bottom_names.append(bottom["name"])
        samples.append({"path": raw, "top": top, "bottom": bottom})

    def majority(names: list[str]) -> str:
        known = [n for n in names if n != "unknown"]
        if not known:
            return "unknown"
        return Counter(known).most_common(1)[0][0]

    return {
        "method": "dominant_rgb_body_crop_split_v1",
        "sample_count": len(samples),
        "top": majority(top_names),
        "bottom": majority(bottom_names),
        "samples": samples,
    }


def build_association_meta(state: dict) -> dict:
    associations = state.get("associations") or []
    return {
        "source": "automatic_multi_cue_v2",
        "association_count": len(associations),
        "auto_pair_score_mean": round(
            sum(float(a.get("auto_score", 0.0)) for a in associations) / max(len(associations), 1),
            4,
        ),
    }


def build_reid_signal(state: dict, _color_signals: dict) -> dict:
    try:
        from forensics.person_creation.models.body_reid import get_body_reid

        reid = get_body_reid()
        if not reid.is_available():
            return {
                "status": "not_computed",
                "reason": "no_reid_model_configured",
                "body_embedding": None,
                "note": "Reserved for body ReID embedding (OSNet or equivalent). Permanent identity is in face_embedding.",
            }

        embeddings = []
        for raw in state.get("best_body_crops", [])[:5]:
            p = Path(raw)
            if not p.exists():
                continue
            img = cv2.imread(str(p.resolve()))
            if img is None:
                continue
            emb = reid.embed(img)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            return {
                "status": "not_computed",
                "reason": "no_reid_embedding_produced",
                "body_embedding": None,
                "note": "Reserved for body ReID embedding (OSNet or equivalent). Permanent identity is in face_embedding.",
            }

        body_embedding = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0)
        norm = np.linalg.norm(body_embedding)
        if norm > 0:
            body_embedding = body_embedding / norm
        return {
            "status": "computed",
            "model": "osnet_x0_25",
            "embedding_dim": int(len(body_embedding)),
            "body_embedding": body_embedding.tolist(),
            "aggregation": "mean_of_best_5_body_crops",
        }
    except Exception as exc:
        return {
            "status": "not_computed",
            "reason": f"reid_error: {exc}",
            "body_embedding": None,
            "note": "Reserved for body ReID embedding (OSNet or equivalent). Permanent identity is in face_embedding.",
        }
