from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


COLOR_EXTRACTOR_NAME = "DominantColorExtractor_v1"
SIGNAL_TYPE = "same_day_supporting_appearance"
_REGIONS = {
    "top": (0.18, 0.55),
    "bottom": (0.50, 0.82),
    "shoes": (0.78, 1.00),
}


def rgb_to_basic_color_name(rgb: list[int] | tuple[int, int, int]) -> str:
    r, g, b = [int(v) for v in rgb]
    color = np.uint8([[[r, g, b]]])
    h, s, v = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]
    h = int(h) * 2
    s = int(s)
    v = int(v)

    if v < 45:
        return "black"
    if s < 28 and v > 220:
        return "white"
    if s < 35:
        return "gray"
    if 20 <= h < 55 and 45 <= v <= 225 and r > g and g > b:
        return "brown"
    if 35 <= h < 70 and s < 95 and v > 145:
        return "beige"
    if 200 <= h < 250 and v < 95:
        return "navy"
    if h < 15 or h >= 345:
        return "pink" if v > 170 and s < 120 else "red"
    if 15 <= h < 40:
        return "orange"
    if 40 <= h < 70:
        return "yellow"
    if 70 <= h < 165:
        return "green"
    if 165 <= h < 195:
        return "cyan"
    if 195 <= h < 255:
        return "blue"
    if 255 <= h < 310:
        return "purple"
    if 310 <= h < 345:
        return "pink"
    return "unknown"


def _empty_aggregate(source_crops: list[str], error: str | None = None) -> dict:
    result = {
        "extractor": COLOR_EXTRACTOR_NAME,
        "source_crops": source_crops,
        "top": None,
        "bottom": None,
        "shoes": None,
        "per_crop_count": 0,
        "signal_type": SIGNAL_TYPE,
    }
    if error:
        result["error"] = error
    return result


class ColorSignalExtractor:
    """Deterministic same-day clothing color signal extractor.

    Colors complement clothing/ReID for the current day only. They are not a
    biometric identity signal and must not replace the face embedding anchor.
    """

    EXTRACTOR_NAME = COLOR_EXTRACTOR_NAME

    def extract_image(self, path: str) -> dict | None:
        crop_path = Path(path)
        if not crop_path.exists():
            print(f"[ColorSignalExtractor][warn] missing crop skipped: {path}")
            return None

        try:
            image_bgr = cv2.imread(str(crop_path.resolve()))
        except Exception as exc:
            print(f"[ColorSignalExtractor][warn] failed to read {path}: {exc}")
            return None
        if image_bgr is None or image_bgr.size == 0:
            print(f"[ColorSignalExtractor][warn] unreadable crop skipped: {path}")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = {"crop": path}
        for region_name, (y0, y1) in _REGIONS.items():
            result[region_name] = self._extract_region(image_rgb, y0, y1)
        return result

    def extract_batch(self, paths: list[str]) -> list[dict]:
        per_crop = []
        for path in paths:
            extracted = self.extract_image(path)
            if extracted:
                per_crop.append(extracted)
        return per_crop

    def _extract_region(self, image_rgb: np.ndarray, y0: float, y1: float) -> dict:
        h, w = image_rgb.shape[:2]
        x_start = int(0.15 * w)
        x_end = max(x_start + 1, int(0.85 * w))
        y_start = int(y0 * h)
        y_end = max(y_start + 1, int(y1 * h))
        region = image_rgb[y_start:y_end, x_start:x_end]
        pixels = region.reshape(-1, 3)
        palette = self._palette(pixels)
        return {
            "dominant": palette[0] if palette else None,
            "palette": palette,
        }

    def _palette(self, pixels: np.ndarray, k: int = 3) -> list[dict]:
        if pixels.size == 0:
            return []

        samples = pixels.astype(np.float32)
        if len(samples) > 12000:
            step = max(1, len(samples) // 12000)
            samples = samples[::step]

        cluster_count = min(k, len(samples))
        if cluster_count <= 0:
            return []
        if cluster_count == 1:
            centers = samples[:1]
            labels = np.zeros((len(samples),), dtype=np.int32)
        else:
            cv2.setRNGSeed(7)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
            _compactness, labels, centers = cv2.kmeans(
                samples,
                cluster_count,
                None,
                criteria,
                3,
                cv2.KMEANS_PP_CENTERS,
            )
            labels = labels.reshape(-1)

        counts = np.bincount(labels, minlength=cluster_count)
        order = np.argsort(counts)[::-1]
        total = max(float(counts.sum()), 1.0)
        palette = []
        for idx in order:
            rgb = np.clip(np.rint(centers[idx]), 0, 255).astype(int).tolist()
            palette.append({
                "name": rgb_to_basic_color_name(rgb),
                "rgb": rgb,
                "coverage": round(float(counts[idx]) / total, 4),
            })
        return palette


def aggregate_color_signals(per_crop_colors: list[dict]) -> dict:
    source_crops = [item["crop"] for item in per_crop_colors if item.get("crop")]
    if not per_crop_colors:
        return _empty_aggregate(source_crops, "No valid body crops for color extraction")

    result = {
        "extractor": COLOR_EXTRACTOR_NAME,
        "source_crops": source_crops,
        "per_crop_count": len(per_crop_colors),
        "signal_type": SIGNAL_TYPE,
    }

    for region in _REGIONS:
        scores: dict[str, float] = defaultdict(float)
        votes: dict[str, int] = defaultdict(int)
        rgb_weighted: dict[str, list[np.ndarray]] = defaultdict(list)
        for crop in per_crop_colors:
            dominant = (crop.get(region) or {}).get("dominant")
            if not dominant:
                continue
            name = dominant.get("name", "unknown")
            coverage = float(dominant.get("coverage") or 0.0)
            scores[name] += coverage
            votes[name] += 1
            rgb_weighted[name].append(np.asarray(dominant.get("rgb") or [0, 0, 0], dtype=np.float32) * coverage)

        if not scores:
            result[region] = None
            continue

        winner = max(scores, key=scores.get)
        total_score = max(sum(scores.values()), 1e-8)
        winner_score = scores[winner]
        rgb_total = np.sum(rgb_weighted[winner], axis=0)
        rgb_mean = np.clip(np.rint(rgb_total / max(winner_score, 1e-8)), 0, 255).astype(int).tolist()
        result[region] = {
            "dominant": winner,
            "confidence": round(float(winner_score / total_score), 4),
            "rgb_mean": rgb_mean,
            "votes": {name: votes[name] for name in sorted(votes)},
        }

    return result


_instance = ColorSignalExtractor()


def get_color_extractor() -> ColorSignalExtractor:
    return _instance
