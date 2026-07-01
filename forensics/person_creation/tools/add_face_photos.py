"""Append pro-cam (iPhone/DSLR) face photos to an existing profile.

The face_embedding for any profile created from surveillance footage is weak —
low-res, poor angle, narrow temporal window. This tool lets you point at a
folder of high-quality face photos and have them added to the gallery, with
profile.face_embedding recomputed as the mean across the expanded set.

Usage:
    python -m forensics.person_creation.tools.add_face_photos \\
        --profile malek \\
        --images-dir /mnt/c/Users/malek/Desktop/iphone_face

Flags:
    --replace   Drop all existing face_crops; rebuild face_embedding from new
                photos only. Default is append.
    --dry-run   Print what would happen without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parents[2]))

from forensics.person_identifier.config import Config

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


class AddFacePhotosError(Exception):
    """Raised on unrecoverable issues (missing profile, bad images_dir, etc.)."""


def _crop(frame: np.ndarray, bbox: list[float], padding: int = 2) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def _largest_face(detections: list[dict]) -> dict | None:
    if not detections:
        return None
    def area(d):
        x1, y1, x2, y2 = d["bbox"]
        return max(0, x2 - x1) * max(0, y2 - y1)
    return max(detections, key=area)


def _resolve_crop_path(raw: str, profile_dir: Path) -> Path:
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    if "face" in raw.lower() or "face_crops" in raw.lower():
        return profile_dir / "face_crops" / name
    return profile_dir / "body_crops" / name


def add_face_photos(
    profile_name: str,
    images_dir: Path,
    replace: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the enrichment. Returns a structured result dict; raises
    AddFacePhotosError on hard failures (missing profile, bad images_dir).
    """
    cfg = Config.load()
    profile_dir = cfg.PROFILE_ROOT / profile_name
    profile_json_path = profile_dir / "profile.json"
    images_dir = Path(images_dir)

    if not profile_json_path.exists():
        raise AddFacePhotosError(f"profile.json not found at {profile_json_path}")
    if not images_dir.is_dir():
        raise AddFacePhotosError(f"images_dir is not a directory: {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    profile = json.loads(profile_json_path.read_text())
    face_crops_dir = profile_dir / "face_crops"

    mode = "replace" if replace else "append"
    existing_total = len(profile.get("face_crops", []) or [])
    skipped: list[dict] = []

    result_template = {
        "profile": profile_name,
        "images_dir": str(images_dir),
        "mode": mode,
        "candidates": len(images),
        "detected": 0,
        "skipped": skipped,
        "total_face_crops_after": existing_total,
        "embedding_norm": 0.0,
        "dry_run": bool(dry_run),
    }

    if not images:
        return result_template

    if not dry_run:
        face_crops_dir.mkdir(parents=True, exist_ok=True)

    # --- Lazy-load models (only when we will actually run detection) ---
    from forensics.person_creation.models.face_detector import get_face_detector
    from forensics.person_creation.models.face_embedder import get_face_embedder

    detector = get_face_detector()
    embedder = get_face_embedder()
    if not dry_run:
        detector.load(device=cfg.DEVICE)
        embedder.load()

    # --- Process each input image ---
    new_crop_names: list[str] = []
    for idx, src in enumerate(images):
        img = cv2.imread(str(src))
        if img is None:
            skipped.append({"file": src.name, "reason": "unreadable"})
            continue

        if dry_run:
            # No detection in dry-run, just count
            new_crop_names.append(f"iphone_{idx:03d}_{src.stem}.jpg")
            continue

        dets = detector.detect(img)
        face = _largest_face(dets)
        if face is None:
            skipped.append({"file": src.name, "reason": "no face detected"})
            continue

        face_crop = _crop(img, face["bbox"])
        if face_crop.size == 0:
            skipped.append({"file": src.name, "reason": "empty crop"})
            continue

        out_name = f"iphone_{idx:03d}_{src.stem}.jpg"
        cv2.imwrite(str(face_crops_dir / out_name), face_crop)
        new_crop_names.append(out_name)

    result_template["detected"] = len(new_crop_names)

    if dry_run:
        if replace:
            result_template["total_face_crops_after"] = len(new_crop_names)
        else:
            result_template["total_face_crops_after"] = existing_total + len(new_crop_names)
        return result_template

    if not new_crop_names:
        # Nothing detected — nothing to write
        return result_template

    # --- Update profile.json face_crops list ---
    new_paths_rel = [str(face_crops_dir / n) for n in new_crop_names]
    if replace:
        existing_basenames = {
            raw.replace("\\", "/").rsplit("/", 1)[-1]
            for raw in profile.get("face_crops", []) or []
        }
        for old_name in existing_basenames:
            if old_name in set(new_crop_names):
                continue
            old_path = face_crops_dir / old_name
            if old_path.exists():
                try:
                    old_path.unlink()
                except OSError:
                    pass
        profile["face_crops"] = new_paths_rel
    else:
        profile["face_crops"] = list(profile.get("face_crops", [])) + new_paths_rel

    # --- Recompute face_embedding from ALL current face_crops on disk ---
    all_embs: list[np.ndarray] = []
    missing = 0
    for raw in profile["face_crops"]:
        p = _resolve_crop_path(raw, profile_dir)
        if not p.exists():
            missing += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            missing += 1
            continue
        all_embs.append(np.asarray(embedder.embed(img), dtype=np.float32))

    if not all_embs:
        raise AddFacePhotosError("no usable face crops to embed after writing — refusing to corrupt profile.json")

    mean = np.mean(np.stack(all_embs), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-8:
        mean = mean / norm
    profile["face_embedding"] = mean.astype(float).tolist()
    profile["face_crop_count"] = len(all_embs)
    profile_json_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))

    result_template["total_face_crops_after"] = len(profile["face_crops"])
    result_template["embedding_norm"] = float(np.linalg.norm(mean))
    if missing:
        skipped.append({"file": "(internal)", "reason": f"{missing} face_crops missing on disk during re-embed"})
    return result_template


def _print_result(result: dict) -> None:
    print(f"[add_face_photos] profile: {result['profile']}")
    print(f"  images-dir   : {result['images_dir']}")
    print(f"  mode         : {result['mode'].upper()}")
    print(f"  candidates   : {result['candidates']}")
    print(f"  detected     : {result['detected']}")
    for s in result["skipped"]:
        print(f"    [skip] {s['file']}: {s['reason']}")
    print(f"  total face_crops after: {result['total_face_crops_after']}")
    if not result["dry_run"]:
        print(f"  embedding norm        : {result['embedding_norm']:.6f}")
    else:
        print("  [dry-run] no files written.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Add pro-cam face photos to an existing profile.")
    parser.add_argument("--profile", required=True, help="Profile name (folder under PROFILE_ROOT).")
    parser.add_argument("--images-dir", required=True, type=Path, help="Folder of high-quality face photos.")
    parser.add_argument("--replace", action="store_true", help="Replace existing face_crops instead of appending.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing.")
    args = parser.parse_args()
    try:
        result = add_face_photos(args.profile, args.images_dir, replace=args.replace, dry_run=args.dry_run)
    except AddFacePhotosError as exc:
        print(f"[add_face_photos] error: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
