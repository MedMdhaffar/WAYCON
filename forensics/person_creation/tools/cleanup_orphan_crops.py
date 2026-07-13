"""Delete crop files in a profile's body_crops/ and face_crops/ folders that
are NOT referenced in profile.json.

Background: process_video.py used to write every detected body/face to disk
before human review. Rejected bystander crops linger on disk even though
profile.json only references the human-confirmed ones. This tool reconciles
the two — on-disk minus referenced = orphans, which get deleted.

Usage:
    python -m forensics.person_creation.tools.cleanup_orphan_crops --profile malek
    python -m forensics.person_creation.tools.cleanup_orphan_crops --profile malek --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parents[2]))

from forensics.person_identifier.config import Config


class CleanupError(Exception):
    """Raised when the cleanup cannot proceed (e.g. profile.json missing)."""


def _referenced_basenames(profile: dict) -> set[str]:
    """Collect filename stems (basenames) from face_crops, body_crops,
    best_body_crops. We compare by basename to sidestep Windows-vs-WSL
    path normalization — each crop filename is unique within a profile."""
    out: set[str] = set()
    for key in ("face_crops", "body_crops", "best_body_crops"):
        for raw in profile.get(key, []) or []:
            name = raw.replace("\\", "/").rsplit("/", 1)[-1]
            if name:
                out.add(name)
    return out


def _list_jpgs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def cleanup(profile_name: str, dry_run: bool = False) -> dict:
    """Return a structured result dict. Raises CleanupError if profile missing."""
    cfg = Config.load()
    profile_dir = cfg.PROFILE_ROOT / profile_name
    profile_json_path = profile_dir / "profile.json"

    if not profile_json_path.exists():
        raise CleanupError(f"profile.json not found at {profile_json_path}")

    profile = json.loads(profile_json_path.read_text())
    referenced = _referenced_basenames(profile)

    body_dir = profile_dir / "body_crops"
    face_dir = profile_dir / "face_crops"
    body_files = _list_jpgs(body_dir)
    face_files = _list_jpgs(face_dir)

    body_orphans = [p for p in body_files if p.name not in referenced]
    face_orphans = [p for p in face_files if p.name not in referenced]
    body_referenced = len(body_files) - len(body_orphans)
    face_referenced = len(face_files) - len(face_orphans)

    total_bytes = sum(p.stat().st_size for p in body_orphans + face_orphans)
    mb = total_bytes / (1024 * 1024)

    deleted = 0
    if not dry_run:
        for p in body_orphans + face_orphans:
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass

    return {
        "profile": profile_name,
        "body_crops": {
            "on_disk": len(body_files),
            "referenced": body_referenced,
            "orphans": len(body_orphans),
        },
        "face_crops": {
            "on_disk": len(face_files),
            "referenced": face_referenced,
            "orphans": len(face_orphans),
        },
        "mb_to_free": round(mb, 4),
        "deleted": deleted,
        "dry_run": bool(dry_run),
    }


def _print_result(result: dict) -> None:
    body = result["body_crops"]
    face = result["face_crops"]
    print(f"[cleanup] profile: {result['profile']}")
    print(f"  body_crops/  : {body['on_disk']:>4} on disk, {body['referenced']:>4} referenced -> {body['orphans']} orphans")
    print(f"  face_crops/  : {face['on_disk']:>4} on disk, {face['referenced']:>4} referenced -> {face['orphans']} orphans")
    print(f"  disk to free : {result['mb_to_free']:.2f} MB")
    if result["dry_run"]:
        print("  [dry-run] no files deleted.")
    elif result["deleted"]:
        print(f"  deleted {result['deleted']} files ({result['mb_to_free']:.2f} MB freed).")
    else:
        print("  nothing to clean up.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete orphan crops not referenced in profile.json.")
    parser.add_argument("--profile", required=True, help="Profile name (folder under PROFILE_ROOT).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without writing.")
    args = parser.parse_args()
    try:
        result = cleanup(args.profile, dry_run=args.dry_run)
    except CleanupError as exc:
        print(f"[cleanup] error: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
