import json
import shutil
from pathlib import Path


def _basenames(items) -> set[str]:
    out: set[str] = set()
    for raw in items or []:
        name = str(raw).replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            out.add(name)
    return out


def _prune_orphans(directory: Path, keep: set[str]) -> tuple[int, int]:
    """Delete files in `directory` whose basenames aren't in `keep`.
    Returns (files_deleted, bytes_freed).
    """
    if not directory.is_dir():
        return 0, 0
    deleted = 0
    freed = 0
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.name in keep:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            deleted += 1
            freed += size
        except OSError:
            pass
    return deleted, freed


def finalize(state: dict) -> dict:
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "profile.json"

    profile = state["profile"]
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"[finalize] profile saved -> {profile_path}")

    # Drop the staging tree wholesale — any reject crops live here.
    staging = output_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[finalize] removed staging tree: {staging}")

    # Defensive orphan sweep — catch anything that slipped through.
    referenced = (
        _basenames(profile.get("face_crops"))
        | _basenames(profile.get("body_crops"))
        | _basenames(profile.get("best_body_crops"))
    )
    for person in profile.get("people") or []:
        referenced |= (
            _basenames(person.get("face_crops"))
            | _basenames(person.get("body_crops"))
            | _basenames(person.get("best_body_crops"))
        )
    body_del, body_bytes = _prune_orphans(output_dir / "body_crops", referenced)
    face_del, face_bytes = _prune_orphans(output_dir / "face_crops", referenced)
    total_mb = (body_bytes + face_bytes) / (1024 * 1024)
    if body_del or face_del:
        print(
            f"[finalize] cleanup: deleted {body_del} orphan body / {face_del} orphan face "
            f"crops ({total_mb:.2f} MB freed)"
        )

    return {}
