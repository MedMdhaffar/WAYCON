"""Promote human-confirmed crops out of `_staging/` to permanent storage.

Runs immediately after `human_in_the_loop`. Anything in `associations` (a
confirmed face<->body pair) is moved from `_staging/{body,face}_crops/` to
`{body,face}_crops/`. Unpaired crops stay in `_staging/` and are removed by
`finalize`.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _move_path(src: Path, dst_dir: Path) -> Path:
    """Move src into dst_dir keeping the filename. If dst already exists, the
    src is assumed to be a duplicate already-promoted path and is returned
    unchanged.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not src.exists():
        return dst if dst.exists() else src
    if dst.exists():
        # Already promoted on a prior pass — drop the staging copy.
        try:
            src.unlink()
        except OSError:
            pass
        return dst
    shutil.move(str(src), str(dst))
    return dst


def promote_crops(state: dict) -> dict:
    output_dir = Path(state["output_dir"])
    body_dst = output_dir / "body_crops"
    face_dst = output_dir / "face_crops"

    associations = state.get("associations") or []
    quality_face_crops = list(state.get("quality_face_crops") or [])
    quality_body_crops = list(state.get("quality_body_crops") or [])

    # Build a path-remap so downstream nodes see permanent paths.
    remap: dict[str, str] = {}

    def promote(raw_path: str, dst_dir: Path) -> str:
        if not raw_path:
            return raw_path
        if raw_path in remap:
            return remap[raw_path]
        new = _move_path(Path(raw_path), dst_dir)
        remap[raw_path] = str(new)
        return remap[raw_path]

    promoted_assoc: list[dict] = []
    for a in associations:
        a = dict(a)  # shallow copy; don't mutate caller state
        if a.get("face_path"):
            a["face_path"] = promote(a["face_path"], face_dst)
        if a.get("body_path"):
            a["body_path"] = promote(a["body_path"], body_dst)
        promoted_assoc.append(a)

    # For quality_*_crops, only promote those that appear in associations.
    # Anything not in associations is a reject — leave it in staging for
    # finalize to clean up.
    promoted_paths = set(remap.keys()) | set(remap.values())

    def filter_and_remap(items: list[dict]) -> list[dict]:
        out: list[dict] = []
        for c in items:
            p = c.get("path")
            if p in remap:
                c = dict(c)
                c["path"] = remap[p]
                out.append(c)
            elif p in promoted_paths:
                out.append(c)
        return out

    promoted_face = filter_and_remap(quality_face_crops)
    promoted_body = filter_and_remap(quality_body_crops)

    promoted_tracks: list[dict] = []
    for track in state.get("person_tracks") or []:
        track = dict(track)
        track["associations"] = [
            {
                **assoc,
                "face_path": remap.get(assoc.get("face_path"), assoc.get("face_path")),
                "body_path": remap.get(assoc.get("body_path"), assoc.get("body_path")),
            }
            for assoc in track.get("associations", [])
        ]
        track["face_paths"] = [
            remap.get(path, path)
            for path in track.get("face_paths", [])
        ]
        track["body_paths"] = [
            remap.get(path, path)
            for path in track.get("body_paths", [])
        ]
        promoted_tracks.append(track)

    print(
        f"[promote_crops] moved "
        f"{sum(1 for v in remap.values() if Path(v).parent.name == 'face_crops')} face, "
        f"{sum(1 for v in remap.values() if Path(v).parent.name == 'body_crops')} body "
        f"to permanent storage; {len(quality_face_crops) - len(promoted_face)} face / "
        f"{len(quality_body_crops) - len(promoted_body)} body rejects left in _staging/"
    )

    out = {
        "associations": promoted_assoc,
        "quality_face_crops": promoted_face,
        "quality_body_crops": promoted_body,
    }
    if promoted_tracks:
        out["person_tracks"] = promoted_tracks
    return out
