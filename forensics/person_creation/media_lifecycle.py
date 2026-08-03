from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forensics.media_paths import (
    MediaPathError,
    get_media_root,
    normalize_media_path,
    resolve_media_path,
)


MEDIA_VALUE_KEYS = frozenset({
    "path",
    "crop_path",
    "face_path",
    "body_path",
    "face_crop_path",
    "body_crop_path",
    "representative_face_path",
    "profile_image",
    "best_face_crop",
    "selected_body_crop",
})
MEDIA_LIST_KEYS = frozenset({"face_crops", "body_crops", "best_body_crops"})


class MediaRelocationError(RuntimeError):
    """A retained crop could not be relocated without risking data loss."""


@dataclass(frozen=True)
class RelocatedMedia:
    profile: dict[str, Any]
    remap: dict[str, str]
    cleanup_pairs: tuple[tuple[str, str], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_readable(path: Path) -> None:
    if not path.is_file():
        raise MediaRelocationError(f"media destination is missing: {path.name}")
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise MediaRelocationError(
            f"media destination is not readable: {path.name}"
        ) from exc


def _same_file_content(left: Path, right: Path) -> bool:
    try:
        return left.stat().st_size == right.stat().st_size and _file_sha256(left) == _file_sha256(right)
    except OSError as exc:
        raise MediaRelocationError("could not validate a media collision") from exc


def copy_media_for_handoff(
    source: str | Path,
    destination: str | Path,
    *,
    media_root: str | Path | None = None,
) -> tuple[str, tuple[str, str] | None]:
    """Copy one in-root image, validate it, and retain the source for later cleanup."""
    root = Path(media_root) if media_root is not None else get_media_root()
    try:
        source_relative = normalize_media_path(
            source,
            media_root=root,
            allow_legacy_absolute=True,
            require_exists=False,
        )
        destination_relative = normalize_media_path(
            destination,
            media_root=root,
            allow_legacy_absolute=True,
            require_exists=False,
        )
        source_path = resolve_media_path(
            source_relative,
            media_root=root,
            allow_legacy_absolute=False,
            require_exists=False,
            image_only=True,
        )
        destination_path = resolve_media_path(
            destination_relative,
            media_root=root,
            allow_legacy_absolute=False,
            require_exists=False,
            image_only=True,
        )
    except (MediaPathError, OSError) as exc:
        raise MediaRelocationError("media relocation escaped the configured root") from exc

    if source_path == destination_path:
        _verified_readable(destination_path)
        return destination_relative, None

    source_exists = source_path.is_file()
    destination_exists = destination_path.is_file()
    if not source_exists and not destination_exists:
        raise MediaRelocationError(
            f"media source and destination are both missing: {source_path.name}"
        )
    if destination_exists:
        _verified_readable(destination_path)
        if source_exists and not _same_file_content(source_path, destination_path):
            raise MediaRelocationError(
                f"media destination collision differs from source: {destination_path.name}"
            )
        return destination_relative, (source_relative, destination_relative) if source_exists else None

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.parent / f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(source_path, temporary)
        _verified_readable(temporary)
        if not _same_file_content(source_path, temporary):
            raise MediaRelocationError(
                f"media copy verification failed: {destination_path.name}"
            )
        os.replace(temporary, destination_path)
        _verified_readable(destination_path)
        if not _same_file_content(source_path, destination_path):
            raise MediaRelocationError(
                f"media destination verification failed: {destination_path.name}"
            )
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination_relative, (source_relative, destination_relative)


def rewrite_media_references(value: Any, remap: dict[str, str]) -> Any:
    """Deep-copy a state/report and replace every exact relocated reference."""
    if isinstance(value, dict):
        rewritten: dict[Any, Any] = {}
        for key, item in value.items():
            new_key = remap.get(key, key) if isinstance(key, str) else key
            rewritten[new_key] = rewrite_media_references(item, remap)
        return rewritten
    if isinstance(value, list):
        return [rewrite_media_references(item, remap) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite_media_references(item, remap) for item in value)
    if isinstance(value, str):
        return remap.get(value, value)
    return deepcopy(value)


def _is_obsolete_session_reference(value: str, output_dir: Path, media_root: Path) -> bool:
    try:
        relative = normalize_media_path(
            value,
            media_root=media_root,
            allow_legacy_absolute=True,
            require_exists=False,
        )
        output_relative = normalize_media_path(
            output_dir,
            media_root=media_root,
            allow_legacy_absolute=True,
            require_exists=False,
        ).rstrip("/")
    except (MediaPathError, OSError):
        return True
    if not relative.startswith(output_relative + "/"):
        return False
    suffix = relative[len(output_relative) + 1 :]
    return suffix.startswith("_staging/") or suffix.startswith("cluster_")


def scrub_obsolete_session_media(
    value: Any,
    *,
    output_dir: str | Path,
    media_root: str | Path | None = None,
    parent_key: str = "",
) -> Any:
    """Remove only path-valued references to disposable staging/cluster media."""
    root = Path(media_root) if media_root is not None else get_media_root()
    output = Path(output_dir)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in MEDIA_VALUE_KEYS and isinstance(item, str):
                cleaned[key] = None if _is_obsolete_session_reference(item, output, root) else item
            elif key in MEDIA_LIST_KEYS and isinstance(item, list):
                cleaned[key] = [
                    raw for raw in item
                    if not isinstance(raw, str)
                    or not _is_obsolete_session_reference(raw, output, root)
                ]
            else:
                cleaned[key] = scrub_obsolete_session_media(
                    item,
                    output_dir=output,
                    media_root=root,
                    parent_key=key,
                )
        return cleaned
    if isinstance(value, list):
        return [
            scrub_obsolete_session_media(
                item,
                output_dir=output,
                media_root=root,
                parent_key=parent_key,
            )
            for item in value
        ]
    return deepcopy(value)


def relocate_profile_media(
    profile: dict[str, Any],
    person_id: str,
    *,
    media_root: str | Path | None = None,
) -> RelocatedMedia:
    """Copy only retained profile evidence into one canonical person directory."""
    root = Path(media_root) if media_root is not None else get_media_root()
    remap: dict[str, str] = {}
    cleanup: list[tuple[str, str]] = []

    def relocate(raw: str, crop_type: str) -> str:
        if raw in remap:
            return remap[raw]
        destination = f"{person_id}/{crop_type}_crops/{Path(raw).name}"
        canonical, pair = copy_media_for_handoff(raw, destination, media_root=root)
        remap[raw] = canonical
        try:
            old_relative = normalize_media_path(
                raw,
                media_root=root,
                allow_legacy_absolute=True,
                require_exists=False,
            )
            remap[old_relative] = canonical
        except (MediaPathError, OSError):
            pass
        if pair is not None and pair not in cleanup:
            cleanup.append(pair)
        return canonical

    rewritten = deepcopy(profile)
    for field, crop_type in (
        ("face_crops", "face"),
        ("body_crops", "body"),
        ("best_body_crops", "body"),
    ):
        values = []
        for raw in rewritten.get(field) or []:
            if raw:
                canonical = relocate(str(raw), crop_type)
                if canonical not in values:
                    values.append(canonical)
        rewritten[field] = values

    rewritten = rewrite_media_references(rewritten, remap)
    if rewritten.get("face_crops"):
        rewritten["profile_image"] = rewritten["face_crops"][0]
    return RelocatedMedia(rewritten, remap, tuple(cleanup))


def cleanup_relocated_sources(
    pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    *,
    media_root: str | Path | None = None,
) -> list[str]:
    """Delete verified obsolete copies; canonical destinations are never removed."""
    root = Path(media_root) if media_root is not None else get_media_root()
    removed: list[str] = []
    for source, destination in pairs:
        source_path = resolve_media_path(
            source,
            media_root=root,
            allow_legacy_absolute=False,
            require_exists=False,
            image_only=True,
        )
        destination_path = resolve_media_path(
            destination,
            media_root=root,
            allow_legacy_absolute=False,
            require_exists=True,
            image_only=True,
        )
        if source_path == destination_path or not source_path.exists():
            continue
        if not _same_file_content(source_path, destination_path):
            raise MediaRelocationError(
                f"refusing to remove unverified source: {source_path.name}"
            )
        source_path.unlink()
        removed.append(source)
    return removed
