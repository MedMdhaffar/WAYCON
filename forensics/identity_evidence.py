"""Stable identifiers for face and body identity evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from forensics.media_paths import resolve_media_path


EVIDENCE_CROP_TYPES = frozenset({"face", "body"})


def identity_evidence_key(
    path: str | Path,
    crop_type: str,
    *,
    media_root: str | Path | None = None,
    allow_legacy_absolute: bool = True,
) -> str:
    """Return ``crop_type + SHA-256(contents)`` for one readable crop.

    The path is intentionally absent from the result, so the identifier is
    unchanged when evidence moves through staging, cluster, and canonical
    person directories.
    """
    normalized_type = str(crop_type).strip().lower()
    if normalized_type not in EVIDENCE_CROP_TYPES:
        raise ValueError("crop_type must be 'face' or 'body'")

    raw_path = Path(path)
    if raw_path.is_absolute():
        resolved = raw_path.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError("identity evidence file was not found")
    else:
        resolved = resolve_media_path(
            path,
            media_root=media_root,
            allow_legacy_absolute=allow_legacy_absolute,
            require_exists=True,
        )

    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{normalized_type}:{digest.hexdigest()}"
