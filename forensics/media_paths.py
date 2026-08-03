from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:/")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MEDIA_MARKER = "/forensics/person_db/"


class MediaPathError(ValueError):
    """A media reference is malformed or escapes the configured media root."""


class UnsupportedMediaTypeError(MediaPathError):
    """A media reference uses a file type that the image API does not serve."""


def get_media_root(root: str | os.PathLike[str] | None = None) -> Path:
    configured = root if root is not None else os.getenv(
        "PERSON_CREATION_MEDIA_ROOT",
        "forensics/person_db",
    )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _portable_text(path: str | os.PathLike[str]) -> str:
    value = str(path).strip().replace("\\", "/")
    while "//" in value and not value.startswith("//"):
        value = value.replace("//", "/")
    return value


def _relative_text(
    path: str | os.PathLike[str],
    *,
    root: Path,
    allow_legacy_absolute: bool,
) -> str:
    value = _portable_text(path)
    if not value or "\x00" in value:
        raise MediaPathError("media path is required")
    if _URI_SCHEME.match(value) and not _WINDOWS_DRIVE.match(value):
        raise MediaPathError("media URLs are not allowed")

    root_text = root.as_posix().rstrip("/")
    lower_value = value.lower()
    lower_root = root_text.lower()
    is_absolute = value.startswith("/") or bool(_WINDOWS_DRIVE.match(value))

    if is_absolute:
        if not allow_legacy_absolute:
            raise MediaPathError("media path must be relative")
        if lower_value == lower_root:
            raise MediaPathError("media path must identify a file")
        if lower_value.startswith(lower_root + "/"):
            value = value[len(root_text) + 1 :]
        elif _MEDIA_MARKER in lower_value:
            marker_index = lower_value.index(_MEDIA_MARKER)
            value = value[marker_index + len(_MEDIA_MARKER) :]
        else:
            raise MediaPathError("absolute media path is outside the configured root")
    else:
        prefixes = (
            "forensics/person_db/",
            "person_db/",
        )
        for prefix in prefixes:
            if lower_value.startswith(prefix):
                value = value[len(prefix) :]
                break

    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise MediaPathError("media path must be relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise MediaPathError("media path traversal is not allowed")
    return pure.as_posix()


def normalize_media_path(
    path: str | os.PathLike[str],
    *,
    media_root: str | os.PathLike[str] | None = None,
    allow_legacy_absolute: bool = True,
    require_exists: bool = False,
) -> str:
    root = get_media_root(media_root)
    relative = _relative_text(
        path,
        root=root,
        allow_legacy_absolute=allow_legacy_absolute,
    )
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MediaPathError("media path escapes the configured root") from exc
    if require_exists and (not resolved.exists() or not resolved.is_file()):
        raise FileNotFoundError("media file was not found")
    return relative


def resolve_media_path(
    path: str | os.PathLike[str],
    *,
    media_root: str | os.PathLike[str] | None = None,
    allow_legacy_absolute: bool = False,
    require_exists: bool = True,
    image_only: bool = False,
) -> Path:
    root = get_media_root(media_root)
    relative = normalize_media_path(
        path,
        media_root=root,
        allow_legacy_absolute=allow_legacy_absolute,
        require_exists=False,
    )
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=require_exists)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError("media file was not found") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MediaPathError("media path escapes the configured root") from exc
    if require_exists and not resolved.is_file():
        raise FileNotFoundError("media file was not found")
    if image_only and resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise UnsupportedMediaTypeError("unsupported image type")
    return resolved


def is_safe_media_path(
    path: str | os.PathLike[str],
    *,
    media_root: str | os.PathLike[str] | None = None,
    require_exists: bool = False,
) -> bool:
    try:
        resolve_media_path(
            path,
            media_root=media_root,
            require_exists=require_exists,
        )
        return True
    except (MediaPathError, FileNotFoundError, OSError):
        return False
