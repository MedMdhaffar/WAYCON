from __future__ import annotations

import re
from pathlib import Path

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/]+(.*)$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return _PROJECT_ROOT


def _windows_to_wsl(raw: str) -> str | None:
    match = _WIN_DRIVE.match(raw)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _is_absolute_input(raw: str, normalized: str, wsl_path: str | None) -> bool:
    return bool(wsl_path) or normalized.startswith("/")


def resolve_path(raw: str, *, must_exist: bool = True, expect_file: bool | None = None) -> Path:
    """Resolve local development paths for the face engine.

    Relative paths are anchored at the repo root and may not escape it. Absolute
    local paths, including WSL `/mnt/...` paths and converted Windows drive
    paths, are allowed for trusted local use.
    """
    if raw is None:
        raise ValueError("path is required")
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        raise ValueError("path is empty")

    normalized = text.replace("\\", "/")
    wsl_path = _windows_to_wsl(text)
    absolute_input = _is_absolute_input(text, normalized, wsl_path)
    root = _PROJECT_ROOT.resolve()

    if absolute_input:
        candidate = Path(wsl_path or normalized).expanduser()
        resolved = candidate.resolve(strict=False)
    else:
        candidate = root / normalized
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"relative path escapes repo root: {raw}") from exc

    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist: {resolved.as_posix()}")
    if expect_file is True and resolved.exists() and not resolved.is_file():
        raise ValueError(f"path is not a file: {resolved.as_posix()}")
    if expect_file is False and resolved.exists() and not resolved.is_dir():
        raise ValueError(f"path is not a directory: {resolved.as_posix()}")
    return resolved


def resolve_file_path(raw: str) -> Path:
    return resolve_path(raw, must_exist=True, expect_file=True)


def resolve_output_dir(raw: str) -> Path:
    return resolve_path(raw, must_exist=False, expect_file=False)
