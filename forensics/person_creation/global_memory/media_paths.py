from __future__ import annotations

import re
from pathlib import Path

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/]+(.*)$")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _windows_to_wsl(raw: str) -> str | None:
    match = _WIN_DRIVE.match(raw)
    if not match:
        return None
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def resolve_media_path(raw: str | None, project_root: Path | None = None) -> Path | None:
    if not raw:
        return None

    root = project_root or _PROJECT_ROOT
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        return None

    candidates: list[Path] = []
    normalized = text.replace("\\", "/")

    # Native path first. On Windows this handles C:/...; on WSL this handles
    # /mnt/c/... and repo-relative POSIX paths.
    candidates.append(Path(normalized))

    wsl_path = _windows_to_wsl(text)
    if wsl_path:
        candidates.append(Path(wsl_path))

    if not Path(normalized).is_absolute():
        candidates.append(root / normalized)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def media_path_for_api(raw: str | None, project_root: Path | None = None) -> str | None:
    path = resolve_media_path(raw, project_root=project_root)
    return path.as_posix() if path else None
