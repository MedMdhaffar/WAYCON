"""Small path helpers used by the person_creation service layer."""

from __future__ import annotations

import re

# Match Windows drive prefix like C:\ or c:/. Anything after the drive letter
# (with one or more slash/backslash separators) is captured as the remainder.
_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/]+(.*)$")


def to_wsl_path(raw: str) -> str:
    """Best-effort convert a Windows-style path to a WSL `/mnt/<drive>/...`
    path. Idempotent: already-Linux paths pass through unchanged. Operators
    sometimes paste paths surrounded by quotes from File Explorer — strip those.
    UNC paths (`\\\\server\\share`) are left alone; we don't handle them.
    """
    if not raw:
        return raw
    s = raw.strip().strip('"').strip("'")
    if not s:
        return s
    if s.startswith("/"):
        return s
    m = _WIN_DRIVE.match(s)
    if not m:
        return s
    drive, rest = m.group(1).lower(), m.group(2)
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive}/{rest}"
