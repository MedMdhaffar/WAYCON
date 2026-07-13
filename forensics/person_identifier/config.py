from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Minimal config used by person_creation's profile-management endpoints."""

    PROFILE_ROOT: Path
    DEVICE: str = "cpu"

    @classmethod
    def load(cls) -> "Config":
        profile_root = Path(os.getenv("FORENSICS_PROFILE_ROOT", "forensics/person_db"))
        device = os.getenv("FORENSICS_DEVICE", "cpu")
        return cls(PROFILE_ROOT=profile_root, DEVICE=device)
