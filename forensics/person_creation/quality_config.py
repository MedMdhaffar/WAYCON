from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class QualityFilterConfig:
    face_min_width: int = 48
    face_min_height: int = 48
    face_min_sharpness: float = 45.0
    body_min_height: int = 80
    body_min_area: int = 3000
    body_min_sharpness: float = 50.0

    def validate(self) -> QualityFilterConfig:
        if self.face_min_width <= 0 or self.face_min_height <= 0:
            raise ValueError("Face minimum dimensions must be positive.")
        if self.body_min_height <= 0 or self.body_min_area <= 0:
            raise ValueError("Body minimum dimensions must be positive.")
        if self.face_min_sharpness < 0 or self.body_min_sharpness < 0:
            raise ValueError("Minimum sharpness must be non-negative.")
        return self

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


DEFAULT_QUALITY_FILTER_CONFIG = QualityFilterConfig()
_CONFIG_FIELDS = tuple(QualityFilterConfig.__dataclass_fields__)
_FIXED_FACE_FIELDS = {
    "face_min_width",
    "face_min_height",
    "face_min_sharpness",
}


def _coerce(name: str, value: Any) -> int | float:
    default = getattr(DEFAULT_QUALITY_FILTER_CONFIG, name)
    try:
        return int(value) if isinstance(default, int) else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid quality-filter value for {name}: {value!r}") from exc


def load_quality_filter_config(
    overrides: Mapping[str, Any] | None = None,
) -> QualityFilterConfig:
    """Resolve one authoritative configuration.

    The recognition face boundary is a product invariant, not a deployment
    tuning knob.  Keeping those three values fixed prevents a long-lived
    service or worker environment from silently restoring the retired 60px
    boundary.  Body settings retain their existing environment overrides.
    """
    supplied = dict(overrides or {})
    unknown = sorted(set(supplied) - set(_CONFIG_FIELDS))
    if unknown:
        raise ValueError(
            "Unknown quality-filter setting(s): " + ", ".join(unknown)
        )

    values: dict[str, int | float] = {}
    for name in _CONFIG_FIELDS:
        env_key = f"PERSON_CREATION_{name.upper()}"
        if name in _FIXED_FACE_FIELDS:
            raw = supplied.get(name, getattr(DEFAULT_QUALITY_FILTER_CONFIG, name))
        else:
            raw = supplied.get(
                name,
                os.getenv(env_key, getattr(DEFAULT_QUALITY_FILTER_CONFIG, name)),
            )
        values[name] = _coerce(name, raw)
    return QualityFilterConfig(**values).validate()
