from __future__ import annotations

import os


# DBSCAN identity-clustering thresholds. Each value resolves in order:
# state["identity_clustering_config"], then env override, then default.
_DEFAULTS = {
    "eps": 0.4,                   # DBSCAN cosine-distance radius
    "min_samples": 3,            # DBSCAN core-point neighbour count
    "min_cluster_face_count": 3,  # below this, a cluster is flagged low_confidence
}


def _env_key(name: str) -> str:
    return f"PERSON_CREATION_IDENTITY_{name.upper()}"


def _coerce(name: str, value):
    default = _DEFAULTS[name]
    if isinstance(default, int):
        return int(value)
    return float(value)


def load_identity_config(state: dict) -> dict:
    supplied = state.get("identity_clustering_config") or {}
    cfg = {}
    for name, default in _DEFAULTS.items():
        raw = supplied.get(name, os.getenv(_env_key(name), default))
        cfg[name] = _coerce(name, raw)
    return cfg
