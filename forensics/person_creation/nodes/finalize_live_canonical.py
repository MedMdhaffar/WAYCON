"""Serialize the canonical rolling state without starting a second pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from forensics.person_creation.nodes.finalize import _write_json


def finalize_live_canonical(state: dict) -> dict:
    """Write live-built profiles and a session report, without recomputation.

    Durable Global Memory mutations have already happened in the rolling
    identity coordinator under its stability policy. This node deliberately
    performs no filtering, embedding, clustering, matching, ReID, VLM, or
    registration.
    """
    if not state.get("_canonical_live_state"):
        raise RuntimeError("canonical live state is required")

    started = time.monotonic()
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = deepcopy(state.get("per_cluster_profiles") or {})
    identities = deepcopy(state.get("_canonical_live_identities") or [])
    profile_root = output_dir / "_live_profiles"
    for raw_cluster_id, raw_profile in profiles.items():
        profile = deepcopy(raw_profile)
        profile.pop("face_embedding", None)
        _write_json(
            profile_root / f"cluster_{int(raw_cluster_id)}" / "profile.json",
            profile,
        )

    report_path = output_dir / "canonical_live_session.json"
    reconciliation_ms = 0.0
    serialization_ms = round((time.monotonic() - started) * 1000.0, 3)
    stop_timings = deepcopy(
        (state.get("stream_stats") or {}).get("stop_timings") or {}
    )
    stop_timings.update({
        "canonical_reconciliation_ms": reconciliation_ms,
        "serialization_ms": serialization_ms,
    })
    if not report_path.exists():
        _write_json(report_path, {
            "source_type": "live_camera",
            "camera_id": state.get("camera_id"),
            "effective_configuration": deepcopy(
                state.get("effective_configuration") or {}
            ),
            "live_identities": identities,
            "identity_clusters": deepcopy(state.get("identity_clusters") or []),
            "unresolved_faces": deepcopy(state.get("unresolved_faces") or []),
            "unattached_bodies": deepcopy(state.get("unattached_bodies") or []),
            "reid_reasons": deepcopy(state.get("reid_reasons") or {}),
            "live_identity_decisions": deepcopy(
                state.get("live_identity_decisions") or []
            ),
            "stream_stats": deepcopy(state.get("stream_stats") or {}),
            "stop_stage_durations": stop_timings,
        })
    first_id = sorted(profiles, key=int)[0] if profiles else None
    return {
        "per_cluster_profiles": profiles,
        "profile": profiles.get(first_id, {}) if first_id is not None else {},
        "canonical_live_report_path": str(report_path),
        "live_finalization_timings": {
            "reconciliation_ms": reconciliation_ms,
            "profile_and_report_serialization_ms": serialization_ms,
        },
    }
