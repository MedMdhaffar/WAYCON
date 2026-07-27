from __future__ import annotations

import hashlib
from pathlib import Path

from forensics.person_creation.graph import route_after_live_capture
from forensics.person_creation.nodes.finalize_live_canonical import (
    finalize_live_canonical,
)


def _state(tmp_path: Path) -> dict:
    return {
        "_canonical_live_state": True,
        "_canonical_live_identities": [{
            "live_identity_id": "live_0001",
            "cluster_label": 0,
            "candidate_person_id": "person_007",
        }],
        "output_dir": str(tmp_path / "session"),
        "source_type": "live_camera",
        "camera_id": "camera-1",
        "effective_configuration": {"live_process_every_n_frames": 3},
        "identity_clusters": [{
            "cluster_id": 0,
            "face_records": [{"crop_path": "session/_staging/face.jpg"}],
        }],
        "unresolved_faces": [],
        "unattached_bodies": [],
        "reid_reasons": {0: "no_reid_model_configured"},
        "live_identity_decisions": [],
        "stream_stats": {"processed_frames": 3},
        "per_cluster_profiles": {
            0: {
                "id": "person_session_cluster_0",
                "face_embedding": [1.0, 0.0],
                "face_crops": ["session/_staging/face.jpg"],
            },
        },
    }


def test_live_graph_routes_canonical_state_away_from_batch_tail(tmp_path):
    state = _state(tmp_path)

    assert route_after_live_capture(state) == "finalize_live_canonical"
    assert route_after_live_capture({"source_type": "live_camera"}) == "filter_quality"


def test_canonical_finalization_is_serialization_only_and_idempotent(tmp_path):
    state = _state(tmp_path)

    first = finalize_live_canonical(state)
    report = Path(first["canonical_live_report_path"])
    profile = (
        Path(state["output_dir"])
        / "_live_profiles"
        / "cluster_0"
        / "profile.json"
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (report, profile)
    }
    second = finalize_live_canonical(state)

    assert second["per_cluster_profiles"] == first["per_cluster_profiles"]
    assert second["canonical_live_report_path"] == first[
        "canonical_live_report_path"
    ]
    assert second["live_finalization_timings"][
        "profile_and_report_serialization_ms"
    ] >= 0
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (report, profile)
    } == before
    assert "face_embedding" not in profile.read_text(encoding="utf-8")
    assert not (tmp_path / "memory.db").exists()
