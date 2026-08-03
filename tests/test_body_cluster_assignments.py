from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from forensics.person_creation.nodes import assign_bodies_to_clusters as assignment_node


def _state(output_dir: Path, *, video: str = "video.mp4") -> dict:
    face_path = str(output_dir / "face.jpg")
    body_path = str(output_dir / "body.jpg")
    return {
        "person_name": "Test Person",
        "video_paths": [video],
        "output_dir": str(output_dir),
        "identity_clusters": [{
            "cluster_id": 4,
            "face_records": [{"crop_path": face_path}],
        }],
        "quality_face_crops": [{
            "path": face_path,
            "frame_idx": 12,
            "video": video,
            "bbox": [40, 20, 60, 44],
            "score": 0.95,
            "sharpness": 300.0,
        }],
        "quality_body_crops": [{
            "path": body_path,
            "frame_idx": 12,
            "video": video,
            "bbox": [0, 0, 100, 200],
            "score": 0.90,
            "sharpness": 600.0,
        }],
    }


def test_pure_computation_has_no_filesystem_side_effects(tmp_path):
    output_dir = tmp_path / "does-not-exist"
    state = _state(output_dir)

    result = assignment_node.compute_body_cluster_assignments(state)

    assert len(result.associations) == 1
    assert not output_dir.exists()


def test_pure_computation_does_not_mutate_or_alias_input(tmp_path):
    state = _state(tmp_path / "output")
    before = deepcopy(state)

    result = assignment_node.compute_body_cluster_assignments(state)
    result.frame_groups[0]["bodies"][0]["sharpness"] = -1
    result.unattached_bodies.append({"path": "new"})

    assert state == before


def test_simple_valid_assignment_preserves_public_record_fields(tmp_path):
    state = _state(tmp_path / "output")

    result = assignment_node.compute_body_cluster_assignments(state)

    assert len(result.associations) == 1
    association = result.associations[0]
    assert association["cluster_id"] == 4
    assert association["frame_idx"] == 12
    assert association["face_path"] == state["quality_face_crops"][0]["path"]
    assert association["body_path"] == state["quality_body_crops"][0]["path"]
    assert association["face_crop_path"] == association["face_path"]
    assert association["body_crop_path"] == association["body_path"]
    assert association["assignment_score"] == association["auto_score"]
    assert result.cluster_assignments[4][0]["assignment_score"] == association["assignment_score"]
    assert result.unattached_bodies == []


def test_empty_input_preserves_valid_empty_result(tmp_path):
    state = {
        "person_name": "Nobody",
        "video_paths": [],
        "output_dir": str(tmp_path / "output"),
        "identity_clusters": [],
        "quality_face_crops": [],
        "quality_body_crops": [],
    }

    result = assignment_node.compute_body_cluster_assignments(state)

    assert result.associations == []
    assert result.cluster_assignments == {}
    assert result.unattached_bodies == []
    assert result.frame_groups == []
    assert result.feedback_data["identity_clusters_found"] == 0
    assert result.feedback_data["candidate_pairs_count"] == 0


def test_feedback_writer_preserves_schema_without_recomputation(
    tmp_path, monkeypatch
):
    result = assignment_node.compute_body_cluster_assignments(_state(tmp_path / "source"))
    feedback_before = deepcopy(result.feedback_data)
    monkeypatch.setattr(
        assignment_node,
        "compute_body_cluster_assignments",
        lambda _state: (_ for _ in ()).throw(AssertionError("unexpected recomputation")),
    )
    output_path = tmp_path / "reports" / "pairing_feedback.json"

    written = assignment_node.write_pairing_feedback(result, output_path)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert set(payload) == {
        "timestamp",
        "pairing_mode",
        "person_name",
        "video_sources",
        "identity_clusters_found",
        "total_frame_groups_shown",
        "candidate_pairs_count",
        "confirmed_pairs_count",
        "rejected_pairs_count",
        "unattached_bodies_count",
        "confirmed_pairs",
        "rejected_pairs",
    }
    assert payload["pairing_mode"] == "face_cluster_assignment_v1"
    assert payload["confirmed_pairs"] == result.feedback_data["confirmed_pairs"]
    assert result.feedback_data == feedback_before


def test_langgraph_wrapper_writes_feedback_and_preserves_updates(tmp_path):
    state = _state(tmp_path / "output")
    expected = assignment_node.compute_body_cluster_assignments(state)

    update = assignment_node.assign_bodies_to_clusters(state)
    feedback_path = Path(update["human_feedback_path"])

    assert feedback_path == (tmp_path / "output" / "pairing_feedback.json").resolve()
    assert feedback_path.is_file()
    assert update["associations"] == expected.associations
    assert update["cluster_assignments"] == expected.cluster_assignments
    assert update["unattached_bodies"] == expected.unattached_bodies
    assert update["frame_groups"] == expected.frame_groups


def test_feedback_is_serializable_and_credential_safe(tmp_path):
    raw_uri = "rtsp://private-user:private-password@camera.local/live?token=secret"
    state = _state(tmp_path / "source", video=raw_uri)
    state["_stop_event"] = threading.Event()
    state["_status_callback"] = lambda *_args: None
    result = assignment_node.compute_body_cluster_assignments(state)

    output_path = tmp_path / "pairing_feedback.json"
    assignment_node.write_pairing_feedback(result, output_path)
    text = output_path.read_text(encoding="utf-8")
    json.loads(text)

    assert "private-user" not in text
    assert "private-password" not in text
    assert "token=secret" not in text
    assert "rtsp://" not in text
    assert "camera.local" not in text
    assert "Event" not in text


def test_association_functions_never_move_delete_or_rename_crops(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    state = _state(output_dir)
    face = Path(state["quality_face_crops"][0]["path"])
    body = Path(state["quality_body_crops"][0]["path"])
    face.write_bytes(b"face-sentinel")
    body.write_bytes(b"body-sentinel")
    before = {path.name: path.read_bytes() for path in (face, body)}

    result = assignment_node.compute_body_cluster_assignments(state)
    assignment_node.write_pairing_feedback(
        result,
        output_dir / "pairing_feedback.json",
    )

    assert {path.name: path.read_bytes() for path in (face, body)} == before
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "body.jpg",
        "face.jpg",
        "pairing_feedback.json",
    ]


def test_wrapper_remains_registered_in_current_graph():
    from forensics.person_creation.graph import build_graph

    graph = build_graph().get_graph()

    assert "assign_bodies_to_clusters" in graph.nodes
