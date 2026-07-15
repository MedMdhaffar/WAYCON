from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading

import pytest

from forensics.person_creation import live_chunk_processing as processing
from forensics.person_creation.nodes.assign_bodies_to_clusters import (
    BodyAssignmentResult,
)
from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


def _crop(path: Path, kind: str, video: str = "camera-source") -> dict:
    return {
        "path": str(path),
        "frame_idx": 12,
        "video": video,
        "video_path": video,
        "bbox": [0, 0, 200, 300] if kind == "body" else [40, 20, 110, 100],
        "confidence": 0.95,
        "sharpness": 300.0,
    }


def _chunk(tmp_path: Path, *, video: str = "camera-source") -> LiveChunkResult:
    return LiveChunkResult(
        chunk_index=4,
        started_at=10.0,
        elapsed_seconds=5.0,
        stop_requested=False,
        frames_read=20,
        frames_processed=4,
        frames_skipped=16,
        frames_dropped=2,
        body_crops=[_crop(tmp_path / "body.jpg", "body", video)],
        face_crops=[_crop(tmp_path / "face.jpg", "face", video)],
        body_detection_count=1,
        face_detection_count=1,
        warnings=[],
    )


def _assignment(state: dict) -> BodyAssignmentResult:
    association = {
        "cluster_id": 0,
        "face_path": state["quality_face_crops"][0]["path"],
        "body_path": state["quality_body_crops"][0]["path"],
    }
    return BodyAssignmentResult(
        associations=[association],
        cluster_assignments={0: [association]},
        unattached_bodies=[],
        frame_groups=[],
        rejected_pairs=[],
        feedback_data={},
    )


def _install_success_stages(monkeypatch, calls: list[str]) -> None:
    def quality(state):
        calls.append("quality")
        return {
            "quality_body_crops": list(state["body_crops"]),
            "quality_face_crops": list(state["face_crops"]),
            "total_quality_body_crops": len(state["body_crops"]),
            "total_quality_face_crops": len(state["face_crops"]),
        }

    def embedding(state):
        calls.append("embedding")
        face = state["quality_face_crops"][0]
        return {
            "all_face_embeddings": [{
                "crop_path": face["path"],
                "embedding": [1.0, 0.0],
                "frame_idx": face["frame_idx"],
                "video": face["video"],
                "bbox": face["bbox"],
                "sharpness": face["sharpness"],
            }],
            "failed_face_embeddings": [],
        }

    def clustering(state):
        calls.append("clustering")
        return {
            "identity_clusters": [{
                "cluster_id": 0,
                "face_records": list(state["all_face_embeddings"]),
                "representative_embedding": [1.0, 0.0],
                "face_count": 1,
                "confidence": 1.0,
                "low_confidence": False,
            }],
            "unresolved_faces": [],
        }

    def association(state):
        calls.append("association")
        return _assignment(state)

    monkeypatch.setattr(processing, "filter_quality", quality)
    monkeypatch.setattr(processing, "embed_all_faces", embedding)
    monkeypatch.setattr(processing, "cluster_identities", clustering)
    monkeypatch.setattr(processing, "compute_body_cluster_assignments", association)


def test_stage_order_local_identity_and_metrics(monkeypatch, tmp_path):
    calls = []
    events = []
    _install_success_stages(monkeypatch, calls)

    result = processing.process_live_chunk_result(
        chunk=_chunk(tmp_path),
        base_state={"person_name": "Test", "identity_clustering_config": {}},
        notify=lambda status, update: events.append((status, update)),
    )

    assert calls == ["quality", "embedding", "clustering", "association"]
    assert result.identity_clusters[0]["cluster_id"] == 0
    assert result.identity_clusters[0]["local_identity_key"] == "chunk_0004:cluster_0"
    assert result.capture_summary["frames_processed"] == 4
    assert result.processing_elapsed_seconds >= 0
    assert [event[0] for event in events] == [
        "chunk_processing_started",
        "quality_filter_completed",
        "face_embedding_completed",
        "local_clustering_completed",
        "body_assignment_completed",
        "chunk_processing_completed",
    ]


def test_only_chunk_crops_are_used_and_inputs_are_not_mutated(monkeypatch, tmp_path):
    calls = []
    _install_success_stages(monkeypatch, calls)
    chunk = _chunk(tmp_path)
    base_state = {
        "person_name": "Test",
        "identity_clustering_config": {"eps": 0.3},
        "body_crops": [{"path": "old-body"}],
        "face_crops": [{"path": "old-face"}],
        "associations": [{"old": True}],
    }
    chunk_before = deepcopy(chunk)
    state_before = deepcopy(base_state)

    result = processing.process_live_chunk_result(chunk=chunk, base_state=base_state)
    result.quality_body_crops[0]["sharpness"] = -1

    assert chunk == chunk_before
    assert base_state == state_before
    assert result.quality_body_crops[0]["path"] == str(tmp_path / "body.jpg")


def test_runtime_state_is_not_copied_or_returned(monkeypatch, tmp_path):
    calls = []
    _install_success_stages(monkeypatch, calls)

    class CopyGuard:
        def __deepcopy__(self, _memo):
            raise AssertionError("complete base_state was deep-copied")

    base_state = {
        "person_name": "Test",
        "identity_clustering_config": {},
        "_stop_event": threading.Event(),
        "buffer": CopyGuard(),
        "lock": threading.Lock(),
        "model": CopyGuard(),
    }
    result = processing.process_live_chunk_result(
        chunk=_chunk(tmp_path), base_state=base_state
    )
    serialized = json.dumps(asdict(result))

    assert "CopyGuard" not in serialized
    assert "Event" not in serialized
    assert "lock" not in serialized


def test_empty_chunk_is_valid_without_constructing_embedder(monkeypatch, tmp_path):
    chunk = _chunk(tmp_path)
    chunk.body_crops = []
    chunk.face_crops = []
    chunk.body_detection_count = 0
    chunk.face_detection_count = 0
    monkeypatch.setattr(
        processing,
        "embed_all_faces",
        lambda _state: (_ for _ in ()).throw(AssertionError("embedder constructed")),
    )

    result = processing.process_live_chunk_result(chunk=chunk, base_state={})

    assert result.quality_body_crops == []
    assert result.face_embeddings == []
    assert result.identity_clusters == []
    assert result.associations == []
    assert "captured chunk contains no crops" in result.warnings


def test_all_quality_rejected_is_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        processing,
        "filter_quality",
        lambda _state: {"quality_body_crops": [], "quality_face_crops": []},
    )
    monkeypatch.setattr(
        processing,
        "embed_all_faces",
        lambda _state: (_ for _ in ()).throw(AssertionError("embedder constructed")),
    )

    result = processing.process_live_chunk_result(chunk=_chunk(tmp_path), base_state={})

    assert result.face_embeddings == []
    assert result.identity_clusters == []
    assert result.associations == []
    assert "all captured crops were rejected by quality filtering" in result.warnings


def test_embedding_failure_propagates_with_stage_context(monkeypatch, tmp_path):
    monkeypatch.setattr(
        processing,
        "filter_quality",
        lambda state: {
            "quality_body_crops": list(state["body_crops"]),
            "quality_face_crops": list(state["face_crops"]),
        },
    )
    monkeypatch.setattr(
        processing,
        "embed_all_faces",
        lambda _state: (_ for _ in ()).throw(OSError("face engine unavailable")),
    )

    with pytest.raises(
        processing.LiveChunkProcessingError,
        match="face embedding failed: face engine unavailable",
    ):
        processing.process_live_chunk_result(chunk=_chunk(tmp_path), base_state={})


def test_pure_association_is_used_and_no_report_is_written(monkeypatch, tmp_path):
    calls = []
    _install_success_stages(monkeypatch, calls)

    processing.process_live_chunk_result(chunk=_chunk(tmp_path), base_state={})

    assert calls[-1] == "association"
    assert not (tmp_path / "pairing_feedback.json").exists()


def test_crops_are_not_moved_deleted_or_renamed(monkeypatch, tmp_path):
    calls = []
    _install_success_stages(monkeypatch, calls)
    body = tmp_path / "body.jpg"
    face = tmp_path / "face.jpg"
    body.write_bytes(b"body-sentinel")
    face.write_bytes(b"face-sentinel")

    processing.process_live_chunk_result(chunk=_chunk(tmp_path), base_state={})

    assert body.read_bytes() == b"body-sentinel"
    assert face.read_bytes() == b"face-sentinel"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["body.jpg", "face.jpg"]


def test_returned_data_and_notifications_hide_camera_uri(monkeypatch, tmp_path):
    calls = []
    events = []
    _install_success_stages(monkeypatch, calls)
    uri = "rtsp://private-user:private-password@camera.local/live?token=secret"
    chunk = _chunk(tmp_path, video=uri)
    chunk.warnings.append(f"source warning: {uri}")

    result = processing.process_live_chunk_result(
        chunk=chunk,
        base_state={"person_name": "Test"},
        notify=lambda status, update: events.append((status, update)),
    )
    text = json.dumps({"result": asdict(result), "events": events})

    for forbidden in ("rtsp://", "private-user", "private-password", "camera.local", "token=secret"):
        assert forbidden not in text


def test_malformed_crop_metadata_raises_without_deleting_crop(tmp_path):
    chunk = _chunk(tmp_path)
    path = Path(chunk.face_crops[0]["path"])
    path.write_bytes(b"face-sentinel")
    del chunk.face_crops[0]["bbox"]

    with pytest.raises(ValueError, match="Malformed face crop.*missing bbox"):
        processing.process_live_chunk_result(chunk=chunk, base_state={})

    assert path.read_bytes() == b"face-sentinel"


def test_forbidden_stages_are_not_referenced_and_graph_is_unchanged():
    names = set(processing.process_live_chunk_result.__code__.co_names)
    assert names.isdisjoint({
        "promote_crops",
        "select_best",
        "compute_reid",
        "describe_clothing",
        "build_profile",
        "finalize",
        "GlobalMemory",
    })

    from forensics.person_creation.graph import build_graph

    nodes = set(build_graph().get_graph().nodes)
    assert "process_live_chunk_result" not in nodes
    assert {"process_video", "process_live_stream", "finalize"} <= nodes


def test_preprocessing_only_runs_quality_then_embedding(monkeypatch, tmp_path):
    calls = []
    _install_success_stages(monkeypatch, calls)

    result = processing.preprocess_live_chunk(
        chunk=_chunk(tmp_path),
        base_state={"person_name": "Test"},
    )

    assert calls == ["quality", "embedding"]
    assert result.chunk_index == 4
    assert len(result.quality_body_crops) == 1
    assert len(result.quality_face_crops) == 1
    assert len(result.face_embeddings) == 1


def test_preprocessing_only_has_no_permanent_or_identity_stages():
    names = set(processing.preprocess_live_chunk.__code__.co_names)
    assert names.isdisjoint({
        "cluster_identities",
        "compute_body_cluster_assignments",
        "promote_crops",
        "select_best",
        "compute_reid",
        "describe_clothing",
        "build_profile",
        "finalize",
        "GlobalMemory",
    })


def test_preview_outputs_match_existing_quality_and_embedding_nodes(
    monkeypatch, tmp_path
):
    import cv2
    import numpy as np

    face_path = tmp_path / "face.jpg"
    assert cv2.imwrite(str(face_path), np.full((120, 120, 3), 127, dtype=np.uint8))
    chunk = _chunk(tmp_path)
    chunk.face_crops[0]["path"] = str(face_path)

    class FakeFaceEngineClient:
        def embed(self, _image):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    import forensics.face_engine.client as face_client

    monkeypatch.setattr(face_client, "FaceEngineClient", FakeFaceEngineClient)
    canonical_state = {
        "body_crops": deepcopy(chunk.body_crops),
        "face_crops": deepcopy(chunk.face_crops),
    }
    quality_update = processing.filter_quality(canonical_state)
    canonical_state.update(quality_update)
    embedding_update = processing.embed_all_faces(canonical_state)

    preview = processing.preprocess_live_chunk(chunk=chunk, base_state={})

    assert preview.quality_body_crops == quality_update["quality_body_crops"]
    assert preview.quality_face_crops == quality_update["quality_face_crops"]
    assert preview.face_embeddings == embedding_update["all_face_embeddings"]
    assert preview.failed_face_embeddings == embedding_update[
        "failed_face_embeddings"
    ]


def test_quality_and_embedding_algorithm_source_hashes_are_unchanged():
    expected = {
        "filter_quality.py": (
            "A44E3A260E74FDC46A10CC9F6B7B041A20A9F29C347F900A466C922245FA02CE"
        ),
        "embed_all_faces.py": (
            "976F2DBFF099E7A7EB65FD8227D39BFD6A86AFE5ECA039E2E2C85A76412CE138"
        ),
    }
    nodes_dir = Path(processing.__file__).parent / "nodes"

    for filename, expected_hash in expected.items():
        actual = hashlib.sha256((nodes_dir / filename).read_bytes()).hexdigest()
        assert actual.upper() == expected_hash
