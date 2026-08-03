"""Safe, reusable processing for one already-captured live-camera chunk."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
import time
from typing import Any, Callable, Mapping

from forensics.person_creation.nodes.embed_all_faces import embed_all_faces
from forensics.person_creation.nodes.filter_quality import filter_quality
from forensics.person_creation.nodes.process_live_stream import LiveChunkResult


NotifyCallback = Callable[[str, dict | None], Any]
_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)
_REQUIRED_CROP_FIELDS = {"path", "frame_idx", "video", "bbox", "sharpness"}
cluster_identities: Callable[[dict], dict] | None = None
compute_body_cluster_assignments: Callable[[dict], Any] | None = None


class LiveChunkProcessingError(RuntimeError):
    """A chunk-local processing stage failed."""


@dataclass(frozen=True)
class ProcessedLiveChunk:
    chunk_index: int
    capture_summary: dict
    quality_body_crops: list[dict]
    quality_face_crops: list[dict]
    face_embeddings: list[dict]
    failed_face_embeddings: list[dict]
    identity_clusters: list[dict]
    unresolved_faces: list[dict]
    associations: list[dict]
    cluster_assignments: dict[int, list[dict]]
    unattached_bodies: list[dict]
    frame_groups: list[dict]
    rejected_pairs: list[dict]
    warnings: list[str]
    processing_elapsed_seconds: float


@dataclass(frozen=True)
class PreprocessedLiveChunk:
    chunk_index: int
    capture_summary: dict
    quality_body_crops: list[dict]
    quality_face_crops: list[dict]
    face_embeddings: list[dict]
    failed_face_embeddings: list[dict]
    warnings: list[str]
    processing_elapsed_seconds: float
    face_rejection_counts: dict[str, int] = field(default_factory=dict)


def _camera_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _camera_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_camera_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_camera_safe(item) for item in value]
    if isinstance(value, str):
        return _CAMERA_URI_RE.sub("<camera-source>", value)
    return value


def _copy_crop_records(records: list[dict], kind: str) -> list[dict]:
    copied: list[dict] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Malformed {kind} crop at index {index}: expected a mapping")
        missing = sorted(_REQUIRED_CROP_FIELDS - record.keys())
        if missing:
            raise ValueError(
                f"Malformed {kind} crop at index {index}: missing {', '.join(missing)}"
            )
        bbox = record["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"Malformed {kind} crop at index {index}: bbox must contain four values"
            )
        try:
            [float(value) for value in bbox]
            float(record["sharpness"])
            int(record["frame_idx"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed {kind} crop at index {index}: invalid numeric metadata"
            ) from exc
        copied.append(_camera_safe(deepcopy(dict(record))))
    return copied


def _notify(callback: NotifyCallback | None, status: str, update: dict) -> None:
    if callback is not None:
        callback(status, _camera_safe(update))


def _run_stage(name: str, stage: Callable[[dict], dict], state: dict) -> dict:
    try:
        update = stage(state)
    except Exception as exc:
        raise LiveChunkProcessingError(f"{name} failed: {exc}") from exc
    if not isinstance(update, dict):
        raise LiveChunkProcessingError(f"{name} failed: expected a dictionary update")
    state.update(update)
    return update


def _local_clustering_stage() -> Callable[[dict], dict]:
    if cluster_identities is not None:
        return cluster_identities
    from forensics.person_creation.nodes.cluster_identities import (
        cluster_identities as stage,
    )

    return stage


def _body_assignment_stage() -> Callable[[dict], Any]:
    if compute_body_cluster_assignments is not None:
        return compute_body_cluster_assignments
    from forensics.person_creation.nodes.assign_bodies_to_clusters import (
        compute_body_cluster_assignments as stage,
    )

    return stage


def preprocess_live_chunk(
    *,
    chunk: LiveChunkResult,
    base_state: Mapping[str, Any],
    notify: NotifyCallback | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> PreprocessedLiveChunk:
    """Run preview-only quality filtering and face embedding for one chunk."""
    started = monotonic()
    warnings = [_camera_safe(str(item)) for item in chunk.warnings]
    body_crops = _copy_crop_records(chunk.body_crops, "body")
    face_crops = _copy_crop_records(chunk.face_crops, "face")
    video_paths = list(dict.fromkeys(
        str(crop["video"]) for crop in [*body_crops, *face_crops]
    ))
    local_state = {
        "person_name": str(base_state.get("person_name") or ""),
        "video_paths": video_paths,
        "body_crops": body_crops,
        "face_crops": face_crops,
    }

    _notify(notify, "chunk_preprocessing_started", {
        "chunk_index": chunk.chunk_index,
    })
    if not body_crops and not face_crops:
        warnings.append("captured chunk contains no crops")

    quality = _run_stage("quality filtering", filter_quality, local_state)
    quality_completed = monotonic()
    quality_body = list(quality.get("quality_body_crops") or [])
    quality_face = list(quality.get("quality_face_crops") or [])
    timing_by_path = {}
    for crop in quality_face:
        timing = dict(crop.get("_latency_timing") or {})
        if timing:
            timing["quality_accepted_monotonic"] = float(quality_completed)
            timing_by_path[str(crop.get("path"))] = timing
        crop.pop("_latency_timing", None)
    if (body_crops or face_crops) and not quality_body and not quality_face:
        warnings.append("all captured crops were rejected by quality filtering")
    _notify(notify, "chunk_quality_filter_completed", {
        "chunk_index": chunk.chunk_index,
        "quality_body_count": len(quality_body),
        "quality_face_count": len(quality_face),
    })

    if quality_face:
        embedded = _run_stage("face embedding", embed_all_faces, local_state)
    else:
        embedded = {"all_face_embeddings": [], "failed_face_embeddings": []}
    face_embeddings = list(embedded.get("all_face_embeddings") or [])
    failed_embeddings = list(embedded.get("failed_face_embeddings") or [])
    embedding_completed = monotonic()
    for record in face_embeddings:
        timing = dict(timing_by_path.get(str(record.get("crop_path"))) or {})
        if not timing:
            continue
        timing["embedding_completed_monotonic"] = float(embedding_completed)
        record["_latency_timing"] = timing
        if timing.get("source_frame_timestamp"):
            record["source_frame_timestamp"] = timing["source_frame_timestamp"]
    if quality_face and not face_embeddings:
        warnings.append("no valid face embeddings were produced")
    if failed_embeddings:
        warnings.append(f"face embedding failed for {len(failed_embeddings)} crop(s)")

    elapsed = max(0.0, monotonic() - started)
    result = PreprocessedLiveChunk(
        chunk_index=int(chunk.chunk_index),
        capture_summary=_camera_safe(chunk.report_metrics()),
        quality_body_crops=deepcopy(quality_body),
        quality_face_crops=deepcopy(quality_face),
        face_embeddings=deepcopy(face_embeddings),
        failed_face_embeddings=deepcopy(failed_embeddings),
        warnings=warnings,
        processing_elapsed_seconds=elapsed,
        face_rejection_counts=deepcopy(
            quality.get("face_rejection_counts") or {}
        ),
    )
    _notify(notify, "chunk_preprocessing_completed", {
        "chunk_index": chunk.chunk_index,
        "processing_elapsed_seconds": round(elapsed, 3),
        "quality_body_count": len(quality_body),
        "quality_face_count": len(quality_face),
        "embedding_count": len(face_embeddings),
        "failed_embedding_count": len(failed_embeddings),
    })
    return result


def process_live_chunk_result(
    *,
    chunk: LiveChunkResult,
    base_state: Mapping[str, Any],
    notify: NotifyCallback | None = None,
) -> ProcessedLiveChunk:
    """Process one chunk into chunk-local identities and body associations.

    Only ``person_name`` and ``identity_clustering_config`` are selected from
    ``base_state``. Crop lists are copied from ``chunk``; runtime objects and
    accumulated graph/session fields are deliberately excluded.
    """
    started = time.perf_counter()
    warnings = [_camera_safe(str(item)) for item in chunk.warnings]
    body_crops = _copy_crop_records(chunk.body_crops, "body")
    face_crops = _copy_crop_records(chunk.face_crops, "face")
    video_paths = list(dict.fromkeys(
        str(crop["video"]) for crop in [*body_crops, *face_crops]
    ))
    local_state = {
        "person_name": str(base_state.get("person_name") or ""),
        "identity_clustering_config": dict(
            base_state.get("identity_clustering_config") or {}
        ),
        "video_paths": video_paths,
        "body_crops": body_crops,
        "face_crops": face_crops,
    }

    _notify(notify, "chunk_processing_started", {"chunk_index": chunk.chunk_index})
    if not body_crops and not face_crops:
        warnings.append("captured chunk contains no crops")

    quality = _run_stage("quality filtering", filter_quality, local_state)
    quality_body = list(quality.get("quality_body_crops") or [])
    quality_face = list(quality.get("quality_face_crops") or [])
    if (body_crops or face_crops) and not quality_body and not quality_face:
        warnings.append("all captured crops were rejected by quality filtering")
    _notify(notify, "quality_filter_completed", {
        "chunk_index": chunk.chunk_index,
        "quality_body_count": len(quality_body),
        "quality_face_count": len(quality_face),
    })

    if quality_face:
        embedded = _run_stage("face embedding", embed_all_faces, local_state)
    else:
        embedded = {"all_face_embeddings": [], "failed_face_embeddings": []}
        local_state.update(embedded)
    face_embeddings = list(embedded.get("all_face_embeddings") or [])
    failed_embeddings = list(embedded.get("failed_face_embeddings") or [])
    if quality_face and not face_embeddings:
        warnings.append("no valid face embeddings were produced")
    if failed_embeddings:
        warnings.append(f"face embedding failed for {len(failed_embeddings)} crop(s)")
    _notify(notify, "face_embedding_completed", {
        "chunk_index": chunk.chunk_index,
        "embedding_count": len(face_embeddings),
        "failed_embedding_count": len(failed_embeddings),
    })

    clustered = _run_stage(
        "local identity clustering",
        _local_clustering_stage(),
        local_state,
    )
    identity_clusters = []
    for cluster in clustered.get("identity_clusters") or []:
        local_cluster = deepcopy(cluster)
        local_cluster["local_identity_key"] = (
            f"chunk_{int(chunk.chunk_index):04d}:cluster_{cluster['cluster_id']}"
        )
        identity_clusters.append(local_cluster)
    local_state["identity_clusters"] = identity_clusters
    unresolved_faces = list(clustered.get("unresolved_faces") or [])
    if face_embeddings and not identity_clusters:
        warnings.append("no local identity clusters were formed")
    _notify(notify, "local_clustering_completed", {
        "chunk_index": chunk.chunk_index,
        "local_identity_count": len(identity_clusters),
        "unresolved_face_count": len(unresolved_faces),
    })

    try:
        assignment = _body_assignment_stage()(local_state)
    except Exception as exc:
        raise LiveChunkProcessingError(f"body association failed: {exc}") from exc
    if identity_clusters and quality_body and not assignment.associations:
        warnings.append("no valid body-to-identity associations were found")
    _notify(notify, "body_assignment_completed", {
        "chunk_index": chunk.chunk_index,
        "association_count": len(assignment.associations),
        "rejected_pair_count": len(assignment.rejected_pairs),
    })

    elapsed = max(0.0, time.perf_counter() - started)
    capture_summary = _camera_safe(chunk.report_metrics())
    result = ProcessedLiveChunk(
        chunk_index=int(chunk.chunk_index),
        capture_summary=capture_summary,
        quality_body_crops=quality_body,
        quality_face_crops=quality_face,
        face_embeddings=face_embeddings,
        failed_face_embeddings=failed_embeddings,
        identity_clusters=identity_clusters,
        unresolved_faces=unresolved_faces,
        associations=assignment.associations,
        cluster_assignments=assignment.cluster_assignments,
        unattached_bodies=assignment.unattached_bodies,
        frame_groups=assignment.frame_groups,
        rejected_pairs=assignment.rejected_pairs,
        warnings=warnings,
        processing_elapsed_seconds=elapsed,
    )
    _notify(notify, "chunk_processing_completed", {
        "chunk_index": chunk.chunk_index,
        "processing_elapsed_seconds": round(elapsed, 3),
        "local_identity_count": len(identity_clusters),
        "association_count": len(assignment.associations),
    })
    return result
