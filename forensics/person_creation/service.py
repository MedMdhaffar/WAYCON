import json
import re
import sqlite3
import threading
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2 as _cv2

from forensics.media_paths import (
    IMAGE_EXTENSIONS,
    MediaPathError,
    UnsupportedMediaTypeError,
    get_media_root,
    normalize_media_path,
    resolve_media_path,
)
from forensics.person_identifier.config import Config as _PIConfig
from forensics.person_creation.path_utils import to_wsl_path as _to_wsl_path
from forensics.person_creation.live_stream import mask_camera_uri
from forensics.person_creation.profile_management import bp as profile_management_bp
from forensics.person_creation.tools.cleanup_orphan_crops import (
    cleanup as _cleanup_orphan_crops,
    CleanupError as _CleanupError,
    _referenced_basenames as _ref_basenames,
    _list_jpgs as _list_jpgs,
)
from forensics.person_creation.tools.add_face_photos import (
    add_face_photos as _add_face_photos,
    AddFacePhotosError as _AddFacePhotosError,
)


def _validate_video_path(p: str) -> str | None:
    """Return None if cv2 can open and read at least one frame; else a reason string."""
    cap = _cv2.VideoCapture(p)
    try:
        if not cap.isOpened():
            return "cv2 cannot open file (does it exist? right codec?)"
        ret, _frame = cap.read()
        if not ret:
            return "cv2 opened but decoded 0 frames (codec / file may be corrupt)"
        return None
    finally:
        cap.release()


class StartRequestError(ValueError):
    def __init__(self, message: str, details: list[dict] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


def _as_int(value, field_name: str) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise StartRequestError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StartRequestError(f"{field_name} must be an integer") from exc
    if str(value).strip() != str(parsed):
        raise StartRequestError(f"{field_name} must be an integer")
    return parsed


def build_initial_state(body: dict, *, validate_video_paths: bool = True) -> dict:
    """Validate a start payload and build graph state for video or live input."""
    if not isinstance(body, dict):
        raise StartRequestError("JSON object required")
    name = str(body.get("name", "")).strip()
    if not name:
        raise StartRequestError("name required")

    input_type = str(body.get("input_type") or "video_file").strip().lower()
    if input_type not in {"video", "video_file", "camera_uri"}:
        raise StartRequestError("input_type must be 'video_file' or 'camera_uri'")
    if input_type == "video":
        input_type = "video_file"

    output_dir = str(body.get("output_dir") or f"forensics/person_db/{name.lower()}")
    try:
        output_relative = normalize_media_path(
            output_dir,
            allow_legacy_absolute=False,
            require_exists=False,
        )
        output_dir = str(get_media_root() / output_relative)
    except MediaPathError as exc:
        raise StartRequestError("output_dir must be inside the configured media root") from exc
    if input_type == "camera_uri":
        from forensics.person_creation.nodes.process_live_stream import (
            live_process_every_n_frames,
        )

        try:
            every_n = live_process_every_n_frames(body.get("every_n"))
        except ValueError as exc:
            raise StartRequestError(str(exc)) from exc
    else:
        every_n = max(1, _as_int(body.get("every_n", 15), "every_n"))
    initial_state = {
        "person_name": name,
        "input_type": input_type,
        "source_type": "live_camera" if input_type == "camera_uri" else "video_file",
        "video_paths": [],
        "output_dir": str(Path(output_dir)),
        "process_every_n": every_n,
        "identity_clustering_config": body.get("identity_clustering_config", {}),
        "reid_config": body.get("reid", body.get("reid_config", {})),
        "body_crops": [],
        "face_crops": [],
    }

    if input_type == "camera_uri":
        camera_uri = str(body.get("camera_uri") or "").strip()
        if not camera_uri:
            raise StartRequestError("camera_uri required when input_type is 'camera_uri'")
        try:
            parsed = urlsplit(camera_uri)
        except ValueError as exc:
            raise StartRequestError("camera_uri is invalid") from exc
        if parsed.scheme.lower() not in {"rtsp", "http", "https", "file"}:
            raise StartRequestError("camera_uri scheme must be rtsp://, http://, https://, or file://")
        if parsed.scheme.lower() == "file":
            if not parsed.path:
                raise StartRequestError("file:// camera_uri must include a path")
        elif not parsed.netloc:
            raise StartRequestError("camera_uri must include a host")

        duration = _as_int(body.get("duration_seconds", 30), "duration_seconds")
        duration = max(5, min(duration, 300))
        camera_id = str(body.get("camera_id") or "").strip()[:128] or None
        live_config = body.get("live_stream_config") or {}
        if not isinstance(live_config, dict):
            raise StartRequestError("live_stream_config must be an object")
        initial_state.update({
            "camera_uri": camera_uri,
            "source_uri_masked": mask_camera_uri(camera_uri),
            "camera_id": camera_id,
            "duration_seconds": duration,
            "live_stream_config": live_config,
        })
        return initial_state

    video_paths = body.get("video_paths", [])
    if not isinstance(video_paths, list) or not video_paths:
        raise StartRequestError("name and video_paths required")
    normalized: list[str] = []
    details: list[dict] = []
    for raw_value in video_paths:
        raw = str(raw_value)
        norm = _to_wsl_path(raw)
        reason = _validate_video_path(norm) if validate_video_paths else None
        if reason is not None:
            details.append({"input": raw, "normalized": norm, "reason": reason})
        else:
            normalized.append(norm)
    if details:
        raise StartRequestError("video_paths failed validation", details)
    initial_state["video_paths"] = normalized
    return initial_state


_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ALLOWED_PROFILE_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
CORS(app)

app.register_blueprint(profile_management_bp)


def initialize_global_memory() -> None:
    """Run schema creation/migrations through one writable startup handle."""
    from forensics.global_memory import GlobalMemory

    memory = GlobalMemory()
    memory.close()


def _is_sqlite_busy_error(error: BaseException) -> bool:
    if not isinstance(error, sqlite3.OperationalError):
        return False
    error_code = getattr(error, "sqlite_errorcode", None)
    if isinstance(error_code, int) and (error_code & 0xFF) in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }:
        return True
    message = str(error).lower()
    return "database is locked" in message or "database table is locked" in message


@app.errorhandler(sqlite3.OperationalError)
def _sqlite_operational_error_response(error: sqlite3.OperationalError):
    if _is_sqlite_busy_error(error):
        return jsonify({"error": "global memory is temporarily busy"}), 503
    return jsonify({"error": "database operation failed"}), 500


def _allowed_profile_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in _ALLOWED_PROFILE_IMAGE_EXTENSIONS


_MEDIA_VALUE_KEYS = {
    "path",
    "crop_path",
    "face_path",
    "body_path",
    "face_crop_path",
    "body_crop_path",
    "representative_face_path",
    "profile_image",
    "best_face_crop",
    "selected_body_crop",
    "best_face_path",
    "best_body_path",
}
_MEDIA_LIST_KEYS = {"face_crops", "body_crops", "best_body_crops"}
_PUBLIC_VLM_ERRORS = {
    "image_decode_failed",
    "inference_error",
    "invalid_output",
    "empty_output",
    "no_valid_body_crop",
    "persistence_error",
    "queue_capacity",
    "timeout",
}


def _public_media_reference(value: Any) -> str | None:
    if not value:
        return None
    try:
        return normalize_media_path(
            str(value),
            allow_legacy_absolute=True,
            require_exists=True,
        )
    except (MediaPathError, OSError):
        return None


def _sanitize_media_references(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in _MEDIA_VALUE_KEYS:
                sanitized[key] = _public_media_reference(item)
            elif key == "vlm_error":
                sanitized[key] = (
                    item if item in _PUBLIC_VLM_ERRORS else "inference_error"
                ) if item else None
            elif key == "failure_reason" and parent_key == "clothing_diagnostics":
                sanitized[key] = (
                    item if item in _PUBLIC_VLM_ERRORS else "inference_error"
                ) if item else None
            elif key in _MEDIA_LIST_KEYS and isinstance(item, list):
                sanitized[key] = [
                    reference
                    for raw in item
                    if (reference := _public_media_reference(raw)) is not None
                ]
            else:
                sanitized[key] = _sanitize_media_references(item, key)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_media_references(item, parent_key) for item in value]
    return value


# Representative-strategy comparison is an offline concern
# (forensics/person_creation/tools/compare_cluster_representatives.py). It
# carries the enrolled-person roster and one similarity row per enrolled
# identity, so the public payload drops it even if a producer reattaches it.
_OFFLINE_DIAGNOSTIC_KEYS = {
    "representative_similarity_diagnostic",
    "medoid_similarity",
    "normalized_mean_similarity",
    "medoid_best_person_id",
    "medoid_best_similarity",
    "normalized_mean_best_person_id",
    "normalized_mean_best_similarity",
    "similarity_drift",
    "maximum_absolute_similarity_drift",
}


def _embedding_free_public_projection(value: Any) -> Any:
    """Recursively remove raw vectors and offline diagnostics from status."""
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                normalized in {"embedding", "embeddings"}
                or normalized.endswith("_embedding")
                or (
                    normalized.endswith("_embeddings")
                    and isinstance(item, (dict, list, tuple))
                )
                or normalized in _OFFLINE_DIAGNOSTIC_KEYS
            ):
                continue
            projected[key] = _embedding_free_public_projection(item)
        return projected
    if isinstance(value, list):
        return [_embedding_free_public_projection(item) for item in value]
    if isinstance(value, tuple):
        return [_embedding_free_public_projection(item) for item in value]
    return value


def _live_vlm_status(rolling: Any) -> Any:
    """Reject non-canonical media references in versioned live identities."""
    if not isinstance(rolling, dict):
        return rolling
    result = deepcopy(rolling)
    identities = result.get("live_identities")
    if not isinstance(identities, list):
        return result
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        person_id = str(identity.get("canonical_person_id") or "")
        for key, crop_type in (
            ("representative_face_path", "face"),
            ("best_face_path", "face"),
            ("best_body_path", "body"),
            ("selected_body_crop", "body"),
        ):
            if key not in identity:
                continue
            reference = _public_media_reference(identity.get(key))
            parts = reference.split("/") if reference else []
            canonical = (
                len(parts) == 3
                and re.fullmatch(r"person_[0-9]+", parts[0])
                and parts[0] == person_id
                and parts[1] == f"{crop_type}_crops"
            )
            provisional_staging = (
                not person_id
                and len(parts) >= 4
                and parts[-3] == "_staging"
                and parts[-2] == f"{crop_type}_crops"
            )
            identity[key] = reference if (
                canonical or provisional_staging
            ) else None
    return result


@dataclass
class JobState:
    job_id: str
    input_type: str = "video_file"
    status: str = "idle"
    # idle | loading_models | processing_video | filtering | embedding | clustering
    # | auto_pairing | selecting | computing_reid | describing | building_profile
    # | finalizing | stop_requested | stopping | done | error
    node: str = ""
    error: str | None = None
    snapshot: dict = field(default_factory=dict)
    output_dir: str = ""


@dataclass
class JobRuntime:
    stop_event: threading.Event = field(default_factory=threading.Event)
    stop_requested_monotonic: float | None = None


_jobs: dict[str, JobState] = {}
_job_runtimes: dict[str, JobRuntime] = {}
_jobs_lock = threading.RLock()
_graph = None
_graph_lock = threading.Lock()


def _rolling_counter(value: Any) -> int:
    try:
        return int(value) if value is not None else -1
    except (TypeError, ValueError):
        return -1


_LIVE_CANONICAL_SNAPSHOT_KEYS = {
    "identity_clusters",
    "unresolved_faces",
    "associations",
    "cluster_assignments",
    "unattached_bodies",
    "frame_groups",
    "rejected_pairs",
    "per_cluster_best_body_crops",
    "best_body_crops",
    "reid_embeddings",
    "reid_crop_counts",
    "reid_reasons",
    "per_cluster_profiles",
    "per_cluster_clothing",
    "profile",
}


def _merge_job_snapshot(job: JobState, snapshot_update: dict) -> None:
    """Atomically project capture or canonical live state into the job registry."""
    from forensics.person_creation.media_lifecycle import (
        rewrite_media_references,
        scrub_obsolete_session_media,
    )

    update = dict(snapshot_update)
    remap = update.pop("_media_path_remap", None)
    finalized_root = update.pop("_media_finalized_root", None)
    update.pop("_media_cleanup_pairs", None)
    if isinstance(remap, dict) and remap:
        job.snapshot = rewrite_media_references(job.snapshot, remap)
    if finalized_root:
        job.snapshot = scrub_obsolete_session_media(
            job.snapshot,
            output_dir=finalized_root,
        )
    incoming = update.pop("rolling_analysis", None)
    if not isinstance(incoming, dict):
        # Capture-only publications can advance counters and chunk metadata, but
        # cannot erase identity state owned by rolling analysis.
        if (
            job.input_type == "camera_uri"
            and isinstance(job.snapshot.get("rolling_analysis"), dict)
        ):
            for key in _LIVE_CANONICAL_SNAPSHOT_KEYS:
                update.pop(key, None)
        job.snapshot.update(update)
        return
    current = job.snapshot.get("rolling_analysis")
    if isinstance(current, dict):
        monotonic_fields = (
            "publication_sequence",
            "evidence_version",
            "analysis_version",
            "requested_version",
        )
        if any(
            _rolling_counter(incoming.get(field))
            < _rolling_counter(current.get(field))
            for field in monotonic_fields
        ):
            return

    canonical_update = {
        key: update.pop(key)
        for key in tuple(update)
        if key in _LIVE_CANONICAL_SNAPSHOT_KEYS
    }
    job.snapshot.update(update)
    for key, value in canonical_update.items():
        if key == "per_cluster_clothing":
            existing = job.snapshot.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged = deepcopy(existing)
                merged.update(deepcopy(value))
                job.snapshot[key] = merged
                continue
        if key == "profile" and not value and job.snapshot.get(key):
            continue
        job.snapshot[key] = deepcopy(value)
    job.snapshot["rolling_analysis"] = deepcopy(incoming)


def _get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            from forensics.person_creation.graph import build_graph
            _graph = build_graph()
    return _graph


_NODE_TO_STATUS = {
    "load_models":       "loading_models",
    "process_video":     "processing_video",
    "process_live_stream": "processing_live_frames",
    "filter_quality":    "filtering",
    "embed_all_faces":   "embedding",
    "cluster_identities": "clustering",
    "assign_bodies_to_clusters": "auto_pairing",
    "promote_crops":     "promoting_crops",
    "cleanup_promoted_crops": "promoting_crops",
    "select_best":       "selecting",
    "compute_reid":      "computing_reid",
    "describe_clothing": "describing",
    "build_profile":     "building_profile",
    "finalize":          "finalizing",
    "cleanup_finalized_media": "finalizing",
}


def _run_pipeline(job_id: str, initial_state: dict) -> None:
    """Run the graph start to end with no human interrupts."""
    with _jobs_lock:
        job = _jobs[job_id]
        runtime = _job_runtimes.get(job_id)

    def update_live_status(status: str, snapshot_update: dict | None = None) -> None:
        with _jobs_lock:
            job.node = "process_live_stream"
            if runtime is None or not runtime.stop_event.is_set() or status == "stopping":
                job.status = status
            if snapshot_update:
                _merge_job_snapshot(job, snapshot_update)

    if initial_state.get("input_type") == "camera_uri":
        initial_state["_status_callback"] = update_live_status
        initial_state["_job_id"] = job_id
        if runtime is not None:
            initial_state["_stop_event"] = runtime.stop_event

    try:
        graph = _get_graph()
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in event.items():
                with _jobs_lock:
                    job.node = node_name
                    if runtime is None or not runtime.stop_event.is_set():
                        job.status = _NODE_TO_STATUS.get(node_name, node_name)
                    if isinstance(update, dict):
                        _merge_job_snapshot(job, update)

        with _jobs_lock:
            terminal_started = time.monotonic()
            if runtime is not None and runtime.stop_requested_monotonic is not None:
                stream_stats = deepcopy(job.snapshot.get("stream_stats") or {})
                stop_timings = deepcopy(stream_stats.get("stop_timings") or {})
                stop_timings["total_stop_ms"] = round(
                    (
                        time.monotonic()
                        - runtime.stop_requested_monotonic
                    ) * 1000.0,
                    3,
                )
                stop_timings["terminal_state_publication_ms"] = round(
                    (time.monotonic() - terminal_started) * 1000.0,
                    3,
                )
                stream_stats["stop_timings"] = stop_timings
                job.snapshot["stream_stats"] = stream_stats
            job.status = "done"

    except Exception:
        error = traceback.format_exc()
        raw_uri = initial_state.get("camera_uri")
        if raw_uri:
            error = error.replace(raw_uri, mask_camera_uri(raw_uri))
        with _jobs_lock:
            job.status = "error"
            job.error = error
    finally:
        with _jobs_lock:
            _job_runtimes.pop(job_id, None)


def _start_pipeline_thread(job_id: str, initial_state: dict) -> None:
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, initial_state),
        daemon=True,
    )
    thread.start()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    from forensics.person_creation.models.device import get_device_status

    return jsonify({
        "status": "ok",
        "service": "waycon-person-creation",
        **get_device_status(),
    })


@app.post("/api/person/start")
def start():
    body = request.get_json(force=True)
    try:
        initial_state = build_initial_state(body)
    except StartRequestError as exc:
        response = {"error": str(exc)}
        if exc.details:
            response["details"] = exc.details
        return jsonify(response), 400

    job_id = str(uuid.uuid4())
    safe_initial_snapshot = {
        "source_type": initial_state.get("source_type", "video_file"),
        "camera_id": initial_state.get("camera_id"),
        "duration_seconds": initial_state.get("duration_seconds"),
        "sampling_interval_frames": initial_state.get("process_every_n"),
        "source_uri_masked": initial_state.get("source_uri_masked", ""),
    }
    job = JobState(
        job_id=job_id,
        input_type=initial_state.get("input_type", "video_file"),
        snapshot=safe_initial_snapshot,
        output_dir=initial_state["output_dir"],
    )
    with _jobs_lock:
        _jobs[job_id] = job
        if job.input_type == "camera_uri":
            _job_runtimes[job_id] = JobRuntime()

    _start_pipeline_thread(job_id, initial_state)
    return jsonify({"job_id": job_id})


@app.post("/api/person/stop/<job_id>")
def stop(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        if job.status in {"done", "error"}:
            return jsonify({"job_id": job_id, "status": "already_finished"})
        if job.input_type != "camera_uri":
            return jsonify({"job_id": job_id, "status": "not_live_camera"}), 409

        runtime = _job_runtimes.get(job_id)
        if runtime is None:
            return jsonify({"job_id": job_id, "status": "stop_unavailable"}), 409
        first_request = not runtime.stop_event.is_set()
        if first_request:
            runtime.stop_requested_monotonic = time.monotonic()
        runtime.stop_event.set()
        job.status = "stop_requested"

    if first_request:
        print(f"[person_creation] stop requested job={job_id}")
    return jsonify({"job_id": job_id, "status": "stop_requested"})


@app.get("/api/person/status/<job_id>")
def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        job_status = job.status
        job_node = job.node
        job_error = job.error
        snap = deepcopy(job.snapshot)
    safe_snap = {
        "quality_body_crops":  snap.get("quality_body_crops", []),
        "quality_face_crops":  snap.get("quality_face_crops", []),
        "frame_groups":        snap.get("frame_groups", []),
        "associations":        snap.get("associations", []),
        "identity_clusters":   snap.get("identity_clusters", []),
        "unresolved_faces":    snap.get("unresolved_faces", []),
        "unattached_bodies":   snap.get("unattached_bodies", []),
        "per_cluster_profiles": snap.get("per_cluster_profiles", {}),
        "best_body_crops":     snap.get("best_body_crops", []),
        "per_cluster_best_body_crops": snap.get("per_cluster_best_body_crops", {}),
        "reid_crop_counts":   snap.get("reid_crop_counts", {}),
        "reid_reasons":       snap.get("reid_reasons", {}),
        "reid_unavailable_reason": snap.get("reid_unavailable_reason", ""),
        "clothing_structured": snap.get("clothing_structured", {}),
        "clothing_raw":        snap.get("clothing_raw", ""),
        "per_cluster_clothing": snap.get("per_cluster_clothing", {}),
        "clothing_diagnostics": snap.get("clothing_diagnostics", []),
        "profile":             snap.get("profile", {}),
        "human_feedback_path": snap.get("human_feedback_path", ""),
        "source_type":        snap.get("source_type", "video_file"),
        "camera_id":          snap.get("camera_id"),
        "duration_seconds":   snap.get("duration_seconds"),
        "source_uri_masked":  snap.get("source_uri_masked", ""),
        "stream_stats":       snap.get("stream_stats", {}),
        "stream_report_path": snap.get("stream_report_path", ""),
        "chunk_index":        snap.get("chunk_index"),
        "completed_chunks":   snap.get("completed_chunks", 0),
        "last_chunk":         snap.get("last_chunk"),
        "session_totals":     snap.get("session_totals", {}),
        "stop_requested":     snap.get("stop_requested", False),
        "continuous":         snap.get("continuous", False),
        "duration_seconds_per_chunk": snap.get("duration_seconds_per_chunk"),
        "live_preprocessing": snap.get("live_preprocessing", {}),
        "rolling_analysis": _live_vlm_status(snap.get("rolling_analysis", {})),
        "effective_configuration": snap.get("effective_configuration", {}),
        "live_finalization_timings": snap.get(
            "live_finalization_timings",
            {},
        ),
        "media_lifecycle_version": snap.get("media_lifecycle_version", 0),
        "media_cleanup_warning": snap.get("media_cleanup_warning", ""),
    }
    return jsonify({
        "job_id":   job_id,
        "status":   job_status,
        "node":     job_node,
        "error":    job_error,
        "snapshot": _sanitize_media_references(
            _embedding_free_public_projection(safe_snap)
        ),
    })


@app.delete("/api/person/crop/<job_id>")
def delete_crop(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        body = request.get_json(silent=True) or {}
        path_str = str(body.get("path") or "")
        crop_type = str(body.get("crop_type") or "body")
        if crop_type not in {"body", "face"}:
            return jsonify({"error": "invalid crop type"}), 400

        snap = job.snapshot
        expected = {
            public_path: str(item.get("path"))
            for item in snap.get(
                "quality_body_crops" if crop_type == "body" else "quality_face_crops",
                [],
            )
            if item.get("path")
            and (public_path := _public_media_reference(item.get("path"))) is not None
        }
        if path_str not in expected:
            return jsonify({"error": "crop does not belong to this job"}), 403
        try:
            recorded_path = expected[path_str]
            target = Path(recorded_path).resolve(strict=True)
            output_root = Path(job.output_dir).resolve(strict=False)
            target.relative_to(output_root)
            if target.parent.name != f"{crop_type}_crops":
                raise MediaPathError("unexpected crop location")
            if target.suffix.lower() not in IMAGE_EXTENSIONS:
                raise UnsupportedMediaTypeError("unsupported crop type")
            target.unlink()
        except FileNotFoundError:
            return jsonify({"error": "crop not found"}), 404
        except (MediaPathError, UnsupportedMediaTypeError, ValueError, OSError):
            return jsonify({"error": "unsafe crop path"}), 403

        if crop_type == "body":
            snap["quality_body_crops"] = [c for c in snap.get("quality_body_crops", []) if c["path"] != recorded_path]
            snap["associations"]       = [a for a in snap.get("associations", []) if a.get("body_path") != recorded_path]
            snap["best_body_crops"]    = [p for p in snap.get("best_body_crops", []) if p != recorded_path]
            # Remove from frame_groups
            for fg in snap.get("frame_groups", []):
                fg["bodies"] = [b for b in fg.get("bodies", []) if b["path"] != recorded_path]
        else:
            snap["quality_face_crops"] = [c for c in snap.get("quality_face_crops", []) if c["path"] != recorded_path]
            snap["associations"]       = [a for a in snap.get("associations", []) if a.get("face_path") != recorded_path]
            for fg in snap.get("frame_groups", []):
                fg["faces"] = [f for f in fg.get("faces", []) if f["path"] != recorded_path]

    return jsonify({"ok": True})


@app.get("/api/person/crops/<job_id>")
def crops(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        snap = deepcopy(job.snapshot)
    return jsonify(_sanitize_media_references({
        "body_crops": snap.get("quality_body_crops", []),
        "face_crops": snap.get("quality_face_crops", []),
    }))


@app.get("/api/images")
def serve_image():
    path_str = request.args.get("path", "")
    try:
        path = resolve_media_path(
            path_str,
            allow_legacy_absolute=False,
            require_exists=True,
            image_only=True,
        )
    except UnsupportedMediaTypeError:
        return jsonify({"error": "unsupported image type"}), 400
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404
    except (MediaPathError, OSError):
        return jsonify({"error": "unsafe image path"}), 403
    return send_file(str(path))


# --- Global Memory read endpoints -------------------------------------------------

@app.get("/api/memory/persons")
def memory_persons():
    from forensics.global_memory import GlobalMemory

    include_inactive = request.args.get("include_inactive", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    gm = GlobalMemory(read_only=True)
    try:
        return jsonify(gm.list_all(include_inactive=include_inactive))
    finally:
        gm.close()


@app.get("/api/memory/persons/<person_id>")
def memory_person_detail(person_id):
    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory(read_only=True)
    try:
        person = gm.get_person(person_id)
        if person is None:
            return jsonify({"error": "not found"}), 404
        history = gm.get_recognition_history(person_id=person_id, limit=50)
        return jsonify({**person, "recognition_history": history})
    finally:
        gm.close()


@app.patch("/api/memory/persons/<person_id>/rename")
def memory_rename_person(person_id):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("name") or "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400
    if len(new_name) > 100:
        return jsonify({"error": "name too long (max 100 chars)"}), 400

    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory()
    try:
        person = gm.get_person(person_id)
        if person is None:
            return jsonify({"error": f"{person_id} not found"}), 404
        gm.rename_person(person_id, new_name)
        return jsonify({
            "person_id": person_id,
            "name": new_name,
            "message": f"Renamed to '{new_name}'",
        })
    finally:
        gm.close()


@app.post("/api/memory/persons/<person_id>/profile-image")
def memory_upload_profile_image(person_id):
    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory()
    try:
        person = gm.get_person(person_id)
        if person is None:
            return jsonify({"error": f"{person_id} not found"}), 404

        if "image" in request.files:
            file = request.files["image"]
            if not file or not _allowed_profile_image(file.filename or ""):
                return jsonify({"error": "Invalid file. Use JPEG or PNG."}), 400

            original = secure_filename(file.filename or "profile_image.jpg")
            ext = original.rsplit(".", 1)[1].lower()
            relative = normalize_media_path(
                f"{person_id}/profile_image.{ext}",
                allow_legacy_absolute=False,
            )
            dest = resolve_media_path(relative, require_exists=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            file.save(str(dest))
            image_path = relative

        elif request.is_json and (request.get_json(silent=True) or {}).get("path"):
            data = request.get_json(silent=True) or {}
            try:
                source = resolve_media_path(
                    str(data.get("path", "")).strip(),
                    allow_legacy_absolute=False,
                    require_exists=True,
                    image_only=True,
                )
                image_path = normalize_media_path(source)
            except UnsupportedMediaTypeError:
                return jsonify({"error": "Invalid file type. Use JPEG or PNG."}), 400
            except FileNotFoundError:
                return jsonify({"error": "Media file not found."}), 404
            except (MediaPathError, OSError):
                return jsonify({"error": "Unsafe media path."}), 403

        else:
            return jsonify({"error": "Provide 'image' file or JSON { path }"}), 400

        gm.set_profile_image(person_id, image_path, source="manual")
        return jsonify({
            "person_id": person_id,
            "profile_image": image_path,
            "source": "manual",
            "message": "Profile image updated.",
        })
    finally:
        gm.close()


@app.post("/api/memory/persons/<person_id>/profile-image/auto")
def memory_auto_profile_image(person_id):
    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory()
    try:
        person = gm.get_person(person_id)
        if person is None:
            return jsonify({"error": f"{person_id} not found"}), 404

        force = request.args.get("force", "false").lower() == "true"
        if person.get("profile_image_source") == "manual" and not force:
            return jsonify({
                "error": "Profile image was manually set. Pass ?force=true to override.",
            }), 409

        best = gm.get_best_face_crop(person_id)
        if best is None:
            return jsonify({"error": "No face crops found in recognition log."}), 404

        gm.set_profile_image(person_id, best["path"], source="auto")
        return jsonify({
            "person_id": person_id,
            "profile_image": best["path"],
            "sharpness": best["sharpness"],
            "source": "auto",
        })
    finally:
        gm.close()


@app.get("/api/memory/persons/<person_id>/gallery")
def memory_person_gallery(person_id):
    from forensics.global_memory import GlobalMemory

    crop_type = request.args.get("type")
    if crop_type and crop_type not in {"face", "body"}:
        return jsonify({"error": "type must be face or body"}), 400

    gm = GlobalMemory(read_only=True)
    try:
        if gm.get_person(person_id) is None:
            return jsonify({"error": f"{person_id} not found"}), 404
        return jsonify(gm.get_gallery(person_id, crop_type=crop_type))
    finally:
        gm.close()


@app.get("/api/memory/log")
def memory_log():
    from forensics.global_memory import GlobalMemory

    person_id = request.args.get("person_id")
    gm = GlobalMemory(read_only=True)
    try:
        return jsonify(gm.get_recognition_history(person_id=person_id, limit=100))
    finally:
        gm.close()


@app.get("/api/memory/search")
def memory_search():
    from forensics.global_memory import GlobalMemory

    q = request.args.get("q", "").strip().lower()
    gm = GlobalMemory(read_only=True)
    try:
        persons = gm.list_all()
        if not q:
            return jsonify(persons)
        results = [
            p for p in persons
            if q in (p.get("name") or "").lower() or q in (p.get("person_id") or "").lower()
        ]
        return jsonify(results)
    finally:
        gm.close()


# ─── Profile management endpoints ─────────────────────────────────────────────

# --- Supervisor identity-review endpoints ---------------------------------------

def _identity_review_memory(*, read_only: bool = False):
    from forensics.global_memory import GlobalMemory

    return GlobalMemory(
        read_only=(
            read_only or bool(app.config.get("GLOBAL_MEMORY_READ_ONLY", False))
        )
    )


def _identity_review_error(error: BaseException):
    from forensics.global_memory import (
        IdentityReviewError,
        InvalidReviewRequestError,
        PersonMergeError,
        ReviewSuggestionConflictError,
        ReviewSuggestionIntegrityError,
        ReviewSuggestionNotFoundError,
        ReviewSuggestionStaleError,
    )
    from forensics.global_memory.store import ReadOnlyGlobalMemoryError

    if _is_sqlite_busy_error(error):
        return jsonify({"error": "global memory is temporarily busy"}), 503
    if isinstance(error, ReviewSuggestionNotFoundError):
        return jsonify({"error": str(error)}), 404
    if isinstance(error, InvalidReviewRequestError):
        return jsonify({"error": str(error)}), 400
    if isinstance(
        error,
        (
            ReviewSuggestionConflictError,
            ReviewSuggestionStaleError,
            ReviewSuggestionIntegrityError,
            PersonMergeError,
        ),
    ):
        return jsonify({"error": str(error)}), 409
    if isinstance(error, ReadOnlyGlobalMemoryError):
        return jsonify({"error": "identity reviews are read-only"}), 503
    if isinstance(error, IdentityReviewError):
        return jsonify({"error": str(error)}), 400
    return jsonify({"error": "identity review operation failed"}), 500


def _identity_review_pagination():
    from forensics.global_memory.review import InvalidReviewRequestError
    from forensics.global_memory.store import (
        IDENTITY_REVIEW_DEFAULT_LIMIT,
        IDENTITY_REVIEW_MAX_OFFSET,
    )

    allowed = {"limit", "offset"}
    unsupported = sorted(set(request.args) - allowed)
    if unsupported:
        raise InvalidReviewRequestError(
            "unsupported query parameters: " + ", ".join(unsupported)
        )
    values = {}
    for field, default in (
        ("limit", IDENTITY_REVIEW_DEFAULT_LIMIT),
        ("offset", 0),
    ):
        raw_values = request.args.getlist(field)
        if not raw_values:
            values[field] = default
            continue
        if len(raw_values) != 1:
            raise InvalidReviewRequestError(
                f"{field} must be provided exactly once"
            )
        raw = raw_values[0]
        if not re.fullmatch(r"[0-9]+", raw):
            raise InvalidReviewRequestError(
                f"{field} must be an unsigned decimal integer"
            )
        if len(raw) > len(str(IDENTITY_REVIEW_MAX_OFFSET)):
            raise InvalidReviewRequestError(
                f"{field} exceeds the maximum supported integer"
            )
        try:
            value = int(raw)
        except ValueError as exc:
            raise InvalidReviewRequestError(
                f"{field} must be an unsigned decimal integer"
            ) from exc
        if value > IDENTITY_REVIEW_MAX_OFFSET:
            raise InvalidReviewRequestError(
                f"{field} exceeds the maximum supported integer"
            )
        values[field] = value
    return values["limit"], values["offset"]


def _identity_review_request_body(*, allow_reason: bool) -> dict[str, Any]:
    from forensics.global_memory.review import InvalidReviewRequestError

    if request.data:
        if not request.is_json:
            raise InvalidReviewRequestError("request body must be JSON")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise InvalidReviewRequestError("request body must be a JSON object")
    else:
        data = {}
    allowed = {"decision_source"}
    if allow_reason:
        allowed.add("reason")
    unsupported = sorted(set(data) - allowed)
    if unsupported:
        raise InvalidReviewRequestError(
            "unsupported request fields: " + ", ".join(unsupported)
        )
    return data


@app.get("/api/identity-reviews")
def identity_review_queue():
    gm = None
    try:
        limit, offset = _identity_review_pagination()
        gm = _identity_review_memory(read_only=True)
        reviews = gm.list_pending_identity_reviews(limit=limit, offset=offset)
        return jsonify({
            "reviews": [review.as_dict() for review in reviews],
            "pending_count": gm.count_pending_identity_reviews(),
            "limit": limit,
            "offset": offset,
        })
    except Exception as exc:
        return _identity_review_error(exc)
    finally:
        if gm is not None:
            gm.close()


@app.get("/api/identity-reviews/<suggestion_id>")
def identity_review_detail(suggestion_id):
    gm = None
    try:
        gm = _identity_review_memory(read_only=True)
        detail = gm.get_identity_review(suggestion_id)
        return jsonify(_sanitize_media_references(detail.as_dict()))
    except Exception as exc:
        return _identity_review_error(exc)
    finally:
        if gm is not None:
            gm.close()


def _resolve_identity_review_request(suggestion_id: str, decision: str):
    gm = None
    try:
        data = _identity_review_request_body(allow_reason=decision == "accept")
        gm = _identity_review_memory()
        result = gm.resolve_identity_review(
            suggestion_id,
            decision,
            reason=data.get("reason"),
            decision_source=data.get("decision_source"),
        )
        return jsonify(result.as_dict())
    except Exception as exc:
        return _identity_review_error(exc)
    finally:
        if gm is not None:
            gm.close()


@app.post("/api/identity-reviews/<suggestion_id>/accept")
def identity_review_accept(suggestion_id):
    return _resolve_identity_review_request(suggestion_id, "accept")


@app.post("/api/identity-reviews/<suggestion_id>/reject")
def identity_review_reject(suggestion_id):
    return _resolve_identity_review_request(suggestion_id, "reject")


def _profile_dir(name: str) -> Path:
    return _PIConfig.load().PROFILE_ROOT / name


def _load_profile_json(name: str) -> dict | None:
    p = _profile_dir(name) / "profile.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _valid_profile_name(name: str) -> bool:
    return bool(_PROFILE_NAME_RE.fullmatch(name or ""))


@app.get("/api/profiles")
def list_profiles():
    """Each folder under PROFILE_ROOT is a distinct profile. The folder name
    is the canonical id (it must be — same folder name would collide on disk).
    profile.json's own "id" field is only displayed as a label, never used
    for routing.
    """
    root = _PIConfig.load().PROFILE_ROOT
    profiles = []
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            prof = _load_profile_json(d.name)
            if not prof:
                continue
            profiles.append({
                "id":               d.name,
                "name":             prof.get("name", d.name),
                "profile_id":       prof.get("id", d.name),
                "created_at":       prof.get("created_at", ""),
                "face_crop_count":  len(prof.get("face_crops", []) or []),
                "body_crop_count":  len(prof.get("body_crops", []) or []),
            })
    return jsonify({"profiles": profiles})


@app.get("/api/profiles/<name>")
def profile_detail(name: str):
    if not _valid_profile_name(name):
        return jsonify({"error": "invalid profile name"}), 400
    prof = _load_profile_json(name)
    if prof is None:
        return jsonify({"error": "profile not found"}), 404

    pdir = _profile_dir(name)
    referenced = _ref_basenames(prof)
    body_files = _list_jpgs(pdir / "body_crops")
    face_files = _list_jpgs(pdir / "face_crops")
    body_on_disk = len(body_files)
    face_on_disk = len(face_files)
    orphan_body = sum(1 for p in body_files if p.name not in referenced)
    orphan_face = sum(1 for p in face_files if p.name not in referenced)
    has_iphone = any(
        raw.replace("\\", "/").rsplit("/", 1)[-1].startswith("iphone_")
        for raw in (prof.get("face_crops") or [])
    )
    return jsonify(_sanitize_media_references({
        "id":                     name,                      # folder slug = canonical id
        "profile_id":             prof.get("id", name),      # profile.json's claimed id (display only)
        "name":                   prof.get("name", name),
        "created_at":             prof.get("created_at", ""),
        "media": {
            "profile_image":      prof.get("profile_image"),
            "face_crops":         prof.get("face_crops", []),
            "body_crops":         prof.get("body_crops", []),
            "best_body_crops":    prof.get("best_body_crops", []),
        },
        "face_crops_referenced":  len(prof.get("face_crops", []) or []),
        "body_crops_referenced":  len(prof.get("body_crops", []) or []),
        "best_body_crops":        len(prof.get("best_body_crops", []) or []),
        "face_crops_on_disk":     face_on_disk,
        "body_crops_on_disk":     body_on_disk,
        "orphan_face":            orphan_face,
        "orphan_body":            orphan_body,
        "has_iphone_photos":      has_iphone,
        "appearance":             prof.get("appearance", {}),
    }))


@app.post("/api/profiles/<name>/cleanup")
def profile_cleanup(name: str):
    if not _valid_profile_name(name):
        return jsonify({"error": "invalid profile name"}), 400
    body = request.get_json(silent=True) or {}
    dry_run = bool(body.get("dry_run", True))
    try:
        result = _cleanup_orphan_crops(name, dry_run=dry_run)
    except _CleanupError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@app.post("/api/profiles/<name>/add-face-photos")
def profile_add_face_photos(name: str):
    if not _valid_profile_name(name):
        return jsonify({"error": "invalid profile name"}), 400
    body = request.get_json(silent=True) or {}
    images_dir = (body.get("images_dir") or "").strip()
    if not images_dir:
        return jsonify({"error": "images_dir is required"}), 400
    p = Path(images_dir)
    if not p.is_dir():
        return jsonify({"error": f"images_dir is not a directory: {images_dir}"}), 400

    replace = bool(body.get("replace", False))
    dry_run = bool(body.get("dry_run", True))
    try:
        result = _add_face_photos(name, p, replace=replace, dry_run=dry_run)
    except _AddFacePhotosError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


if __name__ == "__main__":
    initialize_global_memory()
    app.run(host="0.0.0.0", port=5009, debug=False)
