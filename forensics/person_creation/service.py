import json
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2 as _cv2

from forensics.person_identifier.config import Config as _PIConfig
from forensics.person_creation.path_utils import to_wsl_path as _to_wsl_path
from forensics.person_creation.live_stream import mask_camera_uri
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
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise StartRequestError(f"{field_name} must be an integer") from exc


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


def _allowed_profile_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in _ALLOWED_PROFILE_IMAGE_EXTENSIONS


@dataclass
class JobState:
    job_id: str
    status: str = "idle"
    # idle | loading_models | processing_video | filtering | embedding | clustering
    # | auto_pairing | selecting | computing_reid | describing | building_profile
    # | finalizing | done | error
    node: str = ""
    error: str | None = None
    snapshot: dict = field(default_factory=dict)


_jobs: dict[str, JobState] = {}
_graph = None
_graph_lock = threading.Lock()


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
    "select_best":       "selecting",
    "compute_reid":      "computing_reid",
    "build_profile":     "building_profile",
    "finalize":          "finalizing",
}


def _run_pipeline(job_id: str, initial_state: dict) -> None:
    """Run the graph start to end with no human interrupts."""
    job = _jobs[job_id]
    graph = _get_graph()

    def update_live_status(status: str, snapshot_update: dict | None = None) -> None:
        job.node = "process_live_stream"
        job.status = status
        if snapshot_update:
            job.snapshot.update(snapshot_update)

    raw_camera_uri = initial_state.get("camera_uri")
    status_token = None
    if initial_state.get("input_type") == "camera_uri":
        # Ingestion no longer happens inside a graph node (see
        # nodes/process_live_stream.py) -- capture the fixed-duration window here,
        # then hand the graph an already-captured segment. The raw camera_uri (which
        # may embed credentials) is dropped from state before the graph ever sees it;
        # only the masked form travels through graph state / job snapshots from here on.
        from forensics.person_creation.single_segment_capture import capture_fixed_duration_segment
        from forensics.person_creation.status_reporting import set_status_callback, reset_status_callback

        status_token = set_status_callback(update_live_status)
        try:
            segment = capture_fixed_duration_segment(
                raw_camera_uri,
                initial_state.get("duration_seconds", 30),
                status_callback=update_live_status,
            )
        except Exception:
            reset_status_callback(status_token)
            job.status = "error"
            error = traceback.format_exc()
            if raw_camera_uri:
                error = error.replace(raw_camera_uri, mask_camera_uri(raw_camera_uri))
            job.error = error
            return

        initial_state.pop("camera_uri", None)
        initial_state["segment_id"] = segment.segment_id
        initial_state["segment_incomplete"] = segment.segment_incomplete
        initial_state["segment_frames"] = segment.frames
        initial_state["segment_frame_timestamps"] = segment.frame_timestamps

    try:
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in event.items():
                job.node = node_name
                job.status = _NODE_TO_STATUS.get(node_name, node_name)
                if isinstance(update, dict):
                    job.snapshot.update(update)

        job.status = "done"

    except Exception:
        job.status = "error"
        error = traceback.format_exc()
        if raw_camera_uri:
            error = error.replace(raw_camera_uri, mask_camera_uri(raw_camera_uri))
        job.error = error
    finally:
        if status_token is not None:
            from forensics.person_creation.status_reporting import reset_status_callback
            reset_status_callback(status_token)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "waycon-person-creation"})


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
        "source_uri_masked": initial_state.get("source_uri_masked", ""),
    }
    job = JobState(job_id=job_id, snapshot=safe_initial_snapshot)
    _jobs[job_id] = job

    t = threading.Thread(target=_run_pipeline, args=(job_id, initial_state), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.get("/api/person/status/<job_id>")
def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    snap = job.snapshot
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
        "reid_embeddings":    snap.get("reid_embeddings", {}),
        "reid_crop_counts":   snap.get("reid_crop_counts", {}),
        "reid_reasons":       snap.get("reid_reasons", {}),
        "reid_unavailable_reason": snap.get("reid_unavailable_reason", ""),
        "clothing_structured": snap.get("clothing_structured", {}),
        "clothing_raw":        snap.get("clothing_raw", ""),
        "per_cluster_clothing": snap.get("per_cluster_clothing", {}),
        "profile":             snap.get("profile", {}),
        "human_feedback_path": snap.get("human_feedback_path", ""),
        "source_type":        snap.get("source_type", "video_file"),
        "camera_id":          snap.get("camera_id"),
        "duration_seconds":   snap.get("duration_seconds"),
        "source_uri_masked":  snap.get("source_uri_masked", ""),
        "stream_stats":       snap.get("stream_stats", {}),
        "stream_report_path": snap.get("stream_report_path", ""),
    }
    return jsonify({
        "job_id":   job_id,
        "status":   job.status,
        "node":     job.node,
        "error":    job.error,
        "snapshot": safe_snap,
    })


@app.delete("/api/person/crop/<job_id>")
def delete_crop(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    body = request.get_json(force=True)
    path_str = body.get("path", "")
    crop_type = body.get("crop_type", "body")

    Path(path_str).resolve().unlink(missing_ok=True)

    snap = job.snapshot
    if crop_type == "body":
        snap["quality_body_crops"] = [c for c in snap.get("quality_body_crops", []) if c["path"] != path_str]
        snap["associations"]       = [a for a in snap.get("associations", []) if a.get("body_path") != path_str]
        snap["best_body_crops"]    = [p for p in snap.get("best_body_crops", []) if p != path_str]
        # Remove from frame_groups
        for fg in snap.get("frame_groups", []):
            fg["bodies"] = [b for b in fg.get("bodies", []) if b["path"] != path_str]
    else:
        snap["quality_face_crops"] = [c for c in snap.get("quality_face_crops", []) if c["path"] != path_str]
        snap["associations"]       = [a for a in snap.get("associations", []) if a.get("face_path") != path_str]
        for fg in snap.get("frame_groups", []):
            fg["faces"] = [f for f in fg.get("faces", []) if f["path"] != path_str]

    return jsonify({"ok": True})


@app.get("/api/person/crops/<job_id>")
def crops(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    snap = job.snapshot
    return jsonify({
        "body_crops": snap.get("quality_body_crops", []),
        "face_crops": snap.get("quality_face_crops", []),
    })


@app.get("/api/images")
def serve_image():
    path_str = request.args.get("path", "")
    path = Path(path_str).resolve()
    if not path.exists() or not path.is_file():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(path))


# --- Global Memory read endpoints -------------------------------------------------

@app.get("/api/memory/persons")
def memory_persons():
    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory()
    try:
        return jsonify(gm.list_all())
    finally:
        gm.close()


@app.get("/api/memory/persons/<person_id>")
def memory_person_detail(person_id):
    from forensics.global_memory import GlobalMemory

    gm = GlobalMemory()
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

            person_dir = Path("forensics/person_db") / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            original = secure_filename(file.filename or "profile_image.jpg")
            ext = original.rsplit(".", 1)[1].lower()
            dest = person_dir / f"profile_image.{ext}"
            file.save(str(dest))
            image_path = str(dest.resolve())

        elif request.is_json and (request.get_json(silent=True) or {}).get("path"):
            data = request.get_json(silent=True) or {}
            source = Path(str(data.get("path", "")).strip())
            if not source.exists() or not source.is_file():
                return jsonify({"error": f"File not found: {source}"}), 400
            if not _allowed_profile_image(str(source)):
                return jsonify({"error": "Invalid file type. Use JPEG or PNG."}), 400
            image_path = str(source.resolve())

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

    gm = GlobalMemory()
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
    gm = GlobalMemory()
    try:
        return jsonify(gm.get_recognition_history(person_id=person_id, limit=100))
    finally:
        gm.close()


@app.get("/api/memory/search")
def memory_search():
    from forensics.global_memory import GlobalMemory

    q = request.args.get("q", "").strip().lower()
    gm = GlobalMemory()
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
    return jsonify({
        "id":                     name,                      # folder slug = canonical id
        "profile_id":             prof.get("id", name),      # profile.json's claimed id (display only)
        "name":                   prof.get("name", name),
        "created_at":             prof.get("created_at", ""),
        "face_crops_referenced":  len(prof.get("face_crops", []) or []),
        "body_crops_referenced":  len(prof.get("body_crops", []) or []),
        "best_body_crops":        len(prof.get("best_body_crops", []) or []),
        "face_crops_on_disk":     face_on_disk,
        "body_crops_on_disk":     body_on_disk,
        "orphan_face":            orphan_face,
        "orphan_body":            orphan_body,
        "has_iphone_photos":      has_iphone,
        "appearance":             prof.get("appearance", {}),
    })


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
    app.run(host="0.0.0.0", port=5009, debug=False)
