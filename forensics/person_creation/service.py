import json
import re
import sys
import threading
import traceback
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from langgraph.types import Command

import cv2 as _cv2

#in the future try to fix this 
#mohamed:
#I think that removing forensics.person_creation will make the code work
try:
    from forensics.person_identifier.config import Config as _PIConfig
except ModuleNotFoundError as exc:
    if exc.name not in {
        "forensics.person_identifier",
        "forensics.person_identifier.config",
    }:
        raise

    class _PIConfig:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        PROFILE_ROOT = PROJECT_ROOT / "forensics" / "person_db"

        @classmethod
        def load(cls):
            return cls()

    _pi_pkg = types.ModuleType("forensics.person_identifier")
    _pi_pkg.__path__ = []
    _pi_config_mod = types.ModuleType("forensics.person_identifier.config")
    _pi_config_mod.Config = _PIConfig
    sys.modules.setdefault("forensics.person_identifier", _pi_pkg)
    sys.modules.setdefault("forensics.person_identifier.config", _pi_config_mod)

from forensics.person_creation.path_utils import to_wsl_path as _to_wsl_path
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

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = Flask(__name__)
CORS(app)


@dataclass
class JobState:
    job_id: str
    status: str = "idle"
    # idle | loading_models | processing_video | filtering | embedding | clustering
    # | auto_pairing | selecting | describing | awaiting_review
    # | finalizing | done | error
    node: str = ""
    error: str | None = None
    snapshot: dict = field(default_factory=dict)
    resume_event: threading.Event = field(default_factory=threading.Event)
    resume_value: dict | None = None


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
    "prepare_runtime":   "loading_models",
    "process_video":     "processing_video",
    "filter_quality":    "filtering",
<<<<<<< HEAD
    "auto_associate":    "associating",
    "track_persons":     "tracking",
    "promote_crops":     "promoting",
    "embed_faces":       "embedding",
    "human_in_the_loop": "awaiting_pairing",
    "select_best":       "selecting",
    "select_best_per_person": "selecting",
=======
    "embed_all_faces":   "embedding",
    "cluster_identities": "clustering",
    "assign_bodies_to_clusters": "auto_pairing",
    "select_best":       "selecting",
    "compute_reid":      "computing_reid",
>>>>>>> Khalifa_branch
    "describe_clothing": "describing",
    "describe_clothing_per_person": "describing",
    "build_profile":     "awaiting_review",
    "build_multi_profile": "awaiting_review",
    "finalize":          "finalizing",
}


def _run_pipeline(job_id: str, initial_state: dict, config: dict) -> None:
    job = _jobs[job_id]
    graph = _get_graph()

    def _stream_until_interrupt(input_val):
        """Stream events, update job state, return interrupt value or None."""
        for event in graph.stream(input_val, config, stream_mode="updates"):
            for node_name, update in event.items():
                if node_name == "__interrupt__":
                    return update[0].value
                job.node = node_name
                job.status = _NODE_TO_STATUS.get(node_name, node_name)
                if isinstance(update, dict):
                    job.snapshot.update(update)
        return None

    try:
        # --- Run to the profile-review interrupt (pairing is fully automatic) ---
        interrupt_val = _stream_until_interrupt(initial_state)

        if interrupt_val and "profile_preview" in interrupt_val:
            job.status = "awaiting_review"
<<<<<<< HEAD
            job.node = "build_multi_profile"
=======
            job.node = "build_profile"
            job.snapshot["profile_preview"] = interrupt_val.get("profile_preview", {})
>>>>>>> Khalifa_branch
            job.resume_event.wait()
            job.resume_event.clear()
            review_resume = job.resume_value or {"approved": True, "corrections": None}
            _stream_until_interrupt(Command(resume=review_resume))

        job.status = "done"

    except Exception:
        job.status = "error"
        job.error = traceback.format_exc()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/person/start")
def start():
    body = request.get_json(force=True)
    name = body.get("name", "").strip()
    video_paths = body.get("video_paths", [])
    output_dir = body.get("output_dir", f"forensics/person_db/{name.lower()}")
    every_n = int(body.get("every_n", 15))
    identity_config = body.get("identity_clustering_config", {})
    reid_config = body.get("reid", body.get("reid_config", {}))

    if not name or not video_paths:
        return jsonify({"error": "name and video_paths required"}), 400

    # Pre-flight: normalize Windows-style paths and confirm each video opens.
    # Bad paths return 400 before we spend ~90s loading models.
    normalized: list[str] = []
    details: list[dict] = []
    for raw in video_paths:
        norm = _to_wsl_path(raw)
        reason = _validate_video_path(norm)
        if reason is not None:
            details.append({"input": raw, "normalized": norm, "reason": reason})
        else:
            normalized.append(norm)
    if details:
        return jsonify({
            "error": "video_paths failed validation",
            "details": details,
        }), 400

    job_id = str(uuid.uuid4())
    job = JobState(job_id=job_id)
    _jobs[job_id] = job

    initial_state = {
        "person_name": name,
        "video_paths": normalized,
        "output_dir": str(Path(output_dir)),
        "process_every_n": every_n,
        "identity_clustering_config": identity_config,
        "reid_config": reid_config,
        "body_crops": [],
        "face_crops": [],
    }
    config = {"configurable": {"thread_id": job_id}}

    t = threading.Thread(target=_run_pipeline, args=(job_id, initial_state, config), daemon=True)
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
<<<<<<< HEAD
        "person_tracks":       snap.get("person_tracks", []),
        "best_body_crops":     snap.get("best_body_crops", []),
        "best_body_crops_by_person": snap.get("best_body_crops_by_person", {}),
=======
        "identity_clusters":   snap.get("identity_clusters", []),
        "unresolved_faces":    snap.get("unresolved_faces", []),
        "unattached_bodies":   snap.get("unattached_bodies", []),
        "per_cluster_profiles": snap.get("per_cluster_profiles", {}),
        "profile_preview":     snap.get("profile_preview", {}),
        "best_body_crops":     snap.get("best_body_crops", []),
        "per_cluster_best_body_crops": snap.get("per_cluster_best_body_crops", {}),
        "reid_embeddings":    snap.get("reid_embeddings", {}),
        "reid_crop_counts":   snap.get("reid_crop_counts", {}),
        "reid_reasons":       snap.get("reid_reasons", {}),
        "reid_unavailable_reason": snap.get("reid_unavailable_reason", ""),
>>>>>>> Khalifa_branch
        "clothing_structured": snap.get("clothing_structured", {}),
        "clothing_by_person":  snap.get("clothing_by_person", {}),
        "clothing_raw":        snap.get("clothing_raw", ""),
<<<<<<< HEAD
        "clothing_raw_by_person": snap.get("clothing_raw_by_person", {}),
=======
        "per_cluster_clothing": snap.get("per_cluster_clothing", {}),
>>>>>>> Khalifa_branch
        "profile":             snap.get("profile", {}),
        "human_feedback_path": snap.get("human_feedback_path", ""),
    }
    return jsonify({
        "job_id":   job_id,
        "status":   job.status,
        "node":     job.node,
        "error":    job.error,
        "snapshot": safe_snap,
    })


@app.post("/api/person/approve/<job_id>")
def approve(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job.status != "awaiting_review":
        return jsonify({"error": "job not awaiting review"}), 400

    body = request.get_json(force=True)
    job.resume_value = {
        "approved":          True,
        "corrections":       body.get("corrections"),
        "clothing_override": body.get("clothing_override"),
    }
    job.resume_event.set()
    return jsonify({"ok": True})


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
