import json
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from langgraph.types import Command
from werkzeug.utils import secure_filename

import cv2 as _cv2

from forensics.person_identifier.config import Config as _PIConfig
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
from forensics.person_creation.models.device import log_device_info_once
from forensics.person_creation.global_memory.config import (
    FACE_AUTO_MATCH_THRESHOLD,
    FACE_NO_MATCH_THRESHOLD,
)
from forensics.person_creation.global_memory.media_paths import resolve_media_path

_GM_COMPARE_THRESHOLD = FACE_AUTO_MATCH_THRESHOLD
_UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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


def _used_output_dir(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    if (path / "session_report.json").exists():
        return True
    return any(p.is_dir() and p.name.startswith("cluster_") for p in path.iterdir())


_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = Flask(__name__)
CORS(app)
_DEVICE_INFO = log_device_info_once()


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
    "load_models":       "loading_models",
    "process_video":     "processing_video",
    "filter_quality":    "filtering",
    "embed_all_faces":   "embedding",
    "cluster_identities": "clustering",
    "assign_bodies_to_clusters": "auto_pairing",
    "select_best":       "selecting",
    "describe_clothing": "describing",
    "build_profile":     "awaiting_review",
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
            job.node = "build_profile"
            job.snapshot["profile_preview"] = interrupt_val.get("profile_preview", {})
            job.resume_event.wait()
            job.resume_event.clear()
            review_resume = job.resume_value or {"approved": True, "corrections": None}
            _stream_until_interrupt(Command(resume=review_resume))

        job.status = "done"

    except Exception:
        job.status = "error"
        job.error = traceback.format_exc()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "device": _DEVICE_INFO})


@app.post("/api/person/start")
def start():
    body = request.get_json(force=True)
    name = body.get("name", "").strip()
    video_paths = body.get("video_paths", [])
    output_dir = body.get("output_dir", f"forensics/person_db/{name.lower()}")
    every_n = int(body.get("every_n", 15))
    identity_config = body.get("identity_clustering_config", {})

    if not name or not video_paths:
        return jsonify({"error": "name and video_paths required"}), 400

    output_path = Path(output_dir)
    if _used_output_dir(output_path):
        return jsonify({
            "error": "Output directory already contains a previous run. Choose a new output directory."
        }), 400

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
        "output_dir": str(output_path),
        "process_every_n": every_n,
        "identity_clustering_config": identity_config,
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
        "identity_clusters":   snap.get("identity_clusters", []),
        "unresolved_faces":    snap.get("unresolved_faces", []),
        "unattached_bodies":   snap.get("unattached_bodies", []),
        "per_cluster_profiles": snap.get("per_cluster_profiles", {}),
        "profile_preview":     snap.get("profile_preview", {}),
        "best_body_crops":     snap.get("best_body_crops", []),
        "per_cluster_best_body_crops": snap.get("per_cluster_best_body_crops", {}),
        "clothing_structured": snap.get("clothing_structured", {}),
        "clothing_raw":        snap.get("clothing_raw", ""),
        "per_cluster_clothing": snap.get("per_cluster_clothing", {}),
        "profile":             snap.get("profile", {}),
        "human_feedback_path": snap.get("human_feedback_path", ""),
        "global_memory":       snap.get("global_memory", {}),
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
    path = resolve_media_path(path_str)
    if path is None:
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


def _global_memory_store():
    from forensics.person_creation.global_memory import GlobalMemoryStore

    return GlobalMemoryStore()


def _strip_embedding_from_person_detail(detail: dict) -> dict:
    safe = dict(detail)
    person = dict(safe.get("person") or {})
    if "face_embedding" in person:
        person["face_embedding_dim"] = len(person.get("face_embedding") or [])
        person.pop("face_embedding", None)
    safe["person"] = person
    return safe


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _safe_upload_folder(name: str) -> Path:
    slug = secure_filename(name.strip()) or "person"
    return Path("forensics/person_db/_global_memory_uploads") / f"{slug}_{uuid.uuid4().hex[:8]}"


def _save_uploaded_photos(files, name: str) -> list[str]:
    upload_dir = _safe_upload_folder(name)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for idx, file in enumerate(files):
        filename = secure_filename(file.filename or "")
        if not filename:
            raise ValueError("one uploaded photo has no filename")
        suffix = Path(filename).suffix.lower()
        if suffix not in _UPLOAD_EXTS:
            raise ValueError(f"unsupported image extension for {filename}; allowed: jpg, jpeg, png, webp")
        out_path = upload_dir / f"{idx:03d}_{filename}"
        file.save(str(out_path))
        saved.append(str(out_path))

    return saved


# â”€â”€â”€ Global Memory endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/api/global-memory/register-face-photos")
def global_memory_register_face_photos():
    name = (request.form.get("name") or "").strip()
    notes = (request.form.get("notes") or "").strip()
    photos = [p for p in request.files.getlist("photos") if p and p.filename]

    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400
    if not photos:
        return jsonify({"ok": False, "error": "at least one photo is required"}), 400

    try:
        image_paths = _save_uploaded_photos(photos, name)
        if notes:
            notes_path = Path(image_paths[0]).parent / "notes.json"
            notes_path.write_text(json.dumps({"name": name, "notes": notes}, indent=2), encoding="utf-8")

        with _global_memory_store() as store:
            result = store.register_face_photo_identity_with_result(name, image_paths)
            person_id = result["person_id"]
            if notes:
                store.update_person_details(person_id, notes=notes)
            detail = store.get_person(person_id) or {}
            person = detail.get("person") or {}
            return jsonify({
                "ok": True,
                "person_id": person_id,
                "name": person.get("name", name),
                "action": result.get("action", "created"),
                "identity_source": person.get("identity_source", ""),
                "image_count": len(image_paths),
                "possible_duplicate": bool(result.get("possible_duplicate")),
                "suggestion_id": result.get("suggestion_id"),
                "best_match": result.get("best_match"),
                "review_threshold": result.get("review_threshold", FACE_NO_MATCH_THRESHOLD),
                "auto_match_threshold": result.get("auto_match_threshold", FACE_AUTO_MATCH_THRESHOLD),
                "message": "Created new phone-photo identity",
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/global-memory/persons")
def global_memory_persons():
    try:
        with _global_memory_store() as store:
            return jsonify({
                "ok": True,
                "db_path": str(store.db_path),
                "persons": store.list_persons(),
                "review_threshold": FACE_NO_MATCH_THRESHOLD,
                "auto_match_threshold": FACE_AUTO_MATCH_THRESHOLD,
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/global-memory/persons/<person_id>")
def global_memory_person_detail(person_id: str):
    try:
        with _global_memory_store() as store:
            detail = store.get_person(person_id)
            if detail is None:
                return jsonify({"ok": False, "error": "person not found"}), 404
            return jsonify({"ok": True, "person": _strip_embedding_from_person_detail(detail)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/global-memory/persons/<person_id>/add-face-photos")
def global_memory_add_face_photos_to_person(person_id: str):
    photos = [p for p in request.files.getlist("photos") if p and p.filename]
    if not photos:
        return jsonify({"ok": False, "error": "at least one photo is required"}), 400

    try:
        with _global_memory_store() as store:
            existing = store.get_person(person_id)
            if existing is None:
                return jsonify({"ok": False, "error": "person not found"}), 404
            person_name = (existing.get("person") or {}).get("name", person_id)
            image_paths = _save_uploaded_photos(photos, person_name)
            result = store.add_face_photos_to_person(person_id, image_paths)
            detail = store.get_person(person_id) or {}
            person = detail.get("person") or {}
            return jsonify({
                "ok": True,
                "person_id": person_id,
                "name": person.get("name", person_name),
                "action": result.get("action", "updated_target"),
                "identity_source": person.get("identity_source", ""),
                "image_count": len(image_paths),
                "message": "Added phone photos to existing identity",
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.patch("/api/global-memory/persons/<person_id>")
def global_memory_update_person(person_id: str):
    body = request.get_json(silent=True) or {}
    try:
        with _global_memory_store() as store:
            person = store.update_person_details(
                person_id,
                name=body.get("name"),
                notes=body.get("notes"),
            )
            person.pop("face_embedding", None)
            return jsonify({"ok": True, "person": person})
    except KeyError:
        return _json_error("person not found", 404)
    except Exception as exc:
        return _json_error(str(exc), 400)


@app.post("/api/global-memory/persons/merge")
def global_memory_merge_persons():
    body = request.get_json(silent=True) or {}
    source_person_id = (body.get("source_person_id") or "").strip()
    target_person_id = (body.get("target_person_id") or "").strip()
    new_name = body.get("new_name")
    if not source_person_id or not target_person_id:
        return _json_error("source_person_id and target_person_id are required", 400)
    try:
        with _global_memory_store() as store:
            result = store.merge_persons(source_person_id, target_person_id, new_name=new_name)
            return jsonify(result)
    except KeyError as exc:
        return _json_error(str(exc), 404)
    except Exception as exc:
        return _json_error(str(exc), 400)


@app.get("/api/global-memory/suggestions")
def global_memory_suggestions():
    try:
        with _global_memory_store() as store:
            return jsonify({
                "ok": True,
                "suggestions": store.list_suggestions(status="pending"),
                "review_threshold": FACE_NO_MATCH_THRESHOLD,
                "auto_match_threshold": FACE_AUTO_MATCH_THRESHOLD,
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/global-memory/suggestions/<suggestion_id>/accept")
def global_memory_accept_suggestion(suggestion_id: str):
    try:
        with _global_memory_store() as store:
            return jsonify(store.accept_suggestion(suggestion_id))
    except KeyError:
        return _json_error("suggestion not found", 404)
    except Exception as exc:
        return _json_error(str(exc), 400)


@app.post("/api/global-memory/suggestions/<suggestion_id>/reject")
def global_memory_reject_suggestion(suggestion_id: str):
    try:
        with _global_memory_store() as store:
            return jsonify(store.reject_suggestion(suggestion_id))
    except KeyError:
        return _json_error("suggestion not found", 404)
    except Exception as exc:
        return _json_error(str(exc), 400)


@app.post("/api/global-memory/compare-profile")
def global_memory_compare_profile():
    body = request.get_json(silent=True) or {}
    profile_raw = (body.get("profile_path") or "").strip()
    if not profile_raw:
        return jsonify({"ok": False, "error": "profile_path is required"}), 400
    profile_path = Path(profile_raw)
    if not profile_path.exists() or not profile_path.is_file():
        return jsonify({"ok": False, "error": f"profile_path not found: {profile_path}"}), 400

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to read profile JSON: {exc}"}), 400

    embedding = profile.get("face_embedding")
    if not embedding:
        return jsonify({"ok": False, "error": "profile has no face_embedding"}), 400

    try:
        with _global_memory_store() as store:
            matches = store.search_by_face(embedding, top_k=10, threshold=None)
            enriched = [
                {
                    **m,
                    "passes_threshold": float(m.get("similarity", 0.0)) >= _GM_COMPARE_THRESHOLD,
                }
                for m in matches
            ]
            return jsonify({
                "ok": True,
                "db_path": str(store.db_path),
                "profile_path": str(profile_path),
                "threshold": _GM_COMPARE_THRESHOLD,
                "review_threshold": FACE_NO_MATCH_THRESHOLD,
                "auto_match_threshold": FACE_AUTO_MATCH_THRESHOLD,
                "matches": enriched,
                "best_match_passes_threshold": bool(enriched and enriched[0]["passes_threshold"]),
            })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


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
