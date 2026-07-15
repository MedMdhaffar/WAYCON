import json
import os
import shutil
from pathlib import Path


def _basenames(items) -> set[str]:
    out: set[str] = set()
    for raw in items or []:
        name = str(raw).replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            out.add(name)
    return out


def _prune_orphans(directory: Path, keep: set[str]) -> tuple[int, int]:
    if not directory.is_dir():
        return 0, 0
    deleted = 0
    freed = 0
    for p in directory.iterdir():
        if not p.is_file() or p.name in keep:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            deleted += 1
            freed += size
        except OSError:
            pass
    return deleted, freed


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _move_or_merge_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_file():
                try:
                    item.unlink()
                except OSError:
                    pass
            continue
        shutil.move(str(item), str(target))
    try:
        src.rmdir()
    except OSError:
        pass


def _remap_crop_paths(paths: list[str], person_dir: Path, crop_dir: str) -> list[str]:
    return [str(person_dir / crop_dir / Path(p).name) for p in (paths or []) if p]


def _remap_sharpness_map(sharpness: dict, person_dir: Path, crop_dir: str) -> dict:
    remapped: dict[str, float] = {}
    for raw_path, value in (sharpness or {}).items():
        if not raw_path:
            continue
        new_path = str(person_dir / crop_dir / Path(raw_path).name)
        try:
            remapped[new_path] = float(value)
        except (TypeError, ValueError):
            remapped[new_path] = 0.0
    return remapped


def _remap_color_sample_paths(color_signal: dict, person_dir: Path) -> dict:
    color_signal = dict(color_signal or {})
    samples = []
    for sample in color_signal.get("samples", []) or []:
        item = dict(sample)
        if item.get("path"):
            item["path"] = str(person_dir / "body_crops" / Path(item["path"]).name)
        samples.append(item)
    color_signal["samples"] = samples
    return color_signal


def _normalize_profile_schema(profile: dict) -> dict:
    profile = dict(profile)
    profile.setdefault("face_embedding_meta", {
        "model": "facenet_pytorch.InceptionResnetV1.vggface2",
        "dim": 512,
        "norm": "L2",
    })
    reid = dict(profile.get("reid") or {})
    association_meta = dict(profile.get("association_meta") or {})
    if "association_source" in reid:
        association_meta.setdefault("source", reid.pop("association_source"))
    if "association_count" in reid:
        association_meta.setdefault("association_count", reid.pop("association_count"))
    if "auto_pair_score_mean" in reid:
        association_meta.setdefault("auto_pair_score_mean", reid.pop("auto_pair_score_mean"))
    for key in ("primary_key", "embedding_model", "face_embedding_dim"):
        reid.pop(key, None)
    if not reid or "status" not in reid:
        reid = {
            "status": "not_computed",
            "reason": "no_reid_model_configured",
            "body_embedding": None,
            "note": "Reserved for body ReID embedding (OSNet or equivalent). Permanent identity is in face_embedding.",
        }
    profile["reid"] = reid
    profile["association_meta"] = association_meta
    return profile


def _session_report(state: dict, profiles_written: int) -> dict:
    clusters = state.get("identity_clusters", [])
    low_confidence = [c for c in clusters if c.get("low_confidence")]
    report = {
        "session_id": Path(state["output_dir"]).name,
        "video_sources": state.get("video_paths", []),
        "source_type": state.get("source_type", "video_file"),
        "profiles_written": profiles_written,
        "total_face_crops": state.get("total_quality_face_crops", len(state.get("quality_face_crops", []))),
        "total_body_crops": state.get("total_quality_body_crops", len(state.get("quality_body_crops", []))),
        "face_rejection_counts": state.get("face_rejection_counts", {}),
        "identity_clusters_found": len(clusters),
        "low_confidence_clusters": len(low_confidence),
        "unresolved_faces": len(state.get("unresolved_faces", [])),
        "unattached_bodies": len(state.get("unattached_bodies", [])),
        "identity_clustering_config": state.get("identity_clustering_config", {}),
        "reid_config": state.get("reid_config", {}),
    }
    if state.get("source_type") == "live_camera":
        report.update({
            "camera_uri_masked": state.get("source_uri_masked", ""),
            "camera_id": state.get("camera_id"),
            "duration_seconds": state.get("duration_seconds", 30),
            "stream_stats": state.get("stream_stats", {}),
            "stream_report_path": state.get("stream_report_path", ""),
        })
    return report


def _keep_staging_on_empty_faces(state: dict) -> bool:
    return (
        os.environ.get("PERSON_CREATION_KEEP_STAGING_ON_EMPTY_FACES") == "1"
        and len(state.get("face_crops") or []) > 0
        and state.get(
            "total_quality_face_crops",
            len(state.get("quality_face_crops") or []),
        ) == 0
    )


def _cleanup_staging(state: dict, staging: Path) -> None:
    print("[finalize] cleanup entered", flush=True)
    try:
        if staging.exists() and _keep_staging_on_empty_faces(state):
            print(
                "[finalize] preserving staging tree because detected faces were "
                "all rejected by quality filtering"
            )
        elif staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
            print(f"[finalize] removed staging tree: {staging}")
    finally:
        print("[finalize] cleanup completed", flush=True)


def finalize(state: dict) -> dict:
    """Write one profile.json per cluster, plus session and rejects reports.

    Input state:  `per_cluster_profiles`, `identity_clusters`,
                  `unresolved_faces`, `unattached_bodies`, `output_dir`.
    Output:       writes `cluster_<id>/profile.json`, `session_report.json`,
                  `rejected_detections.json`; prunes orphan crops and staging.
    """
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_db_dir = output_dir.parent if output_dir.name else Path("forensics/person_db")
    profiles = state.get("per_cluster_profiles") or {}
    finalized_profiles: dict[int, dict] = {}

    # Goal 3: register completed profiles into global memory. The DB is the
    # source of truth for identity; profile.json below is only a debug export.
    from forensics.global_memory import GlobalMemory
    gm = GlobalMemory()

    for raw_cid, profile in profiles.items():
        cid = int(raw_cid)
        profile = _normalize_profile_schema(profile)
        assigned_id = gm.register(profile)
        profile["id"] = assigned_id
        stored_person = gm.get_person(assigned_id) or {}
        profile["name"] = stored_person.get("name") or assigned_id.replace("_", " ").title()
        person_dir = base_db_dir / assigned_id
        person_dir.mkdir(parents=True, exist_ok=True)

        cluster_dir = output_dir / f"cluster_{cid}"
        _move_or_merge_dir(cluster_dir / "body_crops", person_dir / "body_crops")
        _move_or_merge_dir(cluster_dir / "face_crops", person_dir / "face_crops")

        profile["body_crops"] = _remap_crop_paths(profile.get("body_crops", []), person_dir, "body_crops")
        profile["best_body_crops"] = _remap_crop_paths(profile.get("best_body_crops", []), person_dir, "body_crops")
        profile["face_crops"] = _remap_crop_paths(profile.get("face_crops", []), person_dir, "face_crops")
        profile["body_crop_sharpness"] = _remap_sharpness_map(
            profile.get("body_crop_sharpness", {}),
            person_dir,
            "body_crops",
        )
        profile["face_crop_sharpness"] = _remap_sharpness_map(
            profile.get("face_crop_sharpness", {}),
            person_dir,
            "face_crops",
        )
        if (profile.get("appearance_signals") or {}).get("color"):
            profile["appearance_signals"]["color"] = _remap_color_sample_paths(
                profile["appearance_signals"]["color"],
                person_dir,
            )
        if profile.get("face_crops"):
            profile["profile_image"] = profile["face_crops"][0]
        gm.update_crop_paths(assigned_id, profile)

        finalized_profiles[cid] = profile
        profile_path = person_dir / "profile.json"
        # profile.json is a debug artifact - human-readable export of the DB record.
        # Source of truth for all queries is forensics/global_memory.db.
        # Downstream modules (MTMC, Goal 4 face engine) must use GlobalMemory,
        # not read this file directly.
        profile_for_disk = {k: v for k, v in profile.items() if k != "face_embedding"}
        _write_json(profile_path, profile_for_disk)
        print(f"[finalize] profile saved -> {profile_path}")

        referenced = (
            _basenames(profile.get("face_crops"))
            | _basenames(profile.get("body_crops"))
            | _basenames(profile.get("best_body_crops"))
        )
        body_del, body_bytes = _prune_orphans(person_dir / "body_crops", referenced)
        face_del, face_bytes = _prune_orphans(person_dir / "face_crops", referenced)
        total_mb = (body_bytes + face_bytes) / (1024 * 1024)
        if body_del or face_del:
            print(
                f"[finalize] cluster_{cid} cleanup: deleted {body_del} orphan body / "
                f"{face_del} orphan face crops ({total_mb:.2f} MB freed)"
            )

    rejected = {
        "unresolved_faces": state.get("unresolved_faces", []),
        "unattached_bodies": state.get("unattached_bodies", []),
    }
    _write_json(output_dir / "rejected_detections.json", rejected)
    session_report = _session_report(state, len(profiles))
    for profile in finalized_profiles.values():
        person_dir = base_db_dir / profile["id"]
        _write_json(person_dir / "session_report.json", session_report)
    if not finalized_profiles:
        _write_json(output_dir / "session_report.json", session_report)
    print(f"[finalize] session report saved")

    staging = output_dir / "_staging"
    _cleanup_staging(state, staging)

    gm.close()

    first_id = sorted(finalized_profiles)[0] if finalized_profiles else None
    return {
        "per_cluster_profiles": finalized_profiles,
        "profile": finalized_profiles.get(first_id, {}) if first_id is not None else {},
    }
