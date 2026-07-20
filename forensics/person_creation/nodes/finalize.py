import json
import os
import shutil
from pathlib import Path

from forensics.media_paths import MediaPathError, get_media_root, normalize_media_path


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
        if not p.is_file():
            continue
        try:
            media_path = normalize_media_path(p, require_exists=True)
        except (MediaPathError, FileNotFoundError, OSError):
            continue
        if media_path in keep:
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


def _copy_or_merge_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            continue
        if item.is_file():
            shutil.copy2(item, target)


def _remove_copied_source(src: Path) -> None:
    if not src.exists():
        return
    for item in src.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
    try:
        src.rmdir()
    except OSError:
        pass


def _remap_crop_paths(paths: list[str], person_dir: Path, crop_dir: str) -> list[str]:
    return [
        normalize_media_path(person_dir / crop_dir / Path(p).name, require_exists=True)
        for p in (paths or [])
        if p and (person_dir / crop_dir / Path(p).name).is_file()
    ]


def _remap_sharpness_map(sharpness: dict, person_dir: Path, crop_dir: str) -> dict:
    remapped: dict[str, float] = {}
    for raw_path, value in (sharpness or {}).items():
        if not raw_path:
            continue
        target = person_dir / crop_dir / Path(raw_path).name
        if not target.is_file():
            continue
        new_path = normalize_media_path(target, require_exists=True)
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
            target = person_dir / "body_crops" / Path(item["path"]).name
            item["path"] = (
                normalize_media_path(target, require_exists=True)
                if target.is_file()
                else None
            )
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


def _accepted_face_observation_count(profile: dict) -> int:
    """Return the accepted embedded faces represented by this profile."""
    for key in ("cluster_face_count", "face_count", "face_crop_count"):
        if profile.get(key) is not None:
            return profile[key]
    return len(profile.get("face_crops") or [])


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
        "clothing_diagnostics": state.get("clothing_diagnostics", []),
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
    base_db_dir = get_media_root()
    profiles = state.get("per_cluster_profiles") or {}
    finalized_profiles: dict[int, dict] = {}

    # Goal 3: register completed profiles into global memory. The DB is the
    # source of truth for identity; profile.json below is only a debug export.
    from forensics.global_memory import GlobalMemory
    gm = GlobalMemory(media_root=base_db_dir)

    try:
        for raw_cid, profile in profiles.items():
            cid = int(raw_cid)
            profile = _normalize_profile_schema(profile)
            registration = gm.register_with_identity_policy(
                profile,
                observation_count=_accepted_face_observation_count(profile),
                low_confidence=bool(profile.get("low_confidence", False)),
            )
            assigned_id = registration.person_id
            profile["id"] = assigned_id
            stored_person = gm.get_person(assigned_id) or {}
            profile["name"] = stored_person.get("name") or assigned_id.replace("_", " ").title()
            person_dir = base_db_dir / assigned_id
            person_dir.mkdir(parents=True, exist_ok=True)

            cluster_dir = output_dir / f"cluster_{cid}"
            source_body = cluster_dir / "body_crops"
            source_face = cluster_dir / "face_crops"
            _copy_or_merge_dir(source_body, person_dir / "body_crops")
            _copy_or_merge_dir(source_face, person_dir / "face_crops")

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

            # Old session files remain valid until the database atomically points
            # at copied canonical files. Only then is the old copy removed.
            _remove_copied_source(source_body)
            _remove_copied_source(source_face)

            finalized_profiles[cid] = profile
            profile_path = person_dir / "profile.json"
        # profile.json is a debug artifact - human-readable export of the DB record.
        # Source of truth for all queries is forensics/global_memory.db.
        # Downstream modules (MTMC, Goal 4 face engine) must use GlobalMemory,
        # not read this file directly.
            profile_for_disk = {k: v for k, v in profile.items() if k != "face_embedding"}
            _write_json(profile_path, profile_for_disk)
            print(f"[finalize] profile saved -> {profile_path}")

            try:
                referenced = gm.referenced_media_paths(assigned_id)
                referenced.update(profile.get("face_crops") or [])
                referenced.update(profile.get("body_crops") or [])
                referenced.update(profile.get("best_body_crops") or [])
            except Exception:
                referenced = None
                print("[finalize] persistent media lookup failed; orphan pruning skipped")
            if referenced is not None:
                body_del, body_bytes = _prune_orphans(person_dir / "body_crops", referenced)
                face_del, face_bytes = _prune_orphans(person_dir / "face_crops", referenced)
                total_mb = (body_bytes + face_bytes) / (1024 * 1024)
                if body_del or face_del:
                    print(
                        f"[finalize] cluster_{cid} cleanup: deleted {body_del} orphan body / "
                        f"{face_del} orphan face crops ({total_mb:.2f} MB freed)"
                    )
    finally:
        gm.close()

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

    first_id = sorted(finalized_profiles)[0] if finalized_profiles else None
    return {
        "per_cluster_profiles": finalized_profiles,
        "profile": finalized_profiles.get(first_id, {}) if first_id is not None else {},
    }
