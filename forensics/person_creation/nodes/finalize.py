import json
import os
import re
import shutil
import uuid
from pathlib import Path

from forensics.media_paths import MediaPathError, get_media_root, normalize_media_path
from forensics.person_creation.media_lifecycle import (
    cleanup_relocated_sources,
    relocate_profile_media,
    rewrite_media_references,
    scrub_obsolete_session_media,
)


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
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


_FINAL_MEDIA_FIELDS = (
    "body_crops",
    "face_crops",
    "quality_body_crops",
    "quality_face_crops",
    "all_face_embeddings",
    "failed_face_embeddings",
    "frame_groups",
    "associations",
    "identity_clusters",
    "cluster_assignments",
    "unresolved_faces",
    "unattached_bodies",
    "best_body_crops",
    "per_cluster_best_body_crops",
    "per_cluster_clothing",
    "clothing_diagnostics",
    "rolling_analysis",
    "stream_stats",
)


def _canonical_replay_person_id(profile: dict) -> str | None:
    person_id = str(profile.get("id") or "")
    if not re.fullmatch(r"person_[0-9]+", person_id):
        return None
    paths = [
        *list(profile.get("face_crops") or []),
        *list(profile.get("body_crops") or []),
        *list(profile.get("best_body_crops") or []),
    ]
    if not paths or any(
        not str(path).replace("\\", "/").startswith(f"{person_id}/")
        for path in paths
    ):
        return None
    return person_id


def _rewrite_feedback_report(
    path: Path,
    remap: dict[str, str],
    *,
    output_dir: Path,
    media_root: Path,
) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload = rewrite_media_references(payload, remap)
    payload = scrub_obsolete_session_media(
        payload,
        output_dir=output_dir,
        media_root=media_root,
    )
    _write_json(path, payload)


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
    complete_remap: dict[str, str] = {}
    cleanup_pairs: list[tuple[str, str]] = []

    # Goal 3: register completed profiles into global memory. The DB is the
    # source of truth for identity; profile.json below is only a debug export.
    from forensics.global_memory import GlobalMemory
    gm = GlobalMemory(media_root=base_db_dir)

    try:
        for raw_cid, profile in profiles.items():
            cid = int(raw_cid)
            profile = _normalize_profile_schema(profile)
            replay_id = _canonical_replay_person_id(profile)
            relocation_holder = {}
            if replay_id is not None and gm.get_person(replay_id) is not None:
                assigned_id = replay_id
                relocation = relocate_profile_media(
                    profile,
                    assigned_id,
                    media_root=base_db_dir,
                )
                profile = relocation.profile
            else:
                def prepare_for_person(person_id: str, raw_profile: dict) -> dict:
                    relocation = relocate_profile_media(
                        raw_profile,
                        person_id,
                        media_root=base_db_dir,
                    )
                    relocation_holder["value"] = relocation
                    return relocation.profile

                registration = gm.register_with_identity_policy(
                    profile,
                    observation_count=_accepted_face_observation_count(profile),
                    low_confidence=bool(profile.get("low_confidence", False)),
                    prepare_profile_for_person=prepare_for_person,
                )
                assigned_id = registration.person_id
                relocation = relocation_holder["value"]
                profile = relocation.profile

            complete_remap.update(relocation.remap)
            for pair in relocation.cleanup_pairs:
                if pair not in cleanup_pairs:
                    cleanup_pairs.append(pair)
            profile["id"] = assigned_id
            stored_person = gm.get_person(assigned_id) or {}
            profile["name"] = stored_person.get("name") or assigned_id.replace("_", " ").title()
            person_dir = base_db_dir / assigned_id
            person_dir.mkdir(parents=True, exist_ok=True)

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

    promotion_remap = state.get("_media_path_remap") or {}
    if isinstance(promotion_remap, dict):
        for staging_path, promoted_path in promotion_remap.items():
            canonical_path = complete_remap.get(promoted_path)
            if canonical_path is not None:
                complete_remap[staging_path] = canonical_path

    rejected = rewrite_media_references({
        "unresolved_faces": state.get("unresolved_faces", []),
        "unattached_bodies": state.get("unattached_bodies", []),
    }, complete_remap)
    rejected = scrub_obsolete_session_media(
        rejected,
        output_dir=output_dir,
        media_root=base_db_dir,
    )
    _write_json(output_dir / "rejected_detections.json", rejected)
    session_report = rewrite_media_references(
        _session_report(state, len(profiles)),
        complete_remap,
    )
    session_report["stream_report_path"] = ""
    session_report = scrub_obsolete_session_media(
        session_report,
        output_dir=output_dir,
        media_root=base_db_dir,
    )
    for profile in finalized_profiles.values():
        person_dir = base_db_dir / profile["id"]
        _write_json(person_dir / "session_report.json", session_report)
    if not finalized_profiles:
        _write_json(output_dir / "session_report.json", session_report)
    print(f"[finalize] session report saved")

    _rewrite_feedback_report(
        output_dir / "pairing_feedback.json",
        complete_remap,
        output_dir=output_dir,
        media_root=base_db_dir,
    )

    first_id = sorted(finalized_profiles)[0] if finalized_profiles else None
    update = {
        "per_cluster_profiles": finalized_profiles,
        "profile": finalized_profiles.get(first_id, {}) if first_id is not None else {},
        "human_feedback_path": "",
        "stream_report_path": "",
        "_media_path_remap": complete_remap,
        "_media_cleanup_pairs": cleanup_pairs,
        "_media_finalized_root": str(output_dir),
        "media_lifecycle_version": int(state.get("media_lifecycle_version", 0)) + 1,
    }
    for field in _FINAL_MEDIA_FIELDS:
        if field in state:
            rewritten = rewrite_media_references(state.get(field), complete_remap)
            update[field] = scrub_obsolete_session_media(
                rewritten,
                output_dir=output_dir,
                media_root=base_db_dir,
            )
    return update


def cleanup_finalized_media(state: dict) -> dict:
    """Run disposable-session cleanup after canonical state publication."""
    warnings: list[str] = []
    relocated_sources_verified = True
    try:
        cleanup_relocated_sources(
            list(state.get("_media_cleanup_pairs") or []),
            media_root=get_media_root(),
        )
    except Exception as exc:
        relocated_sources_verified = False
        warnings.append(f"relocated source cleanup failed: {type(exc).__name__}")
    output_dir = Path(state["output_dir"])
    if relocated_sources_verified:
        try:
            _cleanup_staging(state, output_dir / "_staging")
            for cluster_dir in output_dir.glob("cluster_*"):
                for crop_dir in (cluster_dir / "body_crops", cluster_dir / "face_crops"):
                    try:
                        crop_dir.rmdir()
                    except OSError:
                        pass
                for generated_report in (
                    cluster_dir / "profile.json",
                    cluster_dir / "session_report.json",
                ):
                    try:
                        generated_report.unlink(missing_ok=True)
                    except OSError:
                        pass
                try:
                    cluster_dir.rmdir()
                except OSError:
                    pass
        except Exception as exc:
            warnings.append(f"session cleanup failed: {type(exc).__name__}")
    return {
        "_media_cleanup_pairs": [],
        "media_cleanup_warning": "; ".join(warnings),
    }
