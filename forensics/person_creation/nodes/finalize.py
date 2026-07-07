import json
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
    return {
        "session_id": Path(state["output_dir"]).name,
        "video_sources": state.get("video_paths", []),
        "profiles_written": profiles_written,
        "total_face_crops": state.get("total_quality_face_crops", len(state.get("quality_face_crops", []))),
        "total_body_crops": state.get("total_quality_body_crops", len(state.get("quality_body_crops", []))),
        "identity_clusters_found": len(clusters),
        "low_confidence_clusters": len(low_confidence),
        "unresolved_faces": len(state.get("unresolved_faces", [])),
        "unattached_bodies": len(state.get("unattached_bodies", [])),
        "identity_clustering_config": state.get("identity_clustering_config", {}),
        "reid_config": state.get("reid_config", {}),
    }


def finalize(state: dict) -> dict:
    """Write one profile.json per cluster, plus session and rejects reports.

    Input state:  `per_cluster_profiles`, `identity_clusters`,
                  `unresolved_faces`, `unattached_bodies`, `output_dir`.
    Output:       writes `cluster_<id>/profile.json`, `session_report.json`,
                  `rejected_detections.json`; prunes orphan crops and staging.
    """
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = state.get("per_cluster_profiles") or {}
    profile_paths: list[str] = []

    for raw_cid, profile in profiles.items():
        cid = int(raw_cid)
        profile = _normalize_profile_schema(profile)
        profile_path = output_dir / f"cluster_{cid}" / "profile.json"
        _write_json(profile_path, profile)
        profile_paths.append(str(profile_path.resolve()))
        print(f"[finalize] profile saved -> {profile_path}")

        referenced = (
            _basenames(profile.get("face_crops"))
            | _basenames(profile.get("body_crops"))
            | _basenames(profile.get("best_body_crops"))
        )
        body_del, body_bytes = _prune_orphans(profile_path.parent / "body_crops", referenced)
        face_del, face_bytes = _prune_orphans(profile_path.parent / "face_crops", referenced)
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
    _write_json(output_dir / "session_report.json", _session_report(state, len(profiles)))
    print(f"[finalize] session report saved -> {output_dir / 'session_report.json'}")

    staging = output_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[finalize] removed staging tree: {staging}")

    return {
        "profile_path": profile_paths[0] if profile_paths else "",
        "profile_paths": profile_paths,
    }
