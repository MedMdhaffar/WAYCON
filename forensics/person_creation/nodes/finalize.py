import json
import shutil
from pathlib import Path

from forensics.person_creation.models.device import device_info


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


def _session_report(state: dict, profiles_written: int) -> dict:
    clusters = state.get("identity_clusters", [])
    low_confidence = [c for c in clusters if c.get("low_confidence")]
    report = {
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
        "device": state.get("device_info") or device_info(),
    }
    if "global_memory" in state:
        report["global_memory"] = state["global_memory"]
        report["global_memory_errors"] = state["global_memory"].get("errors", [])
    return report


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
    global_memory = {"db_path": "", "registrations": [], "registered_person_ids": [], "errors": []}

    try:
        from forensics.person_creation.global_memory import GlobalMemoryStore

        memory_store = GlobalMemoryStore()
        global_memory["db_path"] = str(memory_store.db_path)
    except Exception as exc:
        memory_store = None
        global_memory["errors"].append(f"open_global_memory_failed: {exc}")

    for raw_cid, profile in profiles.items():
        cid = int(raw_cid)
        profile_path = output_dir / f"cluster_{cid}" / "profile.json"
        _write_json(profile_path, profile)
        print(f"[finalize] profile saved -> {profile_path}")

        if memory_store is not None:
            try:
                if hasattr(memory_store, "register_profile_with_result"):
                    result = memory_store.register_profile_with_result(
                        profile,
                        profile_path=str(profile_path),
                        output_dir=str(output_dir),
                    )
                    person_id = result["person_id"]
                else:
                    from forensics.person_creation.global_memory.config import FACE_AUTO_MATCH_THRESHOLD

                    person_id = memory_store.register_profile(
                        profile,
                        profile_path=str(profile_path),
                        output_dir=str(output_dir),
                    )
                    result = {
                        "profile_path": str(profile_path),
                        "person_id": person_id,
                        "action": "unknown",
                        "matched": None,
                        "best_match": None,
                        "threshold": FACE_AUTO_MATCH_THRESHOLD,
                    }
                result = {"profile_path": str(profile_path), **result}
                global_memory["registrations"].append(result)
                global_memory["registered_person_ids"].append(person_id)
                print(
                    f"[finalize] global memory registered cluster_{cid} -> "
                    f"{person_id} ({result.get('action')})"
                )
            except Exception as exc:
                msg = f"cluster_{cid}: {exc}"
                global_memory["errors"].append(msg)
                print(f"[finalize] global memory registration failed for cluster_{cid}: {exc}")

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

    if memory_store is not None:
        memory_store.close()

    rejected = {
        "unresolved_faces": state.get("unresolved_faces", []),
        "unattached_bodies": state.get("unattached_bodies", []),
    }
    _write_json(output_dir / "rejected_detections.json", rejected)
    report_state = {**state, "global_memory": global_memory, "device_info": device_info()}
    _write_json(output_dir / "session_report.json", _session_report(report_state, len(profiles)))
    print(f"[finalize] session report saved -> {output_dir / 'session_report.json'}")

    staging = output_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[finalize] removed staging tree: {staging}")

    return {"global_memory": global_memory, "device_info": report_state["device_info"]}
