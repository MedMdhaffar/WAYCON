from datetime import date
from pathlib import Path

from langgraph.types import interrupt

from forensics.person_creation.nodes.profile_signals import (
    build_reid_signal,
    color_signals_from_crops,
)


def _session_id(state: dict) -> str:
    return Path(state["output_dir"]).name or state.get("person_name", "session")


def _profile_for_cluster(state: dict, cluster: dict) -> dict:
    cid = int(cluster["cluster_id"])
    associations = [a for a in state.get("associations", []) if int(a.get("cluster_id", -1)) == cid]
    per_cluster_best = state.get("per_cluster_best_body_crops") or {}
    per_cluster_clothing = state.get("per_cluster_clothing") or {}
    best_body_crops = per_cluster_best.get(cid, per_cluster_best.get(str(cid), []))
    clothing = per_cluster_clothing.get(cid, per_cluster_clothing.get(str(cid), {}))
    clothing_structured = clothing.get("structured") or {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"}
    color_signals = color_signals_from_crops(best_body_crops)

    cluster_state = {
        **state,
        "associations": associations,
        "mean_face_embedding": cluster.get("representative_embedding", []),
    }
    reid_signal = build_reid_signal(cluster_state, color_signals)
    sid = _session_id(state)
    profile_id = f"person_{sid}_cluster_{cid}"

    return {
        "id": profile_id,
        "name": profile_id,
        "cluster_id": cid,
        "cluster_confidence": cluster.get("confidence", 0.0),
        "cluster_face_count": cluster.get("face_count", 0),
        "low_confidence": bool(cluster.get("low_confidence", False)),
        "face_count": cluster.get("face_count", 0),
        "created_at": date.today().isoformat(),
        "face_embedding": cluster.get("representative_embedding", []),
        "face_crop_count": cluster.get("face_count", 0),
        "face_crops": [r["crop_path"] for r in cluster.get("face_records", []) if r.get("crop_path")],
        "appearance": {
            "date": date.today().isoformat(),
            **clothing_structured,
        },
        "body_crops": [a["body_path"] for a in associations],
        "best_body_crops": best_body_crops,
        "video_sources": state["video_paths"],
        "appearance_signals": {
            "color": color_signals,
        },
        "reid": reid_signal,
    }


def build_profile(state: dict) -> dict:
    """Assemble one profile dict per identity cluster and interrupt for review.

    Input state:  `identity_clusters`, `associations`,
                  `per_cluster_best_body_crops`, `per_cluster_clothing`.
    Output state: `per_cluster_profiles`, single-cluster `profile`,
                  `review_feedback`, `approved`.
    """
    clusters = state.get("identity_clusters") or []
    if not clusters:
        print("[build_profile] no identity clusters - no profile review needed")
        return {"per_cluster_profiles": {}, "profile": {}, "approved": True}

    profiles: dict[int, dict] = {}
    for cluster in clusters:
        profile = _profile_for_cluster(state, cluster)
        profiles[int(cluster["cluster_id"])] = profile

    preview = {
        "profiles_count": len(profiles),
        "profiles": [
            {
                "cluster_id": p["cluster_id"],
                "name": p["name"],
                "face_crop_count": p["face_crop_count"],
                "associations_count": len(p["body_crops"]),
                "cluster_confidence": p["cluster_confidence"],
                "appearance": p["appearance"],
                "color_signals": p["appearance_signals"]["color"],
                "reid": p["reid"],
                "best_body_crops": p["best_body_crops"],
            }
            for p in profiles.values()
        ],
    }

    feedback = interrupt({
        "message": "Review the generated person profiles below. Reply with 'approve' or provide corrections.",
        "profile_preview": preview,
    })

    override = (feedback or {}).get("clothing_override") or {}
    if override:
        for profile in profiles.values():
            profile["appearance"].update({k: v for k, v in override.items() if v})

    first_id = sorted(profiles)[0]
    print(f"[build_profile] built {len(profiles)} profile(s)")
    return {
        "per_cluster_profiles": profiles,
        "profile": profiles[first_id],
        "review_feedback": feedback,
        "approved": True,
    }
