from datetime import date
from pathlib import Path

from forensics.person_creation.nodes.profile_signals import (
    build_association_meta,
    color_signals_from_crops,
)


def _session_id(state: dict) -> str:
    return Path(state["output_dir"]).name or state.get("person_name", "session")


def _reid_signal_for_cluster(state: dict, cid: int) -> dict:
    embeddings = state.get("reid_embeddings") or {}
    reasons = state.get("reid_reasons") or {}
    crop_counts = state.get("reid_crop_counts") or {}
    config = state.get("reid_config") or {}

    body_embedding = embeddings.get(cid, embeddings.get(str(cid)))
    reason = reasons.get(cid, reasons.get(str(cid), "no_reid_model_configured"))
    crop_count = int(crop_counts.get(cid, crop_counts.get(str(cid), 0)) or 0)

    if body_embedding:
        return {
            "status": "computed",
            "model": config.get("model", "osnet_x0_25"),
            "weights": config.get("weights", "market1501"),
            "embedding_dim": int(config.get("embedding_dim", len(body_embedding))),
            "body_embedding": body_embedding,
            "aggregation": "mean_of_best_5_body_crops",
            "crop_count": crop_count,
        }

    return {
        "status": "not_computed",
        "reason": reason,
        "body_embedding": None,
        "note": "Reserved for body ReID embedding (OSNet or equivalent). Permanent identity is in face_embedding.",
    }


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
        "best_body_crops": best_body_crops,
    }
    association_meta = build_association_meta(cluster_state)
    reid_signal = _reid_signal_for_cluster(state, cid)
    sid = _session_id(state)
    profile_id = f"person_{sid}_cluster_{cid}"
    display_name = state.get("person_name") or state.get("name") or "Unknown"
    face_crop_sharpness = {
        r["crop_path"]: float(r.get("sharpness", 0.0) or 0.0)
        for r in cluster.get("face_records", [])
        if r.get("crop_path")
    }
    body_crop_sharpness = {
        a["body_path"]: float(a.get("body_sharpness", 0.0) or 0.0)
        for a in associations
        if a.get("body_path")
    }
    video_sources = list(state.get("video_paths") or [])
    if state.get("source_type") == "live_camera" and state.get("source_uri_masked"):
        video_sources = [state["source_uri_masked"]]

    return {
        "id": profile_id,
        "name": display_name,
        "cluster_id": cid,
        "cluster_confidence": cluster.get("confidence", 0.0),
        "cluster_face_count": cluster.get("face_count", 0),
        "low_confidence": bool(cluster.get("low_confidence", False)),
        "face_count": cluster.get("face_count", 0),
        "created_at": date.today().isoformat(),
        "face_embedding": cluster.get("representative_embedding", []),
        "face_embedding_meta": {
            "model": "facenet_pytorch.InceptionResnetV1.vggface2",
            "dim": 512,
            "norm": "L2",
        },
        "face_crop_count": cluster.get("face_count", 0),
        "face_crops": [r["crop_path"] for r in cluster.get("face_records", []) if r.get("crop_path")],
        "face_crop_sharpness": face_crop_sharpness,
        "appearance": {
            "date": date.today().isoformat(),
            **clothing_structured,
        },
        "body_crops": [a["body_path"] for a in associations],
        "best_body_crops": best_body_crops,
        "body_crop_sharpness": body_crop_sharpness,
        "video_sources": video_sources,
        "source_type": state.get("source_type", "video_file"),
        "camera_id": state.get("camera_id"),
        "appearance_signals": {
            "color": color_signals,
        },
        "association_meta": association_meta,
        "reid": reid_signal,
    }


def build_profile(state: dict) -> dict:
    """Assemble one profile dict per identity cluster.

    Fully automatic — trusts the VLM clothing output and the automated
    face/body assignment with no human review step.

    Input state:  `identity_clusters`, `associations`,
                  `per_cluster_best_body_crops`, `per_cluster_clothing`.
    Output state: `per_cluster_profiles`, single-cluster `profile`.
    """
    clusters = state.get("identity_clusters") or []
    if not clusters:
        print("[build_profile] no identity clusters - no profile built")
        return {"per_cluster_profiles": {}, "profile": {}}

    profiles: dict[int, dict] = {}
    for cluster in clusters:
        profile = _profile_for_cluster(state, cluster)
        profiles[int(cluster["cluster_id"])] = profile

    first_id = sorted(profiles)[0]
    print(f"[build_profile] built {len(profiles)} profile(s)")
    return {
        "per_cluster_profiles": profiles,
        "profile": profiles[first_id],
    }
