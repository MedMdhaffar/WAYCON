from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from forensics.person_creation.nodes.auto_pair import (
    _MIN_MATCH_SCORE,
    _association,
    _assignment,
    _conf_sharp_score,
    _geometry_score,
)


def _build_cluster_lookup(identity_clusters: list[dict]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for cluster in identity_clusters:
        cid = int(cluster["cluster_id"])
        for record in cluster.get("face_records", []):
            path = record.get("crop_path")
            if path:
                lookup[path] = cid
    return lookup


def _build_frame_groups(quality_face: list[dict], quality_body: list[dict], cluster_lookup: dict[str, int]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for face in quality_face:
        cid = cluster_lookup.get(face.get("path"))
        if cid is None:
            continue
        key = (face["frame_idx"], face["video"])
        item = dict(face)
        item["cluster_id"] = cid
        groups.setdefault(key, {
            "frame_idx": face["frame_idx"],
            "video": face["video"],
            "video_name": Path(face["video"]).name,
            "faces": [],
            "bodies": [],
        })["faces"].append(item)

    for body in quality_body:
        key = (body["frame_idx"], body["video"])
        groups.setdefault(key, {
            "frame_idx": body["frame_idx"],
            "video": body["video"],
            "video_name": Path(body["video"]).name,
            "faces": [],
            "bodies": [],
        })["bodies"].append(body)

    frame_groups = [g for g in groups.values() if g["faces"] and g["bodies"]]
    frame_groups.sort(key=lambda x: (x["video"], x["frame_idx"]))
    return frame_groups


def _score_pair(face: dict, body: dict) -> tuple[bool, float, dict]:
    hard_ok, geo_score, geo_parts = _geometry_score(face, body)
    if not hard_ok:
        return False, 0.0, {"reject": geo_parts.get("reject", "geometry_infeasible"), "geometry": geo_parts}
    conf_sharp = _conf_sharp_score(face, body)
    score = 0.82 * geo_score + 0.18 * conf_sharp
    return True, score, {
        "geometry": geo_parts,
        "raw": {"geometry": round(geo_score, 4), "conf_sharp": round(conf_sharp, 4)},
        "weights_used": {"geometry": 0.82, "conf_sharp": 0.18},
    }


def _match_group(group: dict) -> tuple[list[dict], list[dict]]:
    faces = group["faces"]
    bodies = group["bodies"]
    pair_info: list[list[tuple]] = []
    costs: list[list[float]] = []

    for face in faces:
        info_row = []
        cost_row = []
        for body in bodies:
            hard_ok, score, parts = _score_pair(face, body)
            info_row.append((hard_ok, score, parts))
            cost_row.append(-score if hard_ok else 1e6)
        pair_info.append(info_row)
        costs.append(cost_row)

    accepted: list[dict] = []
    rejected: list[dict] = []
    for face_idx, body_idx in _assignment(costs):
        hard_ok, score, parts = pair_info[face_idx][body_idx]
        face = faces[face_idx]
        body = bodies[body_idx]
        if hard_ok and score >= _MIN_MATCH_SCORE:
            assoc = _association(face, body, score, parts)
            assoc["cluster_id"] = int(face["cluster_id"])
            assoc["assignment_score"] = assoc["auto_score"]
            assoc["face_crop_path"] = assoc["face_path"]
            assoc["body_crop_path"] = assoc["body_path"]
            accepted.append(assoc)
        else:
            rejected.append({
                "face_path": face.get("path"),
                "body_path": body.get("path"),
                "frame_idx": group["frame_idx"],
                "video": group["video"],
                "cluster_id": face.get("cluster_id"),
                "auto_score": round(float(score), 4),
                "reason": "weak_match" if hard_ok else parts.get("reject", "geometry_infeasible"),
            })

    return accepted, rejected


def _dedupe_cluster_frame(assignments: list[dict]) -> tuple[list[dict], list[dict]]:
    winners: dict[tuple[int, str, int], dict] = {}
    rejected: list[dict] = []
    for assoc in assignments:
        key = (int(assoc["cluster_id"]), assoc["video"], int(assoc["frame_idx"]))
        prev = winners.get(key)
        if prev is None or assoc["assignment_score"] > prev["assignment_score"]:
            if prev is not None:
                rejected.append({**prev, "reject_reason": "lower_score_duplicate_cluster_frame"})
            winners[key] = assoc
        else:
            rejected.append({**assoc, "reject_reason": "lower_score_duplicate_cluster_frame"})
    return list(winners.values()), rejected


def assign_bodies_to_clusters(state: dict) -> dict:
    """Pair each clustered face to a body crop in the same frame (Hungarian match).

    Input state:  `identity_clusters`, `quality_face_crops`, `quality_body_crops`.
    Output state: `cluster_assignments` (per-cluster face/body pairs), flat
                  `associations` (compatibility), `unattached_bodies`,
                  `frame_groups`, `human_feedback_path`.
    """
    clusters = state.get("identity_clusters", [])
    quality_face = state.get("quality_face_crops", [])
    quality_body = state.get("quality_body_crops", [])
    output_dir = Path(state["output_dir"])
    feedback_path = output_dir / "pairing_feedback.json"

    cluster_lookup = _build_cluster_lookup(clusters)
    frame_groups = _build_frame_groups(quality_face, quality_body, cluster_lookup)

    accepted_all: list[dict] = []
    rejected_pairs: list[dict] = []
    for group in frame_groups:
        accepted, rejected = _match_group(group)
        accepted_all.extend(accepted)
        rejected_pairs.extend(rejected)

    accepted_all, duplicate_rejections = _dedupe_cluster_frame(accepted_all)
    rejected_pairs.extend(duplicate_rejections)

    cluster_assignments: dict[int, list[dict]] = {int(c["cluster_id"]): [] for c in clusters}
    for assoc in sorted(accepted_all, key=lambda a: (a["cluster_id"], a["video"], a["frame_idx"])):
        cluster_assignments.setdefault(int(assoc["cluster_id"]), []).append({
            "frame_idx": assoc["frame_idx"],
            "video": assoc["video"],
            "face_crop_path": assoc["face_path"],
            "body_crop_path": assoc["body_path"],
            "assignment_score": assoc["assignment_score"],
            "body_sharpness": assoc.get("body_sharpness", 0.0),
        })

    attached_body_paths = {a["body_path"] for a in accepted_all}
    unattached_bodies = [b for b in quality_body if b.get("path") not in attached_body_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "pairing_mode": "face_cluster_assignment_v1",
        "person_name": state.get("person_name", ""),
        "video_sources": state.get("video_paths", []),
        "identity_clusters_found": len(clusters),
        "total_frame_groups_shown": len(frame_groups),
        "candidate_pairs_count": len(accepted_all) + len(rejected_pairs),
        "confirmed_pairs_count": len(accepted_all),
        "rejected_pairs_count": len(rejected_pairs),
        "unattached_bodies_count": len(unattached_bodies),
        "confirmed_pairs": accepted_all,
        "rejected_pairs": rejected_pairs,
    }
    feedback_path.write_text(json.dumps(feedback_data, indent=2), encoding="utf-8")

    print(
        f"[assign_bodies_to_clusters] clusters={len(clusters)} frame_groups={len(frame_groups)} "
        f"assignments={len(accepted_all)} unattached_bodies={len(unattached_bodies)}"
    )
    return {
        "associations": accepted_all,
        "cluster_assignments": cluster_assignments,
        "unattached_bodies": unattached_bodies,
        "frame_groups": frame_groups,
        "human_feedback_path": str(feedback_path.resolve()),
    }
