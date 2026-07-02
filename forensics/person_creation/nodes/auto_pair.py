from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path


_MIN_MATCH_SCORE = 0.52
_MAX_TRACK_GAP = 45
_MAX_TRACK_DISTANCE = 0.85


def _bbox_size(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _face_body_score(face: dict, body: dict) -> tuple[float, dict]:
    fx1, fy1, fx2, fy2 = face["bbox"]
    bx1, by1, bx2, by2 = body["bbox"]
    face_w, face_h = _bbox_size(face["bbox"])
    body_w, body_h = _bbox_size(body["bbox"])
    if face_w <= 0 or face_h <= 0 or body_w <= 0 or body_h <= 0:
        return 0.0, {"reject": "empty_bbox"}

    fcx, fcy = _center(face["bbox"])
    bcx = (bx1 + bx2) / 2.0
    upper_y = by1 + 0.20 * body_h

    inside = bx1 <= fcx <= bx2 and by1 <= fcy <= by2
    face_y_rel = (fcy - by1) / body_h
    upper_region = -0.05 <= face_y_rel <= 0.48
    ratio = face_h / body_h

    if not inside:
        return 0.0, {"reject": "face_center_outside_body", "face_height_ratio": ratio}
    if ratio < 0.055 or ratio > 0.55:
        return 0.0, {"reject": "face_body_ratio_out_of_range", "face_height_ratio": ratio}

    dx = abs(fcx - bcx) / max(body_w * 0.5, 1.0)
    dy = abs(fcy - upper_y) / max(body_h * 0.35, 1.0)
    distance_score = _clamp01(1.0 - math.sqrt(dx * dx + dy * dy) / 1.6)

    ratio_score = _clamp01(1.0 - abs(ratio - 0.24) / 0.26)
    upper_score = _clamp01(1.0 - abs(face_y_rel - 0.22) / 0.32) if upper_region else 0.0
    conf_score = _clamp01((float(face.get("score", 0.5)) + float(body.get("score", 0.5))) / 2.0)
    sharp_score = _clamp01(float(body.get("sharpness", 0.0)) / 1200.0)

    score = (
        0.24 * float(inside)
        + 0.24 * upper_score
        + 0.22 * ratio_score
        + 0.22 * distance_score
        + 0.05 * conf_score
        + 0.03 * sharp_score
    )
    parts = {
        "face_center_inside_body": inside,
        "face_y_rel": round(face_y_rel, 4),
        "upper_region": upper_region,
        "face_height_ratio": round(ratio, 4),
        "upper_center_distance": round(math.sqrt(dx * dx + dy * dy), 4),
        "distance_score": round(distance_score, 4),
        "ratio_score": round(ratio_score, 4),
        "upper_score": round(upper_score, 4),
        "confidence_score": round(conf_score, 4),
        "sharpness_score": round(sharp_score, 4),
    }
    return score, parts


def _assignment(costs: list[list[float]]) -> list[tuple[int, int]]:
    if not costs or not costs[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(costs)
        return list(zip([int(r) for r in rows], [int(c) for c in cols]))
    except Exception:
        pass

    row_count = len(costs)
    col_count = len(costs[0])
    transposed = False
    matrix = costs
    if row_count > col_count:
        transposed = True
        matrix = [[costs[r][c] for r in range(row_count)] for c in range(col_count)]
        row_count, col_count = col_count, row_count

    memo: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}

    def solve(row: int, used_cols: int) -> tuple[float, list[tuple[int, int]]]:
        if row >= row_count:
            return 0.0, []
        key = (row, used_cols)
        if key in memo:
            return memo[key]
        best_cost = float("inf")
        best_pairs: list[tuple[int, int]] = []
        for col in range(col_count):
            if used_cols & (1 << col):
                continue
            tail_cost, tail_pairs = solve(row + 1, used_cols | (1 << col))
            total = matrix[row][col] + tail_cost
            if total < best_cost:
                best_cost = total
                best_pairs = [(row, col)] + tail_pairs
        memo[key] = (best_cost, best_pairs)
        return memo[key]

    _cost, pairs = solve(0, 0)
    if transposed:
        return [(col, row) for row, col in pairs]
    return pairs


def _association(face: dict, body: dict, score: float, components: dict) -> dict:
    fx1, fy1, fx2, fy2 = face["bbox"]
    bx1, by1, bx2, by2 = body["bbox"]
    face_h = fy2 - fy1
    body_h = by2 - by1
    body_area = (bx2 - bx1) * body_h
    return {
        "face_path": face["path"],
        "body_path": body["path"],
        "frame_idx": face["frame_idx"],
        "video": face["video"],
        "video_name": Path(face["video"]).name,
        "body_sharpness": body["sharpness"],
        "iou_score": None,
        "auto_score": round(float(score), 4),
        "auto_score_components": components,
        "face_bbox": [fx1, fy1, fx2, fy2],
        "body_bbox": [bx1, by1, bx2, by2],
        "face_w": int(fx2 - fx1),
        "face_h": int(face_h),
        "body_w": int(bx2 - bx1),
        "body_h": int(body_h),
        "body_area": int(body_area),
        "face_height_ratio": round(face_h / body_h, 3) if body_h > 0 else 0,
        "confirmed_by_human": False,
        "confirmed_automatically": True,
    }


def _build_frame_groups(quality_face: list[dict], quality_body: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for item, key_name in ((f, "faces") for f in quality_face):
        key = (item["frame_idx"], item["video"])
        groups.setdefault(key, {
            "frame_idx": item["frame_idx"],
            "video": item["video"],
            "video_name": Path(item["video"]).name,
            "faces": [],
            "bodies": [],
        })[key_name].append(item)
    for item, key_name in ((b, "bodies") for b in quality_body):
        key = (item["frame_idx"], item["video"])
        groups.setdefault(key, {
            "frame_idx": item["frame_idx"],
            "video": item["video"],
            "video_name": Path(item["video"]).name,
            "faces": [],
            "bodies": [],
        })[key_name].append(item)

    frame_groups = [g for g in groups.values() if g["faces"] and g["bodies"]]
    frame_groups.sort(key=lambda x: (x["video"], x["frame_idx"]))
    return frame_groups


def _match_group(group: dict) -> tuple[list[dict], list[dict]]:
    faces = group["faces"]
    bodies = group["bodies"]
    scores: list[list[tuple[float, dict]]] = []
    for face in faces:
        row = []
        for body in bodies:
            row.append(_face_body_score(face, body))
        scores.append(row)

    costs = [[-score for score, _parts in row] for row in scores]
    accepted: list[dict] = []
    rejected: list[dict] = []
    for face_idx, body_idx in _assignment(costs):
        score, parts = scores[face_idx][body_idx]
        face = faces[face_idx]
        body = bodies[body_idx]
        if score >= _MIN_MATCH_SCORE:
            accepted.append(_association(face, body, score, parts))
        else:
            rejected.append({
                "face_path": face["path"],
                "body_path": body["path"],
                "frame_idx": group["frame_idx"],
                "video": group["video"],
                "auto_score": round(float(score), 4),
                "reason": "weak_match",
                "components": parts,
            })
    return accepted, rejected


def _track_distance(a: dict, b: dict) -> float:
    acx, acy = _center(a["body_bbox"])
    bcx, bcy = _center(b["body_bbox"])
    ah = max(float(a.get("body_h", 1)), 1.0)
    bh = max(float(b.get("body_h", 1)), 1.0)
    scale = max((ah + bh) / 2.0, 1.0)
    center_dist = math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / scale
    ratio_dist = abs(math.log(max(ah, 1.0) / max(bh, 1.0)))
    return center_dist + 0.35 * ratio_dist


def _select_primary_track(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    if not candidates:
        return [], []
    tracks: list[list[dict]] = []
    for assoc in sorted(candidates, key=lambda a: (a["video"], a["frame_idx"], -a["auto_score"])):
        best_track = None
        best_dist = None
        for track in tracks:
            last = track[-1]
            if last["video"] != assoc["video"]:
                continue
            if assoc["frame_idx"] <= last["frame_idx"]:
                continue
            if assoc["frame_idx"] - last["frame_idx"] > _MAX_TRACK_GAP:
                continue
            dist = _track_distance(last, assoc)
            if dist <= _MAX_TRACK_DISTANCE and (best_dist is None or dist < best_dist):
                best_track = track
                best_dist = dist
        if best_track is None:
            tracks.append([assoc])
        else:
            best_track.append(assoc)

    def track_score(track: list[dict]) -> tuple[float, float, float]:
        count = len(track)
        avg_score = sum(a["auto_score"] for a in track) / count
        avg_area = sum(a["body_area"] for a in track) / count
        return count * 10.0 + avg_score, avg_score, avg_area

    primary = max(tracks, key=track_score)
    primary_paths = {a["body_path"] for a in primary} | {a["face_path"] for a in primary}
    rejected = [
        {**a, "reject_reason": "non_primary_track"}
        for track in tracks
        if track is not primary
        for a in track
        if a["body_path"] not in primary_paths and a["face_path"] not in primary_paths
    ]
    return primary, rejected


def _load_reference_feedback(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _validate_against_reference(associations: list[dict], reference: dict | None) -> dict:
    if not reference:
        return {"available": False}
    ref_pairs = {
        (_basename(p.get("face_path", "")), _basename(p.get("body_path", "")))
        for p in reference.get("confirmed_pairs", []) or []
    }
    auto_pairs = {
        (_basename(p.get("face_path", "")), _basename(p.get("body_path", "")))
        for p in associations
    }
    if not ref_pairs:
        return {"available": False, "reason": "reference_has_no_pairs"}
    tp = len(auto_pairs & ref_pairs)
    fp = len(auto_pairs - ref_pairs)
    fn = len(ref_pairs - auto_pairs)
    return {
        "available": True,
        "reference_pairs": len(ref_pairs),
        "auto_pairs": len(auto_pairs),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(tp / max(len(auto_pairs), 1), 4),
        "recall": round(tp / max(len(ref_pairs), 1), 4),
    }


def auto_pair(state: dict) -> dict:
    quality_face = state["quality_face_crops"]
    quality_body = state["quality_body_crops"]
    output_dir = Path(state["output_dir"])
    feedback_path = output_dir / "pairing_feedback.json"
    reference_feedback = _load_reference_feedback(feedback_path)

    frame_groups = _build_frame_groups(quality_face, quality_body)
    all_candidates: list[dict] = []
    weak_rejections: list[dict] = []
    for group in frame_groups:
        accepted, rejected = _match_group(group)
        all_candidates.extend(accepted)
        weak_rejections.extend(rejected)

    associations, non_primary_rejections = _select_primary_track(all_candidates)

    keep_face_paths = {a["face_path"] for a in associations}
    keep_body_paths = {a["body_path"] for a in associations}
    filtered_face = [c for c in quality_face if c["path"] in keep_face_paths]
    filtered_body = [c for c in quality_body if c["path"] in keep_body_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _validate_against_reference(associations, reference_feedback)
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "pairing_mode": "automatic_geometry_v1",
        "person_name": state.get("person_name", ""),
        "video_sources": state.get("video_paths", []),
        "total_frame_groups_shown": len(frame_groups),
        "candidate_pairs_count": len(all_candidates),
        "confirmed_pairs_count": len(associations),
        "rejected_pairs_count": len(weak_rejections) + len(non_primary_rejections),
        "deleted_paths_count": 0,
        "confirmed_pairs": associations,
        "rejected_pairs": weak_rejections + non_primary_rejections,
        "deleted_paths": [],
        "validation_against_previous_feedback": validation,
    }
    feedback_path.write_text(json.dumps(feedback_data, indent=2), encoding="utf-8")

    print(
        f"[auto_pair] {len(frame_groups)} frame groups -> {len(all_candidates)} confident "
        f"pairs -> {len(associations)} primary-track pairs"
    )
    if validation.get("available"):
        print(
            f"[auto_pair] validation precision={validation['precision']} "
            f"recall={validation['recall']} vs existing feedback"
        )
    print(f"[auto_pair] feedback saved to {feedback_path}")

    return {
        "associations": associations,
        "frame_groups": frame_groups,
        "human_feedback_path": str(feedback_path.resolve()),
        "quality_body_crops": filtered_body,
        "quality_face_crops": filtered_face,
    }
