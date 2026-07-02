import json
from datetime import datetime
from pathlib import Path

from scipy.optimize import linear_sum_assignment


_MATCHING_METHOD = "geometry_hungarian_v1"
_ACCEPT_COST_THRESHOLD = 0.50


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_parts(crop: dict) -> tuple[float, float, float, float, float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in crop["bbox"]]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    return x1, y1, x2, y2, w, h, cx, cy


def _distance_outside_rect(
    x: float,
    y: float,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> float:
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return (dx * dx + dy * dy) ** 0.5


def _geometry_cost(face: dict, body: dict) -> float:
    _fx1, _fy1, _fx2, _fy2, _face_w, face_h, face_cx, face_cy = _bbox_parts(face)
    body_x1, body_y1, body_x2, _body_y2, body_w, body_h, body_cx, _body_cy = _bbox_parts(body)

    if body_w <= 0 or body_h <= 0 or face_h <= 0:
        return 1.0

    upper_top = body_y1
    upper_bottom = body_y1 + (0.35 * body_h)
    contain_norm = max(body_w, upper_bottom - upper_top, 1.0)
    c_contain = _clip01(
        _distance_outside_rect(
            face_cx,
            face_cy,
            body_x1,
            upper_top,
            body_x2,
            upper_bottom,
        )
        / contain_norm
    )
    c_align = _clip01(abs(face_cx - body_cx) / max(0.5 * body_w, 1.0))
    c_size = _clip01(abs((face_h / body_h) - 0.17) / 0.17)
    c_vertical = _clip01(abs(((face_cy - body_y1) / body_h) - 0.15))

    return (
        (0.40 * c_contain)
        + (0.25 * c_align)
        + (0.20 * c_size)
        + (0.15 * c_vertical)
    )


def _group_by_frame(quality_face: list[dict], quality_body: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], dict] = {}

    for face in quality_face:
        key = (face["video"], face["frame_idx"])
        groups.setdefault(
            key,
            {
                "frame_idx": face["frame_idx"],
                "video": face["video"],
                "video_name": Path(face["video"]).name,
                "faces": [],
                "bodies": [],
            },
        )
        groups[key]["faces"].append(face)

    for body in quality_body:
        key = (body["video"], body["frame_idx"])
        groups.setdefault(
            key,
            {
                "frame_idx": body["frame_idx"],
                "video": body["video"],
                "video_name": Path(body["video"]).name,
                "faces": [],
                "bodies": [],
            },
        )
        groups[key]["bodies"].append(body)

    frame_groups = [
        group
        for group in groups.values()
        if len(group["faces"]) > 0 and len(group["bodies"]) > 0
    ]
    frame_groups.sort(key=lambda group: (group["video"], group["frame_idx"]))
    return frame_groups


def _association(face: dict, body: dict, cost: float) -> dict:
    fx1, fy1, fx2, fy2, face_w, face_h, _face_cx, _face_cy = _bbox_parts(face)
    bx1, by1, bx2, by2, body_w, body_h, _body_cx, _body_cy = _bbox_parts(body)
    body_area = body_w * body_h

    return {
        "face_path": face["path"],
        "body_path": body["path"],
        "frame_idx": face["frame_idx"],
        "video": face["video"],
        "video_name": Path(face["video"]).name,
        "body_sharpness": body["sharpness"],
        "iou_score": None,
        "face_bbox": [fx1, fy1, fx2, fy2],
        "body_bbox": [bx1, by1, bx2, by2],
        "face_w": int(face_w),
        "face_h": int(face_h),
        "body_w": int(body_w),
        "body_h": int(body_h),
        "body_area": int(body_area),
        "face_height_ratio": round(face_h / body_h, 3) if body_h > 0 else 0,
        "confirmed_by_human": False,
        "matching_method": _MATCHING_METHOD,
        "matching_cost": round(cost, 4),
    }


def _delete_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            Path(path).resolve().unlink(missing_ok=True)
        except OSError:
            pass


def auto_associate(state: dict) -> dict:
    quality_face = list(state["quality_face_crops"])
    quality_body = list(state["quality_body_crops"])
    output_dir = Path(state["output_dir"])

    frame_groups = _group_by_frame(quality_face, quality_body)
    associations: list[dict] = []
    accepted_face_paths: set[str] = set()
    accepted_body_paths: set[str] = set()
    accepted_costs: list[float] = []

    for group in frame_groups:
        faces = group["faces"]
        bodies = group["bodies"]
        cost_matrix = [
            [_geometry_cost(face, body) for body in bodies]
            for face in faces
        ]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for face_idx, body_idx in zip(row_ind, col_ind):
            cost = float(cost_matrix[face_idx][body_idx])
            if cost > _ACCEPT_COST_THRESHOLD:
                continue

            face = faces[face_idx]
            body = bodies[body_idx]
            associations.append(_association(face, body, cost))
            accepted_face_paths.add(face["path"])
            accepted_body_paths.add(body["path"])
            accepted_costs.append(cost)

    deleted_paths = [
        crop["path"]
        for crop in [*quality_face, *quality_body]
        if crop["path"] not in accepted_face_paths
        and crop["path"] not in accepted_body_paths
    ]
    _delete_paths(deleted_paths)

    cleaned_face = [
        crop for crop in quality_face
        if crop["path"] in accepted_face_paths
    ]
    cleaned_body = [
        crop for crop in quality_body
        if crop["path"] in accepted_body_paths
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = output_dir / "pairing_feedback.json"
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "person_name": state.get("person_name", ""),
        "video_sources": state.get("video_paths", []),
        "total_frame_groups_shown": len(frame_groups),
        "confirmed_pairs_count": len(associations),
        "deleted_paths_count": len(deleted_paths),
        "confirmed_by_human": False,
        "associations": associations,
        "confirmed_pairs": associations,
        "deleted_paths": deleted_paths,
        "matching_method": _MATCHING_METHOD,
    }
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2)

    avg_cost = sum(accepted_costs) / len(accepted_costs) if accepted_costs else 0.0
    print(f"[auto_associate] frame_groups={len(frame_groups)}")
    print(f"[auto_associate] associations={len(associations)}")
    print(f"[auto_associate] rejected_crops={len(deleted_paths)}")
    print(f"[auto_associate] avg_accepted_cost={avg_cost:.4f}")

    return {
        "associations": associations,
        "frame_groups": frame_groups,
        "human_feedback_path": str(feedback_path.resolve()),
        "quality_body_crops": cleaned_body,
        "quality_face_crops": cleaned_face,
    }
