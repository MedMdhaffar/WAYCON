import json
from datetime import datetime
from pathlib import Path

from langgraph.types import interrupt


def human_in_the_loop(state: dict) -> dict:
    quality_face = state["quality_face_crops"]
    quality_body = state["quality_body_crops"]
    output_dir = Path(state["output_dir"])

    # --- Group by (frame_idx, video) ---
    groups: dict[tuple, dict] = {}
    for face in quality_face:
        key = (face["frame_idx"], face["video"])
        if key not in groups:
            groups[key] = {
                "frame_idx": face["frame_idx"],
                "video": face["video"],
                "video_name": Path(face["video"]).name,
                "faces": [],
                "bodies": [],
            }
        groups[key]["faces"].append(face)

    for body in quality_body:
        key = (body["frame_idx"], body["video"])
        if key not in groups:
            groups[key] = {
                "frame_idx": body["frame_idx"],
                "video": body["video"],
                "video_name": Path(body["video"]).name,
                "faces": [],
                "bodies": [],
            }
        groups[key]["bodies"].append(body)

    # Keep only frames with ≥1 face AND ≥1 body
    frame_groups = [
        g for g in groups.values()
        if len(g["faces"]) > 0 and len(g["bodies"]) > 0
    ]
    frame_groups.sort(key=lambda x: (x["video"], x["frame_idx"]))

    print(f"[human_in_the_loop] {len(frame_groups)} frame groups with face+body — waiting for pairing")

    # --- Interrupt: hand off to UI ---
    feedback = interrupt({
        "message": "Pair faces to bodies. Delete noise. Click Confirm when done.",
        "frame_groups": frame_groups,
        "total_frames": len(frame_groups),
    })

    # feedback = {"human_pairs": [{face_path, body_path}, ...], "deleted_paths": [...]}
    human_pairs = feedback.get("human_pairs", [])
    deleted_paths = feedback.get("deleted_paths", [])

    # --- Build face/body metadata lookup ---
    face_by_path = {f["path"]: f for f in quality_face}
    body_by_path = {b["path"]: b for b in quality_body}

    # --- Construct associations ---
    associations = []
    for pair in human_pairs:
        fp = pair.get("face_path", "")
        bp = pair.get("body_path", "")
        face_meta = face_by_path.get(fp)
        body_meta = body_by_path.get(bp)
        if not face_meta or not body_meta:
            continue

        fx1, fy1, fx2, fy2 = face_meta["bbox"]
        bx1, by1, bx2, by2 = body_meta["bbox"]
        face_h = fy2 - fy1
        body_h = by2 - by1
        body_area = (bx2 - bx1) * body_h

        associations.append({
            "face_path":          fp,
            "body_path":          bp,
            "frame_idx":          face_meta["frame_idx"],
            "video":              face_meta["video"],
            "video_name":         Path(face_meta["video"]).name,
            "body_sharpness":     body_meta["sharpness"],
            "iou_score":          None,
            "face_bbox":          [fx1, fy1, fx2, fy2],
            "body_bbox":          [bx1, by1, bx2, by2],
            "face_w":             int(fx2 - fx1),
            "face_h":             int(face_h),
            "body_w":             int(bx2 - bx1),
            "body_h":             int(body_h),
            "body_area":          int(body_area),
            "face_height_ratio":  round(face_h / body_h, 3) if body_h > 0 else 0,
            "confirmed_by_human": True,
        })

    # --- Save feedback JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = output_dir / "pairing_feedback.json"
    feedback_data = {
        "timestamp":                datetime.now().isoformat(),
        "person_name":              state.get("person_name", ""),
        "video_sources":            state.get("video_paths", []),
        "total_frame_groups_shown": len(frame_groups),
        "confirmed_pairs_count":    len(associations),
        "deleted_paths_count":      len(deleted_paths),
        "confirmed_pairs":          associations,
        "deleted_paths":            deleted_paths,
    }
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2)

    # Propagate deletions into LangGraph state so downstream nodes see correct paths
    deleted_set = set(deleted_paths)
    filtered_body = [c for c in quality_body if c["path"] not in deleted_set]
    filtered_face = [c for c in quality_face if c["path"] not in deleted_set]

    print(f"[human_in_the_loop] {len(associations)} pairs confirmed → feedback saved to {feedback_path}")

    return {
        "associations":        associations,
        "frame_groups":        frame_groups,
        "human_feedback_path": str(feedback_path.resolve()),
        "quality_body_crops":  filtered_body,
        "quality_face_crops":  filtered_face,
    }
