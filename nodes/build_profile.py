from datetime import date
from langgraph.types import interrupt


def build_profile(state: dict) -> dict:
    profile = {
        "id": state["person_name"].lower(),
        "name": state["person_name"],
        "created_at": date.today().isoformat(),
        "face_embedding": state["mean_face_embedding"],
        "face_crop_count": len(state["quality_face_crops"]),
        "face_crops": [c["path"] for c in state["quality_face_crops"]],
        "appearance": {
            "date": date.today().isoformat(),
            **state["clothing_structured"],
        },
        "body_crops": [a["body_path"] for a in state["associations"]],
        "best_body_crops": state["best_body_crops"],
        "video_sources": state["video_paths"],
    }

    feedback = interrupt({
        "message": "Review the person profile below. Reply with 'approve' or provide corrections.",
        "profile_preview": {
            "name": profile["name"],
            "face_crop_count": profile["face_crop_count"],
            "associations_count": len(state["associations"]),
            "appearance": profile["appearance"],
            "best_body_crops": profile["best_body_crops"],
        },
    })

    # Apply user clothing corrections if they edited the fields in the UI
    override = (feedback or {}).get("clothing_override") or {}
    if override:
        profile["appearance"].update({k: v for k, v in override.items() if v})

    return {"profile": profile, "review_feedback": feedback, "approved": True}
