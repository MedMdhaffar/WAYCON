from datetime import date
from langgraph.types import interrupt


def _profile_reid_block(reid: dict | None) -> dict:
    reid = reid or {}
    # ReID is a supporting same-day appearance signal. Face embedding remains
    # the permanent identity key.
    block = {
        "model": reid.get("model", "OSNet_x1_0"),
        "embedding_dim": reid.get("embedding_dim"),
        "embedding": reid.get("embedding"),
        "source_crops": reid.get("source_crops", []),
        "per_crop_count": len(reid.get("per_crop") or []),
        "signal_type": reid.get("signal_type", "same_day_supporting_appearance"),
    }
    if reid.get("error"):
        block["error"] = reid["error"]
    return block


def _profile_color_block(color_signals: dict | None) -> dict:
    color_signals = color_signals or {}
    # Color signals are daily supporting appearance signals. Face embedding
    # remains the permanent identity key.
    block = {
        "extractor": color_signals.get("extractor", "DominantColorExtractor_v1"),
        "signal_type": color_signals.get("signal_type", "same_day_supporting_appearance"),
        "source_crops": color_signals.get("source_crops", []),
        "per_crop_count": color_signals.get("per_crop_count", 0),
        "top": color_signals.get("top"),
        "bottom": color_signals.get("bottom"),
        "shoes": color_signals.get("shoes"),
    }
    if color_signals.get("error"):
        block["error"] = color_signals["error"]
    return block


def _appearance_colors(color_signals: dict) -> dict:
    return {
        region: color_signals[region]["dominant"]
        for region in ("top", "bottom", "shoes")
        if color_signals.get(region)
    }


def build_profile(state: dict) -> dict:
    color_signals = _profile_color_block(state.get("color_signals"))
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
            "colors": _appearance_colors(color_signals),
        },
        "reid": _profile_reid_block(state.get("reid")),
        "color_signals": color_signals,
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
