from datetime import date

from langgraph.types import interrupt


_FACE_MODEL_NAME = "FaceNet_InceptionResnetV1_VGGFace2"
_CLOTHING_MODEL_NAME = "InternVL3.5-2B"
_UNKNOWN_CLOTHING = {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"}


def _embedding_dim(embedding: list[float] | None) -> int | None:
    return len(embedding) if embedding else None


def _face_identity_block(person_id: str, state: dict, face_crops: list[str]) -> dict:
    embedding = (state.get("face_embedding_by_person") or {}).get(person_id)
    return {
        "model": _FACE_MODEL_NAME,
        "embedding_dim": _embedding_dim(embedding),
        "embedding": embedding,
        "source_crops": face_crops,
        "crop_count": len(face_crops),
        "signal_type": "permanent_biometric_identity",
    }


def _clothing_block(person_id: str, state: dict, best_body_crops: list[str]) -> dict:
    clothing = dict((state.get("clothing_by_person") or {}).get(person_id, _UNKNOWN_CLOTHING))
    return {
        "model": _CLOTHING_MODEL_NAME,
        "top": clothing.get("top", "unknown"),
        "bottom": clothing.get("bottom", "unknown"),
        "shoes": clothing.get("shoes", "unknown"),
        "full": clothing.get("full", "unknown"),
        "source_crops": best_body_crops,
    }


def _reid_block(reid: dict | None) -> dict:
    reid = reid or {}
    # ReID is a same-day supporting appearance signal, not identity.
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


def _color_block(color_signals: dict | None) -> dict:
    color_signals = color_signals or {}
    # Color signals are same-day supporting appearance signals, not identity.
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


def _person_profile(track: dict, state: dict) -> dict:
    person_id = track["person_id"]
    best_by_person = state.get("best_body_crops_by_person") or {}
    reid_by_person = state.get("reid_by_person") or {}
    color_by_person = state.get("color_signals_by_person") or {}

    face_crops = track.get("face_paths", [])
    body_crops = track.get("body_paths", [])
    best_body_crops = best_by_person.get(person_id, [])

    return {
        "person_id": person_id,
        "track": {
            "frame_range": track.get("frame_range", [0, 0]),
            "num_observations": track.get("num_observations", 0),
            "avg_center_movement": track.get("avg_center_movement"),
            "avg_iou": track.get("avg_iou"),
        },
        "identity": {
            "face": _face_identity_block(person_id, state, face_crops),
        },
        "appearance": {
            "date": date.today().isoformat(),
            "clothing": _clothing_block(person_id, state, best_body_crops),
            "reid": _reid_block(reid_by_person.get(person_id)),
            "colors": _color_block(color_by_person.get(person_id)),
        },
        "crops": {
            "faces": face_crops,
            "bodies": body_crops,
            "best_bodies": best_body_crops,
        },
    }


def _model_metadata() -> dict:
    return {
        "person_detector": "yolo26m.pt",
        "face_detector": "YOLOv8-Face",
        "face_embedder": _FACE_MODEL_NAME,
        "clothing_describer": _CLOTHING_MODEL_NAME,
        "reid": "OSNet_x1_0",
        "color_extractor": "DominantColorExtractor_v1",
        "association": "auto_associate",
    }


def build_multi_profile(state: dict) -> dict:
    people = [
        _person_profile(track, state)
        for track in state.get("person_tracks") or []
    ]

    profile = {
        "schema_version": "2.0",
        "profile_type": "multi_person_session",
        "session": {
            "id": state["person_name"].lower(),
            "name": state["person_name"],
            "created_at": date.today().isoformat(),
            "video_sources": state["video_paths"],
            "process_every_n": state.get("process_every_n", 5),
            "people_count": len(people),
        },
        "models": _model_metadata(),
        "people": people,
    }

    feedback = interrupt({
        "message": "Review the multi-person profile below. Reply with 'approve' or provide corrections.",
        "profile_preview": {
            "session": profile["session"],
            "people": [
                {
                    "person_id": person["person_id"],
                    "num_observations": person["track"]["num_observations"],
                    "face_crop_count": person["identity"]["face"]["crop_count"],
                    "best_body_crops": person["crops"]["best_bodies"],
                    "clothing": person["appearance"]["clothing"],
                }
                for person in people
            ],
        },
    })

    override = (feedback or {}).get("clothing_override") or {}
    if override and profile["people"]:
        clothing = profile["people"][0]["appearance"]["clothing"]
        clothing.update({k: v for k, v in override.items() if v})

    return {"profile": profile, "review_feedback": feedback, "approved": True}
