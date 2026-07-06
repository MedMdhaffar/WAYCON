from datetime import date

from langgraph.types import interrupt


_UNKNOWN = {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"}


def _track_people(state: dict) -> list[dict]:
    best_by_person = state.get("best_body_crops_by_person") or {}
    clothing_by_person = state.get("clothing_by_person") or {}
    people = []

    for track in state.get("person_tracks") or []:
        person_id = track["person_id"]
        people.append({
            "person_id": person_id,
            "description": clothing_by_person.get(person_id, dict(_UNKNOWN)),
            "best_body_crops": best_by_person.get(person_id, []),
            "face_crops": track.get("face_paths", []),
            "body_crops": track.get("body_paths", []),
            "frame_range": track.get("frame_range", [0, 0]),
            "num_observations": track.get("num_observations", 0),
        })

    return people


def build_multi_profile(state: dict) -> dict:
    people = _track_people(state)
    first = people[0] if people else None
    first_description = first["description"] if first else dict(_UNKNOWN)

    profile = {
        "id": state["person_name"].lower(),
        "name": state["person_name"],
        "created_at": date.today().isoformat(),
        "people_count": len(people),
        "people": people,
        "video_sources": state["video_paths"],
        "face_embedding": state.get("mean_face_embedding", []),
        "face_crop_count": len(first["face_crops"]) if first else 0,
        "face_crops": first["face_crops"] if first else [],
        "appearance": {
            "date": date.today().isoformat(),
            **first_description,
        },
        "body_crops": first["body_crops"] if first else [],
        "best_body_crops": first["best_body_crops"] if first else [],
    }

    feedback = interrupt({
        "message": "Review the multi-person profile below. Reply with 'approve' or provide corrections.",
        "profile_preview": {
            "name": profile["name"],
            "people_count": profile["people_count"],
            "people": [
                {
                    "person_id": person["person_id"],
                    "num_observations": person["num_observations"],
                    "face_crop_count": len(person["face_crops"]),
                    "best_body_crops": person["best_body_crops"],
                    "appearance": person["description"],
                }
                for person in people
            ],
            "face_crop_count": profile["face_crop_count"],
            "associations_count": len(state.get("associations") or []),
            "appearance": profile["appearance"],
            "best_body_crops": profile["best_body_crops"],
        },
    })

    override = (feedback or {}).get("clothing_override") or {}
    if override and first:
        profile["appearance"].update({k: v for k, v in override.items() if v})
        profile["people"][0]["description"].update({k: v for k, v in override.items() if v})

    return {"profile": profile, "review_feedback": feedback, "approved": True}
