import argparse
import json
from pathlib import Path

import numpy as np


_FORBIDDEN_TOP_LEVEL = {
    "face_embedding",
    "appearance",
    "reid",
    "color_signals",
    "face_crops",
    "body_crops",
    "best_body_crops",
}


def _norm(values) -> float | None:
    if not values:
        return None
    return float(np.linalg.norm(np.asarray(values, dtype=np.float32)))


def _check_embedding_norm(label: str, embedding, errors: list[str], lines: list[str]) -> None:
    norm = _norm(embedding)
    if norm is None:
        lines.append(f"{label}_norm=missing")
        return
    lines.append(f"{label}_norm={norm:.4f}")
    if not np.isclose(norm, 1.0, atol=1e-3):
        errors.append(f"{label} embedding norm is {norm:.4f}, expected approximately 1.0")


def _check_crop_sources(person: dict, errors: list[str]) -> None:
    person_id = person.get("person_id", "<missing>")
    crops = person.get("crops") or {}
    bodies = set(crops.get("bodies") or [])
    best_bodies = set(crops.get("best_bodies") or [])
    valid_body_sources = bodies | best_bodies

    appearance = person.get("appearance") or {}
    for signal_name in ("clothing", "reid", "colors"):
        signal = appearance.get(signal_name) or {}
        for source in signal.get("source_crops") or []:
            if source not in valid_body_sources:
                errors.append(
                    f"{person_id} {signal_name} source crop is not owned by that person: {source}"
                )


def validate(profile: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = []

    if not profile.get("schema_version"):
        errors.append("missing schema_version")
    if not isinstance(profile.get("session"), dict):
        errors.append("missing session object")
    if not isinstance(profile.get("people"), list):
        errors.append("missing people list")
        people = []
    else:
        people = profile["people"]

    forbidden = sorted(_FORBIDDEN_TOP_LEVEL & set(profile))
    if forbidden:
        errors.append(f"forbidden top-level legacy keys present: {', '.join(forbidden)}")

    session = profile.get("session") or {}
    if session.get("people_count") != len(people):
        errors.append(
            f"session.people_count={session.get('people_count')} does not match len(people)={len(people)}"
        )

    seen_crops: dict[str, str] = {}
    for person in people:
        person_id = person.get("person_id")
        if not person_id:
            errors.append("person missing person_id")
            person_id = "<missing>"

        for key in ("track", "identity", "appearance", "crops"):
            if not isinstance(person.get(key), dict):
                errors.append(f"{person_id} missing {key}")

        identity = person.get("identity") or {}
        appearance = person.get("appearance") or {}
        crops = person.get("crops") or {}
        face = identity.get("face") or {}

        for key in ("clothing", "reid", "colors"):
            if not isinstance(appearance.get(key), dict):
                errors.append(f"{person_id} missing appearance.{key}")

        line_parts = [person_id]
        _check_embedding_norm(f"{person_id} face", face.get("embedding"), errors, line_parts)
        _check_embedding_norm(
            f"{person_id} reid",
            (appearance.get("reid") or {}).get("embedding"),
            errors,
            line_parts,
        )
        lines.append(" ".join(line_parts))

        _check_crop_sources(person, errors)
        owned_crops = [
            *(crops.get("faces") or []),
            *(crops.get("bodies") or []),
            *(crops.get("best_bodies") or []),
        ]
        for crop in owned_crops:
            owner = seen_crops.get(crop)
            if owner and owner != person_id:
                errors.append(f"crop appears in multiple people: {crop} in {owner} and {person_id}")
            seen_crops[crop] = person_id

    return errors, lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WAYCON profile schema v2.")
    parser.add_argument("profile_json", help="Path to profile.json")
    args = parser.parse_args()

    path = Path(args.profile_json)
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)

    errors, lines = validate(profile)
    if errors:
        print("[validate_profile_schema] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("[validate_profile_schema] PASS")
    print(f"people_count={len(profile.get('people') or [])}")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
