from __future__ import annotations

import argparse
import os
import sys

from forensics.person_creation.global_memory.face_photo_registration import embed_face_photos
from forensics.person_creation.global_memory.similarity import cosine_similarity

_ENV_KEYS = (
    "PERSON_CREATION_USE_FACE_ENGINE",
    "FACE_ENGINE_URL",
    "FACE_ENGINE_FALLBACK_LOCAL",
)


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare local phone-photo embedding with face_engine-backed embedding."
    )
    parser.add_argument("--images", nargs="+", required=True, help="Phone face photo paths.")
    parser.add_argument(
        "--face-engine-url",
        default=os.getenv("FACE_ENGINE_URL", "http://127.0.0.1:5010"),
        help="face_engine base URL.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.9999,
        help="Minimum acceptable cosine similarity.",
    )
    args = parser.parse_args()

    saved = {key: os.getenv(key) for key in _ENV_KEYS}
    try:
        os.environ["PERSON_CREATION_USE_FACE_ENGINE"] = "0"
        local_embedding = embed_face_photos(args.images)

        os.environ["PERSON_CREATION_USE_FACE_ENGINE"] = "1"
        os.environ["FACE_ENGINE_URL"] = args.face_engine_url
        os.environ["FACE_ENGINE_FALLBACK_LOCAL"] = "0"
        service_embedding = embed_face_photos(args.images)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        _restore_env(saved)

    similarity = cosine_similarity(local_embedding, service_embedding)
    print(f"local_dim: {len(local_embedding)}")
    print(f"service_dim: {len(service_embedding)}")
    print(f"cosine_similarity: {similarity:.8f}")
    if similarity < args.min_similarity:
        print(
            f"error: similarity below threshold {args.min_similarity:.8f}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
