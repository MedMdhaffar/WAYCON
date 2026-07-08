from __future__ import annotations

import argparse
import sys

from forensics.person_creation.global_memory import GlobalMemoryStore
from forensics.person_creation.global_memory.config import FACE_AUTO_MATCH_THRESHOLD


def main() -> int:
    parser = argparse.ArgumentParser(description="Register clear phone face photos into Global Memory.")
    parser.add_argument("--name", required=True, help="Person name.")
    parser.add_argument("--images", nargs="+", required=True, help="Phone face photo paths.")
    parser.add_argument("--threshold", type=float, default=FACE_AUTO_MATCH_THRESHOLD, help="Face match threshold.")
    args = parser.parse_args()

    try:
        with GlobalMemoryStore() as store:
            person_id = store.register_face_photo_identity(
                name=args.name,
                image_paths=args.images,
                threshold=args.threshold,
            )
            result = store.last_registration_result
            print(f"database: {store.db_path}")
            print(f"person_id: {person_id}")
            print(f"result: {result.get('action', 'created')} phone_photo identity")
            if result.get("possible_duplicate") and result.get("best_match"):
                match = result["best_match"]
                print(
                    "possible_duplicate: "
                    f"{match['name']} ({match['person_id']}) similarity={match['similarity']:.4f}; "
                    "created separate identity"
                )
            print(f"registered_images: {result.get('registered_images', 0)}")
            for skipped in result.get("skipped", []):
                print(f"skipped: {skipped['image_path']} - {skipped['reason']}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
