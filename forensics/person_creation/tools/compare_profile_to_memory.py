from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forensics.person_creation.global_memory import GlobalMemoryStore
from forensics.person_creation.global_memory.config import FACE_AUTO_MATCH_THRESHOLD

_THRESHOLD = FACE_AUTO_MATCH_THRESHOLD


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a finalized profile.json face embedding to Global Memory.")
    parser.add_argument("--profile", required=True, help="Path to cluster_<id>/profile.json.")
    args = parser.parse_args()

    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"error: profile path does not exist: {profile_path}", file=sys.stderr)
        return 1
    if not profile_path.is_file():
        print(f"error: profile path is not a file: {profile_path}", file=sys.stderr)
        return 1

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"error: failed to read profile JSON: {exc}", file=sys.stderr)
        return 1

    embedding = profile.get("face_embedding")
    if not embedding:
        print("error: profile has no face_embedding", file=sys.stderr)
        return 1

    try:
        with GlobalMemoryStore() as store:
            matches = store.search_by_face(embedding, top_k=10, threshold=None)
            print(f"Global Memory DB: {store.db_path}")
            print(f"Profile: {profile_path}")
            print(f"Threshold: {_THRESHOLD}")
            print()

            if not store.list_persons():
                print("Global Memory has no persons yet.")
                return 0

            if not matches:
                print("Matches: none")
                print()
                print("Best match passes threshold: no")
                return 0

            print("Matches:")
            for idx, match in enumerate(matches, start=1):
                passed = match["similarity"] >= _THRESHOLD
                status = "PASS" if passed else "FAIL"
                print(
                    f"{idx}. {match['name']} | person_id={match['person_id']} | "
                    f"source={match['identity_source']} | similarity={match['similarity']:.4f} | {status}"
                )

            best_passes = matches[0]["similarity"] >= _THRESHOLD
            print()
            print(f"Best match passes threshold: {'yes' if best_passes else 'no'}")
    except Exception as exc:
        print(f"error: compare failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
