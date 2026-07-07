from __future__ import annotations

import argparse
import json

from forensics.global_memory.store import GlobalMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a profile.json file into WAYCON Global Memory.")
    parser.add_argument("--profile", required=True, help="Path to profile.json.")
    parser.add_argument("--db", help="Optional SQLite database path.")
    args = parser.parse_args()

    store = GlobalMemoryStore(args.db)
    result = store.register_profile_file(args.profile)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
