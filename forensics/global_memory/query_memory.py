from __future__ import annotations

import argparse
import json

from forensics.global_memory.store import GlobalMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query WAYCON Global Memory.")
    parser.add_argument("--db", help="SQLite database path.")
    parser.add_argument("--date", help="Appearance date, for example 2026-07-07.")
    parser.add_argument("--camera", help="Camera id inferred from video filename, for example 4.")
    parser.add_argument("--person", help="Person id.")
    args = parser.parse_args()

    store = GlobalMemoryStore(args.db)
    if args.date:
        result = store.get_by_date(args.date)
    elif args.camera:
        result = store.get_by_camera(args.camera)
    elif args.person:
        result = store.get_person(args.person)
    else:
        parser.error("provide one of --date, --camera, or --person")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
