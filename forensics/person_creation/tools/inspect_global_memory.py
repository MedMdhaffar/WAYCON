from __future__ import annotations

import argparse
import sys

from forensics.person_creation.global_memory import GlobalMemoryStore


def _print_persons(store: GlobalMemoryStore, include_inactive: bool = False) -> None:
    persons = store.list_persons(include_inactive=include_inactive)
    if not persons:
        print("persons: none")
        return
    print("persons:")
    for p in persons:
        status = "active" if int(p.get("is_active", 1)) else f"merged_into={p.get('merged_into_person_id')}"
        print(
            f"- {p['person_id']} | {p['name']} | {p['identity_source']} | "
            f"appearances={p['appearance_count']} profile_runs={p['profile_run_count']} "
            f"phone_photos={p['face_photo_count']} | {status}"
        )


def _print_date(store: GlobalMemoryStore, date: str) -> None:
    rows = store.search_by_date(date)
    print(f"appearances on {date}:")
    if not rows:
        print("- none")
        return
    for row in rows:
        print(
            f"- {row['person_id']} | {row['name']} | top={row.get('top') or 'unknown'} | "
            f"bottom={row.get('bottom') or 'unknown'} | shoes={row.get('shoes') or 'unknown'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect person_creation Global Memory.")
    parser.add_argument("--date", help="Show appearances registered for YYYY-MM-DD.")
    parser.add_argument("--include-inactive", action="store_true", help="Include soft-merged inactive persons.")
    args = parser.parse_args()

    try:
        with GlobalMemoryStore() as store:
            print(f"database: {store.db_path}")
            _print_persons(store, include_inactive=args.include_inactive)
            if args.date:
                _print_date(store, args.date)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
