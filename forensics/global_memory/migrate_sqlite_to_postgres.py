"""One-time backfill: copy an existing SQLite global_memory.db into Postgres.

GlobalMemory no longer talks to SQLite at all (see store.py) -- this script is the
only remaining SQLite consumer, and only for this one-shot migration. It bypasses
GlobalMemory.register() deliberately: register() re-runs identity matching and
would reassign person_ids / recompute merged embeddings, which is wrong for a
backfill -- the goal is an exact copy (same person_ids, same embeddings, same
timestamps), not re-enrollment.

Idempotent: every insert uses ON CONFLICT DO NOTHING keyed on each table's natural
identity, so re-running against the same Postgres target after a partial run (or a
crash) skips rows that already made it across instead of duplicating or failing.

Usage:
    python -m forensics.global_memory.migrate_sqlite_to_postgres \\
        --sqlite-path forensics/global_memory.db

    # dry run: report what would be migrated without writing anything
    python -m forensics.global_memory.migrate_sqlite_to_postgres --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config


def _open_sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _migrate_persons(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    rows = sconn.execute("SELECT * FROM persons").fetchall()
    count = 0
    for row in rows:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO persons (
                person_id, name, embedding, embedding_count, enrolled_at, updated_at,
                cameras, profile_image, profile_image_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id) DO NOTHING
            """,
            (
                row["person_id"],
                row["name"],
                embedding,
                row["embedding_count"],
                row["enrolled_at"],
                row["updated_at"],
                row["cameras"],
                row["profile_image"],
                row["profile_image_source"] if "profile_image_source" in row.keys() else "auto",
            ),
        )
        count += 1
    return count


def _migrate_appearances(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    rows = sconn.execute("SELECT * FROM appearances").fetchall()
    count = 0
    for row in rows:
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO appearances (
                person_id, date, segment_id, top, bottom, shoes, full_description,
                top_color, bottom_color, best_body_crops, video_sources
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id, date) DO NOTHING
            """,
            (
                row["person_id"],
                row["date"],
                row["segment_id"] if "segment_id" in row.keys() else None,
                row["top"],
                row["bottom"],
                row["shoes"],
                row["full_description"],
                row["top_color"],
                row["bottom_color"],
                row["best_body_crops"],
                row["video_sources"],
            ),
        )
        count += 1
    return count


def _migrate_recognition_log(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    rows = sconn.execute("SELECT * FROM recognition_log").fetchall()
    count = 0
    for row in rows:
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO recognition_log (
                person_id, event_type, similarity, embedding_count_before,
                embedding_count_after, video_sources, best_face_crop, ts
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["person_id"],
                row["event_type"],
                row["similarity"],
                row["embedding_count_before"],
                row["embedding_count_after"],
                row["video_sources"],
                row["best_face_crop"] if "best_face_crop" in row.keys() else None,
                row["ts"],
            ),
        )
        count += 1
    return count


def _migrate_person_gallery(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    rows = sconn.execute("SELECT * FROM person_gallery").fetchall()
    count = 0
    for row in rows:
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO person_gallery (
                person_id, crop_type, path, sharpness, session_date,
                video_source, width, height
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id, crop_type, path) DO NOTHING
            """,
            (
                row["person_id"],
                row["crop_type"],
                row["path"],
                row["sharpness"],
                row["session_date"],
                row["video_source"],
                row["width"],
                row["height"],
            ),
        )
        count += 1
    return count


def _migrate_clothing_jobs(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    if not _table_exists(sconn, "clothing_jobs"):
        return 0
    rows = sconn.execute("SELECT * FROM clothing_jobs").fetchall()
    count = 0
    for row in rows:
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO clothing_jobs (
                job_id, person_id, segment_id, crop_path, pipeline_version,
                status, attempts, next_attempt_at, error, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO NOTHING
            """,
            (
                row["job_id"], row["person_id"], row["segment_id"], row["crop_path"],
                row["pipeline_version"], row["status"], row["attempts"],
                row["next_attempt_at"] if "next_attempt_at" in row.keys() else None,
                row["error"], row["created_at"], row["updated_at"],
            ),
        )
        count += 1
    return count


def _migrate_segments(sconn: sqlite3.Connection, pconn: psycopg.Connection, dry_run: bool) -> int:
    if not _table_exists(sconn, "segments"):
        return 0
    rows = sconn.execute("SELECT * FROM segments").fetchall()
    count = 0
    for row in rows:
        if dry_run:
            count += 1
            continue
        pconn.execute(
            """
            INSERT INTO segments (
                segment_id, seq_num, codec, pipeline_version, segment_start_ts,
                segment_end_ts, status, retry_count, segment_incomplete, error,
                created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (segment_id) DO NOTHING
            """,
            (
                row["segment_id"], row["seq_num"], row["codec"],
                row["pipeline_version"] if "pipeline_version" in row.keys() else config.PIPELINE_VERSION,
                row["segment_start_ts"], row["segment_end_ts"], row["status"],
                row["retry_count"], bool(row["segment_incomplete"]), row["error"],
                row["created_at"], row["updated_at"],
            ),
        )
        count += 1
    return count


def _sync_counter(pconn: psycopg.Connection, dry_run: bool) -> int:
    """After copying `persons`, seed counters.person_count from the highest migrated
    person_XXX id (same defensive intent as store.py::_next_person_id's collision
    guard) so the next enrollment continues the sequence instead of restarting at 1
    and colliding with an already-migrated person_id.
    """
    row = pconn.execute("SELECT person_id FROM persons").fetchall()
    max_n = 0
    for r in row:
        pid = r["person_id"]
        if pid.startswith("person_"):
            suffix = pid[len("person_"):]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
    if not dry_run:
        pconn.execute(
            "UPDATE counters SET value = %s WHERE key = 'person_count' AND value < %s",
            (max_n, max_n),
        )
    return max_n


def run(sqlite_path: str, *, dry_run: bool = False) -> None:
    sqlite_file = Path(sqlite_path)
    if not sqlite_file.exists():
        print(f"SQLite source not found: {sqlite_file}", file=sys.stderr)
        sys.exit(1)

    sconn = _open_sqlite(str(sqlite_file))
    pdsn = config.connection_string()
    print(f"Source (SQLite): {sqlite_file}")
    print(f"Target (Postgres): host={config.HOST} port={config.PORT} db={config.DBNAME}")
    if dry_run:
        print("--dry-run: no writes will be made\n")

    with psycopg.connect(pdsn, row_factory=psycopg.rows.dict_row) as pconn:
        register_vector(pconn)
        if not dry_run:
            schema_path = Path(__file__).with_name("schema_postgres.sql")
            pconn.execute(schema_path.read_text(encoding="utf-8"))

        with pconn.transaction():
            n_persons = _migrate_persons(sconn, pconn, dry_run)
            n_appearances = _migrate_appearances(sconn, pconn, dry_run)
            n_log = _migrate_recognition_log(sconn, pconn, dry_run)
            n_gallery = _migrate_person_gallery(sconn, pconn, dry_run)
            n_clothing_jobs = _migrate_clothing_jobs(sconn, pconn, dry_run)
            n_segments = _migrate_segments(sconn, pconn, dry_run)
            max_person_n = _sync_counter(pconn, dry_run)
            # Every _migrate_* helper above is itself a no-op write when dry_run is
            # set (they only count rows), and the schema-creation call above is
            # skipped entirely for dry_run -- so this transaction has nothing to
            # roll back in that case; it commits trivially.

    sconn.close()

    print()
    print(f"persons:          {n_persons}")
    print(f"appearances:      {n_appearances}")
    print(f"recognition_log:  {n_log}")
    print(f"person_gallery:   {n_gallery}")
    print(f"clothing_jobs:    {n_clothing_jobs}")
    print(f"segments:         {n_segments}")
    print(f"counters seeded:  person_count >= {max_person_n}")
    print("\nDone." if not dry_run else "\nDry run complete -- nothing was written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sqlite-path", default=config.DB_PATH, help=f"default: {config.DB_PATH}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.sqlite_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
