"""One-time migration: rename existing cluster IDs to person_### IDs.

Run once:
    python -m forensics.global_memory.migrate_names
"""

from __future__ import annotations

import sqlite3

from .config import DB_PATH


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS counters (
            key   TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO counters (key, value) VALUES ('person_count', 0)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recognition_log (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id              TEXT NOT NULL,
            event_type             TEXT NOT NULL,
            similarity             REAL,
            embedding_count_before INTEGER,
            embedding_count_after  INTEGER NOT NULL,
            video_sources          TEXT NOT NULL DEFAULT '[]',
            ts                     TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_log_person ON recognition_log(person_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_log_ts ON recognition_log(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_log_event ON recognition_log(event_type)")


def run() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)

    existing = conn.execute(
        "SELECT person_id, name FROM persons ORDER BY rowid"
    ).fetchall()

    if not existing:
        conn.execute("UPDATE counters SET value=0 WHERE key='person_count'")
        conn.commit()
        conn.close()
        print("No persons found. Nothing to migrate.")
        print("recognition_log table created.")
        return

    print(f"Migrating {len(existing)} persons...")

    # Use temporary IDs first so reruns or mixed old/new rows cannot collide
    # with UNIQUE(person_id) while we rewrite primary keys.
    temp_pairs = []
    for i, row in enumerate(existing, start=1):
        old_id = row["person_id"]
        temp_id = f"__migration_tmp_{i:03d}"
        temp_pairs.append((old_id, temp_id))
        if old_id != temp_id:
            conn.execute(
                "UPDATE persons SET person_id=? WHERE person_id=?",
                (temp_id, old_id),
            )
            conn.execute(
                "UPDATE appearances SET person_id=? WHERE person_id=?",
                (temp_id, old_id),
            )
            conn.execute(
                "UPDATE recognition_log SET person_id=? WHERE person_id=?",
                (temp_id, old_id),
            )

    for i, (old_id, temp_id) in enumerate(temp_pairs, start=1):
        new_id = f"person_{i:03d}"
        new_name = f"Person {i:03d}"
        conn.execute(
            "UPDATE persons SET person_id=?, name=? WHERE person_id=?",
            (new_id, new_name, temp_id),
        )
        conn.execute(
            "UPDATE appearances SET person_id=? WHERE person_id=?",
            (new_id, temp_id),
        )
        conn.execute(
            "UPDATE recognition_log SET person_id=? WHERE person_id=?",
            (new_id, temp_id),
        )
        print(f"  {old_id} -> {new_id} ({new_name})")

    conn.execute(
        "UPDATE counters SET value=? WHERE key='person_count'",
        (len(existing),),
    )

    conn.commit()
    conn.close()
    print("recognition_log table created.")
    print("Migration complete.")
    print(f"Counter seeded at {len(existing)}.")
    print("Next enrollment will be assigned person_{:03d}".format(len(existing) + 1))


if __name__ == "__main__":
    run()
