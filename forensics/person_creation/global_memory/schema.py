from __future__ import annotations

import sqlite3


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, column_sql: str) -> None:
    column_name = column_sql.split()[0]
    if column_name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_sql}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            face_embedding_json TEXT NOT NULL,
            identity_source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (identity_source IN ('video_profile', 'phone_photo', 'mixed'))
        );

        CREATE TABLE IF NOT EXISTS appearances (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            date TEXT,
            top TEXT,
            bottom TEXT,
            shoes TEXT,
            full TEXT,
            color_signals_json TEXT,
            reid_json TEXT,
            body_reid_embedding_json TEXT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS profile_runs (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            profile_path TEXT,
            output_dir TEXT,
            video_sources_json TEXT,
            cluster_id TEXT,
            cluster_confidence REAL,
            cluster_face_count INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS face_photo_sources (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            face_crop_path TEXT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS crop_references (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            crop_type TEXT NOT NULL,
            crop_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (crop_type IN ('face', 'body', 'best_body')),
            FOREIGN KEY (person_id) REFERENCES persons(person_id)
        );

        CREATE TABLE IF NOT EXISTS identity_match_suggestions (
            id TEXT PRIMARY KEY,
            new_person_id TEXT NOT NULL,
            candidate_person_id TEXT NOT NULL,
            similarity REAL NOT NULL,
            threshold REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolved_at TEXT NULL,
            CHECK (status IN ('pending', 'accepted', 'rejected')),
            FOREIGN KEY (new_person_id) REFERENCES persons(person_id),
            FOREIGN KEY (candidate_person_id) REFERENCES persons(person_id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_appearances_person_date
            ON appearances(person_id, date);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_profile_runs_profile_path
            ON profile_runs(profile_path)
            WHERE profile_path IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS ux_face_photo_sources_person_image
            ON face_photo_sources(person_id, image_path);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_crop_references_person_type_path
            ON crop_references(person_id, crop_type, crop_path);
        CREATE INDEX IF NOT EXISTS ix_appearances_date ON appearances(date);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_identity_match_suggestions_pending
            ON identity_match_suggestions(new_person_id, candidate_person_id)
            WHERE status = 'pending';
        """
    )
    _add_column(conn, "persons", "notes TEXT NULL")
    _add_column(conn, "persons", "merged_into_person_id TEXT NULL")
    _add_column(conn, "persons", "is_active INTEGER NOT NULL DEFAULT 1")
