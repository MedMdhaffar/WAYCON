from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
import forensics.global_memory.migrate_names as migrate_names


def _profile(values) -> dict:
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return {
        "face_embedding": vector.tolist(),
        "face_crops": [],
        "appearance": {"date": "2026-07-20"},
        "best_body_crops": [],
        "video_sources": [],
    }


def _create_legacy_database(
    path: Path,
    *,
    include_gallery: bool = True,
    populate_gallery: bool = True,
) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE persons (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE appearances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                date TEXT NOT NULL
            );
            CREATE TABLE recognition_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                similarity REAL,
                embedding_count_before INTEGER,
                embedding_count_after INTEGER NOT NULL,
                video_sources TEXT NOT NULL DEFAULT '[]',
                best_face_crop TEXT DEFAULT NULL,
                ts TEXT NOT NULL
            );
            INSERT INTO persons VALUES ('malek', 'Malek');
            INSERT INTO persons VALUES ('cluster_3', 'Cluster 3');
            INSERT INTO appearances(person_id, date)
            VALUES ('malek', '2026-07-20');
            INSERT INTO appearances(person_id, date)
            VALUES ('cluster_3', '2026-07-21');
            INSERT INTO recognition_log(
                person_id, event_type, similarity,
                embedding_count_before, embedding_count_after,
                video_sources, best_face_crop, ts
            ) VALUES (
                'malek', 'recognized', 0.91, 2, 3,
                '["camera-a.mp4"]', 'malek/face_crops/best.jpg',
                '2026-07-20T10:00:00'
            );
            INSERT INTO recognition_log(
                person_id, event_type, similarity,
                embedding_count_before, embedding_count_after,
                video_sources, best_face_crop, ts
            ) VALUES (
                'cluster_3', 'new_enrollment', NULL, NULL, 1,
                '["camera-b.mp4"]', 'cluster_3/face_crops/best.jpg',
                '2026-07-21T11:00:00'
            );
            """
        )
        if include_gallery:
            connection.executescript(
                """
                CREATE TABLE person_gallery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL REFERENCES persons(person_id),
                    crop_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sharpness REAL NOT NULL DEFAULT 0.0,
                    session_date TEXT NOT NULL,
                    video_source TEXT,
                    width INTEGER,
                    height INTEGER,
                    UNIQUE(person_id, crop_type, path)
                );
                """
            )
            if populate_gallery:
                connection.executemany(
                    """INSERT INTO person_gallery(
                        id, person_id, crop_type, path, sharpness,
                        session_date, video_source, width, height
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            7,
                            "malek",
                            "face",
                            "malek/face_crops/face-001.jpg",
                            0.93,
                            "2026-07-20",
                            "camera-a.mp4",
                            640,
                            480,
                        ),
                        (
                            11,
                            "cluster_3",
                            "body",
                            "cluster_3/body_crops/body-002.png",
                            0.71,
                            "2026-07-21",
                            "camera-b.mp4",
                            360,
                            720,
                        ),
                    ],
                )
        connection.commit()
    finally:
        connection.close()


def _rows(path: Path, table: str) -> list[tuple]:
    connection = sqlite3.connect(str(path))
    try:
        return connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
    finally:
        connection.close()


def test_phase3d_database_is_refused_before_any_write(monkeypatch, tmp_path):
    database = tmp_path / "phase3d.db"
    memory = GlobalMemory(database)
    try:
        source = memory.register(_profile((1.0, 0.0, 0.0)))
        candidate = memory.register(_profile((0.0, 1.0, 0.0)))
        memory._conn.execute(
            """INSERT INTO identity_match_suggestions(
                source_person_id, candidate_person_id, similarity, created_at
            ) VALUES (?, ?, 0.8, ?)""",
            (source, candidate, datetime.now().isoformat()),
        )
        memory._conn.execute(
            """INSERT INTO identity_merge_audit(
                source_person_id, target_person_id, decision_source,
                source_embedding, source_embedding_count, created_at
            ) VALUES ('historical-source', 'historical-target', 'test', ?, 1, ?)""",
            (b"historical-vector", datetime.now().isoformat()),
        )
        memory._conn.execute(
            """INSERT INTO person_gallery(
                person_id, crop_type, path, session_date
            ) VALUES (?, 'face', 'person_001/face.jpg', '2026-07-20')""",
            (source,),
        )
        memory._conn.execute(
            """INSERT INTO person_gallery(
                person_id, crop_type, path, session_date
            ) VALUES (?, 'body', 'person_002/body.jpg', '2026-07-20')""",
            (candidate,),
        )
    finally:
        memory.close()

    before = {
        table: _rows(database, table)
        for table in (
            "persons",
            "identity_match_suggestions",
            "identity_merge_audit",
            "person_gallery",
        )
    }
    monkeypatch.setattr(migrate_names, "DB_PATH", database)
    with pytest.raises(RuntimeError, match="refuses Phase 3D"):
        migrate_names.run()

    after = {table: _rows(database, table) for table in before}
    connection = sqlite3.connect(str(database))
    try:
        foreign_key_check = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    assert after == before
    assert foreign_key_check == []


def test_supported_legacy_name_migration_is_transactional(monkeypatch, tmp_path):
    database = tmp_path / "legacy.db"
    _create_legacy_database(database)
    gallery_before = _rows(database, "person_gallery")
    monkeypatch.setattr(migrate_names, "DB_PATH", database)

    migrate_names.run()

    connection = sqlite3.connect(str(database))
    try:
        people = connection.execute(
            "SELECT person_id, name FROM persons ORDER BY person_id"
        ).fetchall()
        appearances = connection.execute(
            "SELECT person_id, date FROM appearances ORDER BY id"
        ).fetchall()
        recognition_log = connection.execute(
            "SELECT person_id FROM recognition_log ORDER BY id"
        ).fetchall()
        gallery = connection.execute(
            "SELECT * FROM person_gallery ORDER BY id"
        ).fetchall()
        counter = connection.execute(
            "SELECT value FROM counters WHERE key='person_count'"
        ).fetchone()[0]
        foreign_key_check = connection.execute("PRAGMA foreign_key_check").fetchall()
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()

    assert people == [("person_001", "Person 001"), ("person_002", "Person 002")]
    assert appearances == [
        ("person_001", "2026-07-20"),
        ("person_002", "2026-07-21"),
    ]
    assert recognition_log == [("person_001",), ("person_002",)]
    expected_gallery = [
        (row[0], f"person_{index:03d}", *row[2:])
        for index, row in enumerate(gallery_before, start=1)
    ]
    assert gallery == expected_gallery
    assert len(gallery) == len(gallery_before)
    assert not ({"malek", "cluster_3"} & {row[1] for row in gallery})
    assert counter == 2
    assert foreign_key_check == []
    assert journal_mode == "wal"


@pytest.mark.parametrize("gallery_mode", ["empty", "absent"])
def test_legacy_gallery_compatibility(monkeypatch, tmp_path, gallery_mode):
    database = tmp_path / f"legacy-{gallery_mode}.db"
    _create_legacy_database(
        database,
        include_gallery=gallery_mode != "absent",
        populate_gallery=False,
    )
    monkeypatch.setattr(migrate_names, "DB_PATH", database)

    migrate_names.run()

    connection = sqlite3.connect(str(database))
    try:
        people = connection.execute(
            "SELECT person_id FROM persons ORDER BY rowid"
        ).fetchall()
        gallery_exists = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='person_gallery'"
        ).fetchone()
        gallery_count = (
            connection.execute("SELECT COUNT(*) FROM person_gallery").fetchone()[0]
            if gallery_exists
            else None
        )
        foreign_key_check = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()

    assert people == [("person_001",), ("person_002",)]
    assert bool(gallery_exists) is (gallery_mode == "empty")
    assert gallery_count == (0 if gallery_mode == "empty" else None)
    assert foreign_key_check == []


def test_injected_legacy_failure_rolls_back_completely(monkeypatch, tmp_path):
    database = tmp_path / "legacy.db"
    _create_legacy_database(database)
    people_before = _rows(database, "persons")
    appearances_before = _rows(database, "appearances")
    recognition_log_before = _rows(database, "recognition_log")
    gallery_before = _rows(database, "person_gallery")
    original_connect = sqlite3.connect
    failing_connections = []

    class FailingConnection(sqlite3.Connection):
        failed = False
        closed = False

        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            normalized = " ".join(str(sql).split()).upper()
            if (
                not self.failed
                and normalized.startswith("UPDATE PERSON_GALLERY SET PERSON_ID")
            ):
                self.failed = True
                raise RuntimeError("injected legacy migration failure")
            return result

        def close(self):
            self.closed = True
            return super().close()

    def failing_connect(*args, **kwargs):
        kwargs["factory"] = FailingConnection
        connection = original_connect(*args, **kwargs)
        failing_connections.append(connection)
        return connection

    monkeypatch.setattr(migrate_names, "DB_PATH", database)
    monkeypatch.setattr(migrate_names.sqlite3, "connect", failing_connect)
    with pytest.raises(RuntimeError, match="injected legacy migration failure"):
        migrate_names.run()

    connection = original_connect(str(database), timeout=1.0)
    try:
        assert connection.execute("SELECT * FROM persons ORDER BY rowid").fetchall() == people_before
        assert connection.execute("SELECT * FROM appearances ORDER BY rowid").fetchall() == appearances_before
        assert connection.execute(
            "SELECT * FROM recognition_log ORDER BY rowid"
        ).fetchall() == recognition_log_before
        assert connection.execute(
            "SELECT * FROM person_gallery ORDER BY rowid"
        ).fetchall() == gallery_before
        assert not {
            row[0]
            for row in connection.execute(
                "SELECT person_id FROM persons "
                "WHERE person_id LIKE '__migration_tmp_%'"
            ).fetchall()
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
    finally:
        connection.close()

    assert "counters" not in tables
    assert failing_connections[0].closed is True
