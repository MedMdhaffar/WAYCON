from __future__ import annotations

from datetime import datetime
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

import forensics.global_memory as global_memory_module
from forensics.global_memory import GlobalMemory
from forensics.global_memory.store import ReadOnlyGlobalMemoryError


_PHASE3B_SCHEMA = """
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY, name TEXT NOT NULL, embedding BLOB NOT NULL,
    embedding_count INTEGER NOT NULL DEFAULT 1, enrolled_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, cameras TEXT NOT NULL DEFAULT '[]',
    profile_image TEXT DEFAULT NULL,
    profile_image_source TEXT NOT NULL DEFAULT 'auto'
);
CREATE TABLE appearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id), date TEXT NOT NULL,
    top TEXT, bottom TEXT, shoes TEXT, full_description TEXT,
    top_color TEXT, bottom_color TEXT,
    clothing_status TEXT NOT NULL DEFAULT 'not_attempted',
    best_body_crops TEXT NOT NULL DEFAULT '[]',
    video_sources TEXT NOT NULL DEFAULT '[]', UNIQUE(person_id, date)
);
CREATE TABLE recognition_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL,
    event_type TEXT NOT NULL, similarity REAL,
    embedding_count_before INTEGER, embedding_count_after INTEGER NOT NULL,
    video_sources TEXT NOT NULL DEFAULT '[]', best_face_crop TEXT DEFAULT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE person_gallery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(person_id),
    crop_type TEXT NOT NULL, path TEXT NOT NULL,
    sharpness REAL NOT NULL DEFAULT 0.0, session_date TEXT NOT NULL,
    video_source TEXT, width INTEGER, height INTEGER,
    UNIQUE(person_id, crop_type, path)
);
CREATE TABLE counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
INSERT INTO counters(key, value) VALUES ('person_count', 1);
"""


def _unit(values=(1.0, 0.0, 0.0)) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def _profile(values=(1.0, 0.0, 0.0), day="2026-07-16") -> dict:
    return {
        "face_embedding": _unit(values),
        "face_crops": [],
        "appearance": {"date": day},
        "best_body_crops": [],
        "video_sources": [],
    }


def _create_phase3b_database(path: Path) -> bytes:
    embedding = np.asarray(_unit(), dtype=np.float32).tobytes()
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(_PHASE3B_SCHEMA)
        connection.execute(
            "INSERT INTO persons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "person_001", "Person 001", embedding, 3,
                "2026-07-15", "2026-07-16", '["camera-1"]',
                "person_001/face_crops/face.jpg", "auto",
            ),
        )
        connection.execute(
            """INSERT INTO appearances(
                person_id, date, top, clothing_status,
                best_body_crops, video_sources
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "person_001", "2026-07-16", "black jacket", "ok",
                '["person_001/body_crops/body.jpg"]', '["clip.mp4"]',
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return embedding


def _schema_snapshot(path: Path) -> list[tuple]:
    connection = sqlite3.connect(str(path))
    try:
        return connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
    finally:
        connection.close()


def test_fresh_database_has_phase3d_schema_and_indexes(tmp_path):
    database = tmp_path / "fresh.db"
    memory = GlobalMemory(database)
    try:
        columns = memory._table_columns("persons")
        tables = {
            row[0] for row in memory._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        indexes = {
            row[0] for row in memory._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    finally:
        memory.close()

    assert {"is_active", "merged_into_person_id"} <= columns
    assert {"identity_match_suggestions", "identity_merge_audit"} <= tables
    assert {
        "idx_persons_active", "uq_suggestion_pending",
        "idx_suggestions_status", "idx_suggestions_source",
        "idx_merge_audit_source", "idx_merge_audit_target",
    } <= indexes


def test_phase3b_database_migrates_without_changing_existing_rows(tmp_path):
    database = tmp_path / "phase3b.db"
    embedding = _create_phase3b_database(database)

    memory = GlobalMemory(database)
    try:
        person = memory._conn.execute("SELECT * FROM persons").fetchone()
        appearance = memory._conn.execute("SELECT * FROM appearances").fetchone()
        suggestion_count = memory._conn.execute(
            "SELECT COUNT(*) FROM identity_match_suggestions"
        ).fetchone()[0]
        audit_count = memory._conn.execute(
            "SELECT COUNT(*) FROM identity_merge_audit"
        ).fetchone()[0]
        foreign_key_check = memory._conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        memory.close()

    assert person["embedding"] == embedding
    assert person["embedding_count"] == 3
    assert person["is_active"] == 1
    assert person["merged_into_person_id"] is None
    assert appearance["top"] == "black jacket"
    assert appearance["clothing_status"] == "ok"
    assert suggestion_count == audit_count == 0
    assert foreign_key_check == []


def test_migration_is_idempotent(tmp_path):
    database = tmp_path / "phase3b.db"
    _create_phase3b_database(database)
    first = GlobalMemory(database)
    first.close()
    before = _schema_snapshot(database)

    second = GlobalMemory(database)
    second.close()

    assert _schema_snapshot(database) == before


def test_failed_migration_rolls_back_schema_changes(monkeypatch, tmp_path):
    database = tmp_path / "phase3b.db"
    _create_phase3b_database(database)

    def fail_after_change(self):
        self._conn.execute(
            "ALTER TABLE persons ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
        )
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(GlobalMemory, "_ensure_schema_columns", fail_after_change)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        GlobalMemory(database)

    connection = sqlite3.connect(str(database))
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(persons)")}
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        count = connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    finally:
        connection.close()
    assert "is_active" not in columns
    assert "identity_match_suggestions" not in tables
    assert count == 1


def test_writable_connection_pragmas(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db")
    try:
        assert memory._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert memory._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert memory._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        memory.close()


def test_read_only_connection_pragmas_and_no_journal_write(monkeypatch, tmp_path):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(database)
    writer.register(_profile())
    writer.close()
    statements: list[str] = []
    original_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            statements.append(str(sql))
            return super().execute(sql, parameters)

    def tracking_connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr("forensics.global_memory.store.sqlite3.connect", tracking_connect)
    reader = GlobalMemory(database, read_only=True)
    try:
        assert reader._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        assert reader._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert reader.query_by_face(_unit())
        with pytest.raises(ReadOnlyGlobalMemoryError):
            reader.register(_profile())
    finally:
        reader.close()
    configured = [statement.replace(" ", "").lower() for statement in statements[:2]]
    assert configured == ["pragmaquery_only=on", "pragmabusy_timeout=5000"]
    assert not any("journal_mode=" in statement.lower() for statement in statements)


def _insert_person(memory: GlobalMemory, values=(1.0, 0.0, 0.0)) -> str:
    return memory.register(_profile(values))


def test_suggestion_foreign_keys_and_constraints(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db")
    try:
        source = _insert_person(memory, (1.0, 0.0, 0.0))
        candidate = _insert_person(memory, (0.0, 1.0, 0.0))
        sql = """INSERT INTO identity_match_suggestions(
            source_person_id, candidate_person_id, similarity, status, created_at
        ) VALUES (?, ?, ?, ?, ?)"""
        now = datetime.now().isoformat()
        memory._conn.execute(sql, (source, candidate, 0.8, "pending", now))
        with pytest.raises(sqlite3.IntegrityError):
            memory._conn.execute(sql, ("missing", candidate, 0.8, "pending", now))
        with pytest.raises(sqlite3.IntegrityError):
            memory._conn.execute(sql, (source, "missing", 0.8, "pending", now))
        with pytest.raises(sqlite3.IntegrityError):
            memory._conn.execute(sql, (source, source, 0.8, "pending", now))
        with pytest.raises(sqlite3.IntegrityError):
            memory._conn.execute(sql, (source, candidate, 0.8, "invalid", now))
        with pytest.raises(sqlite3.IntegrityError):
            memory._conn.execute(sql, (source, candidate, 0.8, "pending", now))
        memory._conn.execute(
            "UPDATE identity_match_suggestions SET status='rejected' WHERE id=1"
        )
        memory._conn.execute(sql, (source, candidate, 0.81, "pending", now))
        memory._conn.execute(
            "UPDATE identity_match_suggestions SET status='stale' WHERE id=2"
        )
        memory._conn.execute(sql, (source, candidate, 0.82, "pending", now))
        memory._conn.execute(
            """INSERT INTO identity_merge_audit(
                source_person_id, target_person_id, decision_source, created_at
            ) VALUES ('deleted-source', 'deleted-target', 'test', ?)""",
            (now,),
        )
        assert memory._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        memory.close()


def test_active_person_reads_and_candidate_matching(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db")
    try:
        active = _insert_person(memory, (1.0, 0.0, 0.0))
        inactive = _insert_person(memory, (0.0, 1.0, 0.0))
        memory._conn.execute(
            "UPDATE persons SET is_active=0, merged_into_person_id=? WHERE person_id=?",
            (active, inactive),
        )

        assert [item["person_id"] for item in memory.list_all()] == [active]
        all_people = memory.list_all(include_inactive=True)
        assert [item["person_id"] for item in all_people] == [active, inactive]
        inactive_detail = memory.get_person(inactive)
        assert inactive_detail["is_active"] is False
        assert inactive_detail["merged_into_person_id"] == active
        assert memory.query_by_date("2026-07-16")[0]["person_id"] == active
        memory._conn.execute(
            "UPDATE persons SET cameras='[\"camera-1\"]' WHERE person_id IN (?, ?)",
            (active, inactive),
        )
        assert [item["person_id"] for item in memory.query_by_camera("camera-1")] == [
            active
        ]
        assert memory.query_by_face(_unit((0.0, 1.0, 0.0)), threshold=0.0)[0][
            "person_id"
        ] == active
        normalized = memory._normalize_embedding(_unit((0.0, 1.0, 0.0)))
        assert memory._find_existing_person(normalized, threshold=0.9) is None
    finally:
        memory.close()


def test_all_active_registration_behavior_is_unchanged(tmp_path):
    memory = GlobalMemory(tmp_path / "memory.db")
    try:
        first = memory.register(_profile())
        second = memory.register(_profile(day="2026-07-17"))
        assert first == second
        assert len(memory.list_all()) == 1
    finally:
        memory.close()


def test_memory_persons_api_include_inactive(monkeypatch):
    calls: list[bool] = []

    class FakeMemory:
        def list_all(self, include_inactive=False):
            calls.append(include_inactive)
            return []

        def close(self):
            pass

    monkeypatch.setattr(global_memory_module, "GlobalMemory", FakeMemory)
    from forensics.person_creation.service import app

    client = app.test_client()
    assert client.get("/api/memory/persons").status_code == 200
    assert client.get("/api/memory/persons?include_inactive=true").status_code == 200
    assert calls == [False, True]


def test_two_writers_and_busy_timeout_serialize(tmp_path):
    database = tmp_path / "memory.db"
    first = GlobalMemory(database)
    second = GlobalMemory(database)
    errors: list[BaseException] = []
    barrier = threading.Barrier(3)

    def register(memory, values):
        try:
            barrier.wait(timeout=2)
            memory.register(_profile(values))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=register, args=(first, (1.0, 0.0, 0.0))),
        threading.Thread(target=register, args=(second, (0.0, 1.0, 0.0))),
    ]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=6)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert len(first.list_all()) == 2
    finally:
        first.close()
        second.close()


def test_write_waits_for_lock_and_read_only_lookup_continues(tmp_path):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(database)
    writer.register(_profile())
    waiting_writer = GlobalMemory(database)
    reader = GlobalMemory(database, read_only=True)
    attempted = threading.Event()
    completed = threading.Event()
    errors: list[BaseException] = []

    def blocked_write():
        attempted.set()
        try:
            waiting_writer.register(_profile((0.0, 1.0, 0.0)))
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()

    try:
        writer._conn.execute("BEGIN IMMEDIATE")
        writer._conn.execute(
            "UPDATE persons SET name='Uncommitted' WHERE person_id='person_001'"
        )
        thread = threading.Thread(target=blocked_write)
        thread.start()
        assert attempted.wait(timeout=1)
        assert not completed.wait(timeout=0.2)
        assert reader.query_by_face(_unit())[0]["person_id"] == "person_001"
        writer._conn.execute("COMMIT")
        assert completed.wait(timeout=5)
        thread.join(timeout=1)
        assert errors == []
    finally:
        if writer._conn.in_transaction:
            writer._conn.execute("ROLLBACK")
        reader.close()
        waiting_writer.close()
        writer.close()
