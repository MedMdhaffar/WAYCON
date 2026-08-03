from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

import forensics.global_memory as global_memory_module
from forensics.global_memory import GlobalMemory
from forensics.person_creation import service


def _unit(values: tuple[float, float, float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def _profile(values: tuple[float, float, float]) -> dict:
    return {
        "face_embedding": _unit(values),
        "face_crops": [],
        "appearance": {"date": "2026-07-22"},
        "best_body_crops": [],
        "video_sources": ["live-camera"],
    }


def _seed_api_database(database: Path, media_root: Path) -> tuple[str, int]:
    memory = GlobalMemory(database, media_root=media_root)
    try:
        source_id = memory.register(_profile((1.0, 0.0, 0.0)))
        candidate_id = memory.register(_profile((0.0, 1.0, 0.0)))
        cursor = memory._conn.execute(
            """
            INSERT INTO identity_match_suggestions(
                source_person_id, candidate_person_id, similarity,
                second_similarity, margin, reason, status, created_at
            ) VALUES(?, ?, 0.88, 0.70, 0.18, 'ambiguous_match',
                     'pending', '2026-07-22')
            """,
            (source_id, candidate_id),
        )
        return source_id, int(cursor.lastrowid)
    finally:
        memory.close()


@pytest.fixture
def memory_api(tmp_path, monkeypatch):
    database = tmp_path / "memory.db"
    media_root = tmp_path / "person_db"
    media_root.mkdir()
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(media_root))
    service.app.config.update(TESTING=True, GLOBAL_MEMORY_READ_ONLY=False)
    person_id, review_id = _seed_api_database(database, media_root)
    yield database, media_root, person_id, review_id
    service.app.config["GLOBAL_MEMORY_READ_ONLY"] = False


def test_repeated_person_reads_succeed_during_identity_persistence_without_lost_writes(
    memory_api,
):
    database, media_root, person_id, _ = memory_api
    write_count = 24
    writer_started = threading.Event()
    writer_errors: list[BaseException] = []

    def persist_identities() -> None:
        memory = GlobalMemory(database, media_root=media_root)
        try:
            writer_started.set()
            for _ in range(write_count):
                assigned = memory.register(_profile((1.0, 0.0, 0.0)))
                assert assigned == person_id
                # Keep the writer active long enough to exercise repeated polling.
                threading.Event().wait(0.002)
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            memory.close()

    writer = threading.Thread(target=persist_identities)
    writer.start()
    assert writer_started.wait(timeout=2)

    statuses: list[int] = []
    with service.app.test_client() as client:
        while writer.is_alive() or len(statuses) < 12:
            response = client.get("/api/memory/persons")
            statuses.append(response.status_code)

    writer.join(timeout=8)
    assert not writer.is_alive()
    assert writer_errors == []
    assert statuses
    assert set(statuses) == {200}

    verifier = GlobalMemory(database, media_root=media_root, read_only=True)
    try:
        person = verifier.get_person(person_id)
        assert person is not None
        assert person["embedding_count"] == write_count + 1
        assert len(verifier.get_recognition_history(person_id, limit=100)) == (
            write_count + 1
        )
    finally:
        verifier.close()


def test_detail_and_review_reads_finish_while_write_transaction_is_active(memory_api):
    database, media_root, person_id, review_id = memory_api
    writer = GlobalMemory(database, media_root=media_root)
    write_lock_held = threading.Event()
    release_write = threading.Event()
    writer_errors: list[BaseException] = []

    def hold_identity_write() -> None:
        try:
            writer._conn.execute("BEGIN IMMEDIATE")
            writer._conn.execute(
                "UPDATE persons SET embedding_count = embedding_count + 1 "
                "WHERE person_id = ?",
                (person_id,),
            )
            write_lock_held.set()
            if not release_write.wait(timeout=5):
                raise TimeoutError("test did not release the write transaction")
            writer._conn.execute("COMMIT")
        except BaseException as exc:
            if writer._conn.in_transaction:
                writer._conn.execute("ROLLBACK")
            writer_errors.append(exc)

    writer_thread = threading.Thread(target=hold_identity_write)
    writer_thread.start()
    assert write_lock_held.wait(timeout=2)

    paths = [
        f"/api/memory/persons/{person_id}",
        "/api/identity-reviews",
        f"/api/identity-reviews/{review_id}",
    ]
    responses: dict[str, int] = {}
    reader_errors: list[BaseException] = []

    def read_endpoint(path: str) -> None:
        try:
            with service.app.test_client() as client:
                responses[path] = client.get(path).status_code
        except BaseException as exc:
            reader_errors.append(exc)

    readers = [threading.Thread(target=read_endpoint, args=(path,)) for path in paths]
    try:
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join(timeout=1)
        assert not any(reader.is_alive() for reader in readers)
        assert reader_errors == []
        assert responses == {path: 200 for path in paths}
    finally:
        release_write.set()
        for reader in readers:
            reader.join(timeout=6)
        writer_thread.join(timeout=6)
        writer.close()

    assert not writer_thread.is_alive()
    assert writer_errors == []
    verifier = GlobalMemory(database, media_root=media_root, read_only=True)
    try:
        person = verifier.get_person(person_id)
        assert person is not None
        assert person["embedding_count"] == 2
    finally:
        verifier.close()


def test_read_requests_are_read_only_close_connections_and_do_not_run_schema_ddl(
    memory_api,
    monkeypatch,
):
    _, _, person_id, review_id = memory_api
    opened: list[GlobalMemory] = []
    schema_calls: list[Path] = []

    class TrackingGlobalMemory(GlobalMemory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def _initialize_schema(self) -> None:
            schema_calls.append(self.db_path)
            super()._initialize_schema()

    monkeypatch.setattr(global_memory_module, "GlobalMemory", TrackingGlobalMemory)
    paths = [
        "/api/memory/persons",
        f"/api/memory/persons/{person_id}",
        f"/api/memory/persons/{person_id}/gallery",
        "/api/memory/log",
        "/api/memory/search?q=person",
        "/api/identity-reviews",
        f"/api/identity-reviews/{review_id}",
    ]
    with service.app.test_client() as client:
        assert [client.get(path).status_code for path in paths] == [200] * len(paths)

    assert len(opened) == len(paths)
    assert all(memory.read_only for memory in opened)
    assert schema_calls == []
    for memory in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            memory._conn.execute("SELECT 1")


def test_persistent_sqlite_busy_is_reported_as_service_unavailable(monkeypatch):
    closed: list[bool] = []

    class LockedMemory:
        def __init__(self, *, read_only=False):
            assert read_only is True

        def list_all(self, include_inactive=False):
            raise sqlite3.OperationalError("database is locked")

        def list_pending_identity_reviews(self, *, limit, offset):
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(global_memory_module, "GlobalMemory", LockedMemory)
    service.app.config.update(TESTING=True, GLOBAL_MEMORY_READ_ONLY=False)
    with service.app.test_client() as client:
        persons = client.get("/api/memory/persons")
        reviews = client.get("/api/identity-reviews")

    assert persons.status_code == 503
    assert reviews.status_code == 503
    assert persons.get_json() == {"error": "global memory is temporarily busy"}
    assert reviews.get_json() == {"error": "global memory is temporarily busy"}
    assert closed == [True, True]


def test_non_lock_operational_error_remains_internal_server_error(monkeypatch):
    closed = []

    class BrokenMemory:
        def __init__(self, *, read_only=False):
            assert read_only is True

        def list_all(self, include_inactive=False):
            raise sqlite3.OperationalError("malformed database schema")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(global_memory_module, "GlobalMemory", BrokenMemory)
    previous_testing = service.app.config.get("TESTING")
    service.app.config.update(TESTING=False, GLOBAL_MEMORY_READ_ONLY=False)
    try:
        response = service.app.test_client().get("/api/memory/persons")
    finally:
        service.app.config["TESTING"] = previous_testing

    assert response.status_code == 500
    assert response.get_json() == {"error": "database operation failed"}
    assert closed == [True]


def test_writable_service_startup_precedes_empty_read_only_apis(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "fresh" / "global_memory.db"
    media_root = tmp_path / "fresh" / "person_db"
    monkeypatch.setenv("FORENSICS_MEMORY_DB", str(database))
    monkeypatch.setenv("PERSON_CREATION_MEDIA_ROOT", str(media_root))
    service.app.config.update(TESTING=True, GLOBAL_MEMORY_READ_ONLY=False)
    assert not database.exists()

    service.initialize_global_memory()

    assert database.is_file()
    with service.app.test_client() as client:
        persons = client.get("/api/memory/persons")
        reviews = client.get("/api/identity-reviews?limit=100&offset=0")
    assert persons.status_code == 200
    assert persons.get_json() == []
    assert reviews.status_code == 200
    assert reviews.get_json()["reviews"] == []
    assert reviews.get_json()["pending_count"] == 0
    assert not media_root.exists()

    with GlobalMemory(database, media_root=media_root) as memory:
        assert memory.register(_profile((1.0, 0.0, 0.0))) == "person_001"


def test_two_simultaneous_first_use_callers_initialize_schema_once(tmp_path, monkeypatch):
    database = tmp_path / "simultaneous.db"
    media_root = tmp_path / "media"
    media_root.mkdir()
    original_initialize = GlobalMemory._initialize_schema
    calls: list[Path] = []
    calls_lock = threading.Lock()

    def counted_initialize(memory: GlobalMemory) -> None:
        with calls_lock:
            calls.append(memory.db_path.resolve())
        original_initialize(memory)

    monkeypatch.setattr(GlobalMemory, "_initialize_schema", counted_initialize)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def first_use() -> None:
        memory = None
        try:
            barrier.wait(timeout=2)
            memory = GlobalMemory(database, media_root=media_root)
            assert memory.list_all() == []
        except BaseException as exc:
            errors.append(exc)
        finally:
            if memory is not None:
                memory.close()

    callers = [threading.Thread(target=first_use) for _ in range(2)]
    for caller in callers:
        caller.start()
    barrier.wait(timeout=2)
    for caller in callers:
        caller.join(timeout=8)

    assert not any(caller.is_alive() for caller in callers)
    assert errors == []
    assert calls == [database.resolve()]


def test_schema_initialization_is_once_per_database_path(tmp_path, monkeypatch):
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    media_root = tmp_path / "media"
    media_root.mkdir()
    original_initialize = GlobalMemory._initialize_schema
    calls: list[Path] = []

    def counted_initialize(memory: GlobalMemory) -> None:
        calls.append(memory.db_path.resolve())
        original_initialize(memory)

    monkeypatch.setattr(GlobalMemory, "_initialize_schema", counted_initialize)
    for database in (first_path, first_path, second_path, second_path):
        memory = GlobalMemory(database, media_root=media_root)
        memory.close()

    assert calls == [first_path.resolve(), second_path.resolve()]
