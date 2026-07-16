from __future__ import annotations

import sqlite3
import threading
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
from forensics.global_memory.store import ReadOnlyGlobalMemoryError


def _unit(values) -> list[float]:
    vec = np.asarray(values, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _profile(
    person_id: str = "person_malek_cluster_0",
    name: str = "person_malek_cluster_0",
    embedding: list[float] | None = None,
    day: str = "2026-05-13",
    face_count: int = 3,
    cameras: list[str] | None = None,
    top: str = "white shirt",
):
    embedding = embedding or _unit([1.0, 0.0, 0.0])
    return {
        "id": person_id,
        "name": name,
        "face_embedding": embedding,
        "face_crops": [f"face_{i}.jpg" for i in range(face_count)],
        "appearance": {
            "date": day,
            "top": top,
            "bottom": "black pants",
            "shoes": "white sneakers",
            "full": f"{top}, black pants, white sneakers",
        },
        "appearance_signals": {
            "color": {
                "top": "white",
                "bottom": "black",
            }
        },
        "best_body_crops": ["body_1.jpg", "body_2.jpg"],
        "video_sources": ["video.mp4"],
        "cameras": cameras or [],
    }


@pytest.fixture
def memory(tmp_path):
    gm = GlobalMemory(str(tmp_path / "memory.db"))
    try:
        yield gm
    finally:
        gm.close()


def _counts(memory: GlobalMemory) -> tuple[int, int]:
    conn = sqlite3.connect(str(memory.db_path))
    try:
        persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        appearances = conn.execute("SELECT COUNT(*) FROM appearances").fetchone()[0]
        return persons, appearances
    finally:
        conn.close()


def test_register_new_person(memory):
    assigned = memory.register(_profile())

    assert assigned == "person_001"
    assert _counts(memory) == (1, 1)
    person = memory.get_person("person_001")
    assert person is not None
    assert person["name"] == "Person 001"
    assert np.linalg.norm(np.asarray(person["embedding"], dtype=np.float32)) == pytest.approx(1.0, abs=1e-5)


def test_register_same_person_new_day(memory):
    emb_a = _unit([1.0, 0.0, 0.0])
    emb_b = _unit([0.8, 0.2, 0.0])
    first = memory.register(_profile(embedding=emb_a, day="2026-05-13"))
    before = np.asarray(memory.get_person(first)["embedding"], dtype=np.float32)

    second = memory.register(_profile(embedding=emb_b, day="2026-05-14"))

    assert second == first
    assert _counts(memory) == (1, 2)
    after = np.asarray(memory.get_person(first)["embedding"], dtype=np.float32)
    assert not np.allclose(after, before)
    assert not np.allclose(after, np.asarray(emb_b, dtype=np.float32))


def test_register_same_person_same_day(memory):
    assigned = memory.register(_profile(top="white shirt"))
    memory.register(_profile(top="blue jacket"))

    assert _counts(memory) == (1, 1)
    assert memory.get_person(assigned)["latest_appearance"]["top"] == "blue jacket"


def test_embedding_averaging_is_weighted(memory):
    emb_a = _unit([1.0, 0.0, 0.0])
    emb_b = _unit([0.8, 0.2, 0.0])
    assigned = memory.register(_profile(embedding=emb_a, face_count=10, day="2026-05-13"))
    memory.register(_profile(embedding=emb_b, face_count=2, day="2026-05-14"))

    person = memory.get_person(assigned)
    merged = np.asarray(person["embedding"], dtype=np.float32)
    assert person["embedding_count"] == 12
    assert float(merged @ np.asarray(emb_a, dtype=np.float32)) > float(merged @ np.asarray(emb_b, dtype=np.float32))


def test_query_by_face_finds_match(memory):
    emb = _unit([1.0, 0.0, 0.0])
    assigned = memory.register(_profile(embedding=emb))

    results = memory.query_by_face(emb, threshold=0.9)

    assert results[0]["person_id"] == assigned
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_query_by_face_rejects_below_threshold(memory):
    memory.register(_profile(embedding=_unit([1.0, 0.0, 0.0])))

    assert memory.query_by_face(_unit([0.0, 1.0, 0.0]), threshold=0.6) == []


def test_query_by_face_ranks_correctly(memory):
    memory.register(_profile("cluster_a", "Cluster A", _unit([1.0, 0.0, 0.0])))
    second = memory.register(_profile("cluster_b", "Cluster B", _unit([0.0, 1.0, 0.0])))

    results = memory.query_by_face(_unit([0.0, 1.0, 0.0]), threshold=0.0)

    assert second == "person_002"
    assert results[0]["person_id"] == "person_002"


def test_query_by_face_staleness_flag(memory):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    emb = _unit([1.0, 0.0, 0.0])
    memory.register(_profile(embedding=emb, day=yesterday))
    assert memory.query_by_face(emb, threshold=0.9)[0]["appearance"]["is_stale"] is True

    memory.register(_profile(embedding=emb, day=today))
    assert memory.query_by_face(emb, threshold=0.9)[0]["appearance"]["is_stale"] is False


def test_query_by_date(memory):
    memory.register(_profile("a", "A", _unit([1.0, 0.0, 0.0]), day="2026-05-13"))
    memory.register(_profile("b", "B", _unit([0.0, 1.0, 0.0]), day="2026-05-13"))
    memory.register(_profile("c", "C", _unit([0.0, 0.0, 1.0]), day="2026-05-14"))

    assert len(memory.query_by_date("2026-05-13")) == 2


def test_query_by_camera(memory):
    memory.register(_profile(cameras=["103"]))

    assert len(memory.query_by_camera("103")) == 1
    assert memory.query_by_camera("104") == []


def test_query_by_camera_empty(memory):
    memory.register(_profile(cameras=[]))

    assert memory.query_by_camera("103") == []


def test_get_person(memory):
    assigned = memory.register(_profile())

    person = memory.get_person(assigned)

    assert person is not None
    assert isinstance(person["embedding"], list)
    assert isinstance(person["embedding"][0], float)


def test_get_person_unknown(memory):
    assert memory.get_person("nobody") is None


def test_list_all(memory):
    memory.register(_profile("p1", "P1", _unit([1.0, 0.0, 0.0])))
    memory.register(_profile("p2", "P2", _unit([0.0, 1.0, 0.0])))
    memory.register(_profile("p3", "P3", _unit([0.0, 0.0, 1.0])))

    results = memory.list_all()

    assert len(results) == 3
    assert [r["person_id"] for r in results] == ["person_001", "person_002", "person_003"]
    assert "embedding" not in results[0]


def test_empty_store_query(memory):
    assert memory.query_by_face(_unit([1.0, 0.0, 0.0])) == []


def test_thread_safety(memory):
    profiles = [
        _profile("p1", "P1", _unit([1.0, 0.0, 0.0])),
        _profile("p2", "P2", _unit([0.0, 1.0, 0.0])),
        _profile("p3", "P3", _unit([0.0, 0.0, 1.0])),
    ]

    threads = [threading.Thread(target=memory.register, args=(profile,)) for profile in profiles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(memory.list_all()) == 3


def _database_contents(path: Path) -> dict[str, list[tuple]]:
    connection = sqlite3.connect(str(path))
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: connection.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid'
            ).fetchall()
            for table in tables
        }
    finally:
        connection.close()


def test_read_only_query_preserves_complete_database_contents(tmp_path):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(str(database))
    embedding = _unit([1.0, 0.0, 0.0])
    writer.register(_profile(embedding=embedding))
    writer.close()
    before = _database_contents(database)
    bytes_before = database.read_bytes()

    reader = GlobalMemory(str(database), read_only=True)
    try:
        assert reader.query_by_face(embedding)[0]["person_id"] == "person_001"
    finally:
        reader.close()

    assert _database_contents(database) == before
    assert database.read_bytes() == bytes_before


def test_read_only_initialization_runs_no_schema_or_migration_writes(tmp_path):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(str(database))
    writer.close()
    before = _database_contents(database)
    bytes_before = database.read_bytes()

    reader = GlobalMemory(str(database), read_only=True)
    reader.close()

    assert _database_contents(database) == before
    assert database.read_bytes() == bytes_before


def test_missing_read_only_database_creates_nothing(tmp_path):
    database = tmp_path / "absent" / "memory.db"

    with pytest.raises(sqlite3.OperationalError):
        GlobalMemory(str(database), read_only=True)

    assert not database.parent.exists()
    assert not database.exists()


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("register", (_profile(),)),
        ("rename_person", ("person_001", "Renamed")),
        ("update_crop_paths", ("person_001", _profile())),
        ("set_profile_image", ("person_001", "face.jpg")),
        ("update_gallery", ("person_001", _profile())),
    ],
)
def test_all_public_mutations_are_rejected_in_read_only_mode(
    tmp_path,
    method_name,
    args,
):
    database = tmp_path / "memory.db"
    writer = GlobalMemory(str(database))
    writer.register(_profile())
    writer.close()
    before = _database_contents(database)

    reader = GlobalMemory(str(database), read_only=True)
    try:
        with pytest.raises(ReadOnlyGlobalMemoryError, match="read-only mode"):
            getattr(reader, method_name)(*args)
    finally:
        reader.close()

    assert _database_contents(database) == before


def test_register_logs_new_and_recognized_events(memory):
    emb = _unit([1.0, 0.0, 0.0])
    assigned = memory.register(_profile(embedding=emb, face_count=4, day="2026-05-13"))
    memory.register(_profile(embedding=emb, face_count=2, day="2026-05-14"))

    history = list(reversed(memory.get_recognition_history(assigned)))

    assert [event["event_type"] for event in history] == ["new_enrollment", "recognized"]
    assert history[0]["embedding_count_before"] is None
    assert history[0]["embedding_count_after"] == 4
    assert history[1]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert history[1]["embedding_count_before"] == 4
    assert history[1]["embedding_count_after"] == 6


def test_auto_ids_ignore_cluster_names(memory):
    first = memory.register(_profile("person_malek_cluster_0", "person_malek_cluster_0", _unit([1.0, 0.0, 0.0])))
    second = memory.register(_profile("person_malek_cluster_1", "person_malek_cluster_1", _unit([0.0, 1.0, 0.0])))

    assert first == "person_001"
    assert second == "person_002"
    assert memory.get_person(first)["name"] == "Person 001"
    assert memory.get_person(second)["name"] == "Person 002"
