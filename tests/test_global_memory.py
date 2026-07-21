from __future__ import annotations

import threading
from datetime import date, timedelta

import numpy as np
import psycopg
import psycopg_pool
import pytest

from forensics.global_memory import GlobalMemory

_EMBEDDING_DIM = 512  # must match schema_postgres.sql's persons.embedding vector(512)

_TRUNCATE_SQL = """
    TRUNCATE persons, appearances, recognition_log, person_gallery,
             clothing_jobs, segments, camera_events RESTART IDENTITY CASCADE;
    UPDATE counters SET value = 0 WHERE key = 'person_count';
"""


def _basis(index: int, dim: int = _EMBEDDING_DIM) -> list[float]:
    """A one-hot unit vector -- the 512-dim equivalent of the old toy [1,0,0]-style
    embeddings, exactly orthogonal to every other _basis(j != index) vector so the
    "these are unrelated people" tests keep the same guarantee at real dimensionality.
    """
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec.tolist()


def _unit(weights: dict[int, float], dim: int = _EMBEDDING_DIM) -> list[float]:
    """A unit vector blended from a few basis directions, e.g. _unit({0: 0.8, 1: 0.2})
    is the 512-dim equivalent of the old _unit([0.8, 0.2, 0.0]) -- "close to person 0,
    slightly toward person 1" for the same-person-different-angle tests.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for index, weight in weights.items():
        vec[index] = weight
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
    embedding = embedding or _basis(0)
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
def memory():
    try:
        gm = GlobalMemory(min_size=1, max_size=2, connect_timeout=2.0)
    except (psycopg.OperationalError, psycopg_pool.PoolTimeout) as exc:
        pytest.skip(f"Postgres not reachable (set GLOBAL_MEMORY_* env vars / run docker-compose up): {exc}")
        return

    with gm._pool.connection() as conn:
        conn.execute(_TRUNCATE_SQL)

    try:
        yield gm
    finally:
        gm.close()


def _counts(memory: GlobalMemory) -> tuple[int, int]:
    with memory._pool.connection() as conn:
        persons = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        appearances = conn.execute("SELECT COUNT(*) AS n FROM appearances").fetchone()["n"]
        return persons, appearances


def test_register_new_person(memory):
    assigned = memory.register(_profile())

    assert assigned == "person_001"
    assert _counts(memory) == (1, 1)
    person = memory.get_person("person_001")
    assert person is not None
    assert person["name"] == "Person 001"
    assert np.linalg.norm(np.asarray(person["embedding"], dtype=np.float32)) == pytest.approx(1.0, abs=1e-5)


def test_register_same_person_new_day(memory):
    emb_a = _basis(0)
    emb_b = _unit({0: 0.8, 1: 0.2})
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
    emb_a = _basis(0)
    emb_b = _unit({0: 0.8, 1: 0.2})
    assigned = memory.register(_profile(embedding=emb_a, face_count=10, day="2026-05-13"))
    memory.register(_profile(embedding=emb_b, face_count=2, day="2026-05-14"))

    person = memory.get_person(assigned)
    merged = np.asarray(person["embedding"], dtype=np.float32)
    assert person["embedding_count"] == 12
    assert float(merged @ np.asarray(emb_a, dtype=np.float32)) > float(merged @ np.asarray(emb_b, dtype=np.float32))


def test_query_by_face_finds_match(memory):
    emb = _basis(0)
    assigned = memory.register(_profile(embedding=emb))

    results = memory.query_by_face(emb, threshold=0.9)

    assert results[0]["person_id"] == assigned
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_query_by_face_rejects_below_threshold(memory):
    memory.register(_profile(embedding=_basis(0)))

    assert memory.query_by_face(_basis(1), threshold=0.6) == []


def test_query_by_face_ranks_correctly(memory):
    memory.register(_profile("cluster_a", "Cluster A", _basis(0)))
    second = memory.register(_profile("cluster_b", "Cluster B", _basis(1)))

    results = memory.query_by_face(_basis(1), threshold=0.0)

    assert second == "person_002"
    assert results[0]["person_id"] == "person_002"


def test_query_by_face_staleness_flag(memory):
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    emb = _basis(0)
    memory.register(_profile(embedding=emb, day=yesterday))
    assert memory.query_by_face(emb, threshold=0.9)[0]["appearance"]["is_stale"] is True

    memory.register(_profile(embedding=emb, day=today))
    assert memory.query_by_face(emb, threshold=0.9)[0]["appearance"]["is_stale"] is False


def test_query_by_date(memory):
    memory.register(_profile("a", "A", _basis(0), day="2026-05-13"))
    memory.register(_profile("b", "B", _basis(1), day="2026-05-13"))
    memory.register(_profile("c", "C", _basis(2), day="2026-05-14"))

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
    memory.register(_profile("p1", "P1", _basis(0)))
    memory.register(_profile("p2", "P2", _basis(1)))
    memory.register(_profile("p3", "P3", _basis(2)))

    results = memory.list_all()

    assert len(results) == 3
    assert [r["person_id"] for r in results] == ["person_001", "person_002", "person_003"]
    assert "embedding" not in results[0]


def test_empty_store_query(memory):
    assert memory.query_by_face(_basis(0)) == []


def test_thread_safety(memory):
    profiles = [
        _profile("p1", "P1", _basis(0)),
        _profile("p2", "P2", _basis(1)),
        _profile("p3", "P3", _basis(2)),
    ]

    threads = [threading.Thread(target=memory.register, args=(profile,)) for profile in profiles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(memory.list_all()) == 3


def test_register_logs_new_and_recognized_events(memory):
    emb = _basis(0)
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
    first = memory.register(_profile("person_malek_cluster_0", "person_malek_cluster_0", _basis(0)))
    second = memory.register(_profile("person_malek_cluster_1", "person_malek_cluster_1", _basis(1)))

    assert first == "person_001"
    assert second == "person_002"
    assert memory.get_person(first)["name"] == "Person 001"
    assert memory.get_person(second)["name"] == "Person 002"
