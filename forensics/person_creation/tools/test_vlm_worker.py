"""Test the async VLM worker (forensics/person_creation/vlm_worker.py) end to end.

No GPU/InternVL/network required -- the clothing describer singleton is stubbed with
a fake `.describe()` so this exercises the real queue/claim/retry/appearance-write
logic against a real Postgres GlobalMemory database (see global_memory/config.py's
GLOBAL_MEMORY_* env vars / docker-compose.yml for how to point this at one). Each
scenario truncates the relevant tables first for isolation -- there's no more
"fresh SQLite tempfile per scenario" now that GlobalMemory targets one shared DB.

Usage:
    python -m forensics.person_creation.tools.test_vlm_worker
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import psycopg
import psycopg_pool


def _require_postgres_or_skip() -> bool:
    """Returns True if reachable; prints instructions and returns False otherwise."""
    from forensics.global_memory.store import GlobalMemory

    try:
        gm = GlobalMemory(connect_timeout=2.0)
        gm.close()
        return True
    except (psycopg.OperationalError, psycopg_pool.PoolTimeout) as exc:
        print(
            "Postgres not reachable -- set GLOBAL_MEMORY_* env vars or run "
            f"`docker-compose up` first (see .env.example). ({exc})"
        )
        return False


def _reset_db() -> None:
    from forensics.global_memory.store import GlobalMemory

    gm = GlobalMemory()
    with gm._pool.connection() as conn:
        conn.execute(
            "TRUNCATE persons, appearances, recognition_log, person_gallery, "
            "clothing_jobs, segments, camera_events RESTART IDENTITY CASCADE; "
            "UPDATE counters SET value = 0 WHERE key = 'person_count';"
        )
    gm.close()


def _make_crop(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, f"crop_{np.random.randint(1_000_000)}.jpg")
    cv2.imwrite(path, np.zeros((32, 32, 3), dtype="uint8"))
    return path


def test_success_path(failures: list[str]) -> None:
    print("=== Scenario 1: success path (enqueue -> claim -> describe -> appearance + done) ===")
    _reset_db()
    tmp_dir = tempfile.mkdtemp()

    from forensics.global_memory.store import GlobalMemory
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.vlm_worker import VLMWorker

    describer = get_clothing_describer()
    describer._model = object()
    describer.describe = lambda crops: (
        "a person wearing a red shirt and blue jeans",
        {"top": "red shirt", "bottom": "blue jeans", "shoes": "sneakers", "full": "red shirt, blue jeans, sneakers"},
    )

    crop_path = _make_crop(tmp_dir)
    gm = GlobalMemory()
    person_id = gm.register({
        "face_embedding": np.random.rand(512).tolist(),
        "face_crops": ["f1.jpg"],
        "appearance": {"date": "2026-07-20", "top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": "unknown"},
        "video_sources": [],
    })
    job_id = gm.insert_clothing_job(person_id=person_id, segment_id="seg-success", crop_path=crop_path)

    worker = VLMWorker(poll_interval_seconds=0.05)
    did_work = worker.run_once()
    job = gm.get_clothing_job(job_id)
    with gm._pool.connection() as conn:
        appearance = conn.execute(
            "SELECT * FROM appearances WHERE segment_id = %s", ("seg-success",)
        ).fetchone()

    print(f"  did_work={did_work} job_status={job['status']} appearance_top={appearance['top'] if appearance else None}")

    if not did_work:
        failures.append("Scenario 1: run_once() reported no work done")
    if job["status"] != "done":
        failures.append(f"Scenario 1: expected job status 'done', got {job['status']!r}")
    if appearance is None or appearance["top"] != "red shirt":
        failures.append("Scenario 1: appearance row missing or has wrong clothing description")
    if worker.jobs_succeeded != 1 or worker.jobs_failed != 0:
        failures.append(f"Scenario 1: worker counters wrong: succeeded={worker.jobs_succeeded} failed={worker.jobs_failed}")

    if worker.run_once() is not False:
        failures.append("Scenario 1: expected no more pending work, but run_once() found something")

    gm.close()
    print()


def test_retry_then_failed_final(failures: list[str]) -> None:
    print("=== Scenario 2: retry -> failed_retryable -> failed_final ===")
    _reset_db()
    tmp_dir = tempfile.mkdtemp()

    from forensics.global_memory.store import GlobalMemory
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.vlm_worker import VLMWorker

    describer = get_clothing_describer()
    describer._model = object()

    def boom(crops):
        raise RuntimeError("simulated VLM crash")

    describer.describe = boom

    crop_path = _make_crop(tmp_dir)
    gm = GlobalMemory()
    person_id = gm.register({
        "face_embedding": np.random.rand(512).tolist(),
        "face_crops": [],
        "appearance": {"date": "2026-07-20"},
        "video_sources": [],
    })
    job_id = gm.insert_clothing_job(person_id=person_id, segment_id="seg-fail", crop_path=crop_path)

    worker = VLMWorker(poll_interval_seconds=0.05, max_attempts=3, backoff_seconds=(0.0, 0.0, 0.0))

    statuses = []
    for _ in range(3):
        worker.run_once()
        statuses.append(gm.get_clothing_job(job_id)["status"])
    print(f"  status sequence: {statuses}")

    if statuses != ["failed_retryable", "failed_retryable", "failed_final"]:
        failures.append(f"Scenario 2: unexpected status sequence {statuses}")

    final = gm.get_clothing_job(job_id)
    if final["attempts"] != 3:
        failures.append(f"Scenario 2: expected 3 attempts, got {final['attempts']}")
    if worker.jobs_failed != 3 or worker.jobs_succeeded != 0:
        failures.append(f"Scenario 2: worker counters wrong: succeeded={worker.jobs_succeeded} failed={worker.jobs_failed}")

    # A failed_final job must never be reclaimed.
    if worker.run_once() is not False:
        failures.append("Scenario 2: a failed_final job was reclaimed by run_once()")

    gm.close()
    print()


def test_idempotent_enqueue_and_segment_overwrite(failures: list[str]) -> None:
    print("=== Scenario 3: idempotent enqueue + idempotent appearance overwrite ===")
    _reset_db()
    tmp_dir = tempfile.mkdtemp()

    from forensics.global_memory.store import GlobalMemory
    from forensics.person_creation.models.clothing_describer import get_clothing_describer
    from forensics.person_creation.vlm_worker import VLMWorker

    describer = get_clothing_describer()
    describer._model = object()
    calls = {"n": 0}

    def describe(crops):
        calls["n"] += 1
        return (f"description #{calls['n']}", {"top": f"top-{calls['n']}", "bottom": "jeans", "shoes": "boots", "full": f"full-{calls['n']}"})

    describer.describe = describe

    crop_path = _make_crop(tmp_dir)
    gm = GlobalMemory()
    person_id = gm.register({
        "face_embedding": np.random.rand(512).tolist(),
        "face_crops": [],
        "appearance": {"date": "2026-07-20"},
        "video_sources": [],
    })

    job_id_1 = gm.insert_clothing_job(person_id=person_id, segment_id="seg-dup", crop_path=crop_path)
    job_id_2 = gm.insert_clothing_job(person_id=person_id, segment_id="seg-dup", crop_path=crop_path)
    print(f"  first insert: {job_id_1}, duplicate insert: {job_id_2}")
    if job_id_1 is None or job_id_2 is not None:
        failures.append("Scenario 3: duplicate (person_id, segment_id) clothing_jobs insert was not rejected")

    worker = VLMWorker(poll_interval_seconds=0.05)
    worker.run_once()
    worker.run_once()
    if calls["n"] != 1:
        failures.append(f"Scenario 3: expected exactly 1 VLM invocation, got {calls['n']}")

    with gm._pool.connection() as conn:
        rows = conn.execute("SELECT * FROM appearances WHERE segment_id = %s", ("seg-dup",)).fetchall()
    if len(rows) != 1:
        failures.append(f"Scenario 3: expected exactly 1 appearance row for the segment, got {len(rows)}")

    gm.close()
    print()


def main() -> int:
    print("=== VLM worker tests (stubbed ClothingDescriber, real Postgres GlobalMemory) ===\n")
    if not _require_postgres_or_skip():
        return 1
    failures: list[str] = []
    test_success_path(failures)
    test_retry_then_failed_final(failures)
    test_idempotent_enqueue_and_segment_overwrite(failures)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All VLM worker scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
