"""Test the segment reliability state machine (realtime_main.RealtimeProcessor)
and the /api/segments* monitoring endpoints (service.py), end to end -- with
model loading, camera capture, and the LangGraph run itself all stubbed out (no
GPU/camera/InternVL/YOLO/GStreamer needed).

Camera ingestion and segment processing moved out of service.py into
realtime_main.py (service.py is now a monitoring/query API plus an offline
video-file job launcher -- see both modules' docstrings). This exercises
RealtimeProcessor.process_segment() (segment PROCESSING/SUCCEEDED/FAILED_*
bookkeeping around a graph run) via its graph_factory/gm_factory test seams --
notably NOT its frame_source_factory, so gst_stream.py's `gi` (PyGObject)
dependency is never imported -- and the real Flask routes (list_segments,
segment_detail, list_clothing_jobs, person_pipeline_status) against a real
Postgres GlobalMemory database (see global_memory/config.py's GLOBAL_MEMORY_*
env vars / docker-compose.yml for how to point this at one).

Usage:
    python -m forensics.person_creation.tools.test_segment_pipeline
"""

from __future__ import annotations

import sys

import numpy as np
import psycopg
import psycopg_pool


def _require_postgres_or_skip() -> bool:
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


def _make_segment_batch(segment_id: str = "seg-1"):
    from forensics.person_creation.presence_segmentation import SegmentBatch, utc_now_iso

    segment = SegmentBatch(
        segment_id=segment_id,
        seq_num=1,
        codec="h264",
        segment_start_ts=utc_now_iso(),
    )
    segment.add_frame(np.zeros((4, 4, 3), dtype="uint8"), utc_now_iso())
    segment.close(reason="duration_elapsed", incomplete=False)
    return segment


def _make_processor(*, graph_events=None, graph=None):
    """A RealtimeProcessor wired to a fake graph, hitting the real (Postgres)
    GlobalMemory. Never touches gst_stream.py -- process_segment() doesn't call
    start()/the frame_source_factory at all.
    """
    from forensics.global_memory.store import GlobalMemory
    from forensics.person_creation.realtime_main import RealtimeProcessor

    class _FakeGraph:
        def stream(self, initial_state, stream_mode="updates"):
            for event in graph_events or []:
                yield event

    return RealtimeProcessor(
        "rtsp://user:pass@example.invalid/stream",
        camera_id="test-cam",
        graph_factory=lambda: (graph or _FakeGraph()),
        gm_factory=GlobalMemory,
    )


def test_segment_succeeds(failures: list[str]) -> None:
    print("=== Scenario 1: successful segment -> PROCESSING -> SUCCEEDED ===")
    _reset_db()
    segment = _make_segment_batch("seg-succeed")
    graph_events = [{"finalize": {"per_cluster_profiles": {}, "profile": {}}}]
    processor = _make_processor(graph_events=graph_events)

    gm = processor._get_gm()
    gm.upsert_segment(
        segment_id=segment.segment_id, seq_num=segment.seq_num, codec=segment.codec,
        segment_start_ts=segment.segment_start_ts, segment_end_ts=segment.segment_end_ts,
        status="READY", segment_incomplete=segment.segment_incomplete,
    )
    processor.process_segment(segment)

    seg_row = gm.get_segment("seg-succeed")
    print(f"  segment row: {seg_row}")
    print(f"  processor.segments_processed={processor.segments_processed} segments_failed={processor.segments_failed}")
    if seg_row is None or seg_row["status"] != "SUCCEEDED":
        failures.append(f"Scenario 1: expected segment status SUCCEEDED, got {seg_row}")
    if processor.segments_processed != 1 or processor.segments_failed != 0:
        failures.append("Scenario 1: processor counters wrong")

    processor._gm_instance.close()
    print()


def test_segment_fails_and_retries(failures: list[str]) -> None:
    print("=== Scenario 2: failing segment -> FAILED_RETRYABLE, then FAILED_FINAL after max retries ===")
    _reset_db()

    class _BoomGraph:
        def stream(self, initial_state, stream_mode="updates"):
            raise RuntimeError("simulated graph failure")
            yield {}  # pragma: no cover - unreachable, keeps this a generator

    from forensics.global_memory import config as gm_config

    processor = _make_processor(graph=_BoomGraph())
    gm = processor._get_gm()
    segment = _make_segment_batch("seg-fail")
    gm.upsert_segment(
        segment_id=segment.segment_id, seq_num=segment.seq_num, codec=segment.codec,
        segment_start_ts=segment.segment_start_ts, segment_end_ts=segment.segment_end_ts,
        status="READY", segment_incomplete=segment.segment_incomplete,
    )

    observed_statuses = []
    for attempt in range(gm_config.SEGMENT_MAX_RETRIES):
        processor.process_segment(segment)  # same segment (and segment_id) every attempt
        seg_row = gm.get_segment("seg-fail")
        observed_statuses.append(seg_row["status"])
        print(f"  attempt {attempt + 1}: segment.status={seg_row['status']} retry_count={seg_row['retry_count']}")

    print(f"  status sequence: {observed_statuses}")
    print(f"  processor.segments_failed={processor.segments_failed}")
    if observed_statuses[-1] != "FAILED_FINAL":
        failures.append(f"Scenario 2: expected final status FAILED_FINAL after {gm_config.SEGMENT_MAX_RETRIES} attempts, got {observed_statuses}")
    if "FAILED_RETRYABLE" not in observed_statuses[:-1]:
        failures.append(f"Scenario 2: expected FAILED_RETRYABLE before the final attempt, got {observed_statuses}")
    if processor.segments_failed != gm_config.SEGMENT_MAX_RETRIES:
        failures.append(f"Scenario 2: expected {gm_config.SEGMENT_MAX_RETRIES} failed attempts counted, got {processor.segments_failed}")

    processor._gm_instance.close()
    print()


def test_monitoring_endpoints(failures: list[str]) -> None:
    print("=== Scenario 3: /api/segments*, /api/clothing-jobs, pipeline-status endpoints ===")
    _reset_db()

    from forensics.global_memory.store import GlobalMemory
    from forensics.global_memory import config as gm_config
    from forensics.person_creation.presence_segmentation import utc_now_iso
    import forensics.person_creation.service as service

    gm = GlobalMemory()
    person_id = gm.register({
        "face_embedding": np.random.randn(512).tolist(),
        "face_crops": [],
        "appearance": {"date": "2026-07-20"},
        "video_sources": [],
    })
    gm.upsert_segment(
        segment_id="seg-mon", seq_num=1, codec="h264",
        segment_start_ts=utc_now_iso(), segment_end_ts=utc_now_iso(),
        status=gm_config.SEGMENT_STATUS_SUCCEEDED,
    )
    job_id = gm.insert_clothing_job(person_id=person_id, segment_id="seg-mon", crop_path="/tmp/x.jpg")
    gm.close()

    client = service.app.test_client()

    resp = client.get("/api/segments")
    body = resp.get_json()
    print(f"  GET /api/segments -> {resp.status_code} {body}")
    if resp.status_code != 200 or not any(s["segment_id"] == "seg-mon" for s in body["segments"]):
        failures.append("Scenario 3: /api/segments did not list the test segment")

    resp = client.get("/api/segments/seg-mon")
    body = resp.get_json()
    print(f"  GET /api/segments/seg-mon -> {resp.status_code} {body}")
    if resp.status_code != 200 or body["segment"]["status"] != "SUCCEEDED":
        failures.append("Scenario 3: /api/segments/<id> did not return the expected segment status")
    if not any(j["job_id"] == job_id for j in body["clothing_jobs"]):
        failures.append("Scenario 3: /api/segments/<id> did not include its clothing_jobs")

    resp = client.get("/api/segments/does-not-exist")
    print(f"  GET /api/segments/does-not-exist -> {resp.status_code}")
    if resp.status_code != 404:
        failures.append(f"Scenario 3: expected 404 for unknown segment, got {resp.status_code}")

    resp = client.get("/api/clothing-jobs?status=pending")
    body = resp.get_json()
    print(f"  GET /api/clothing-jobs?status=pending -> {resp.status_code} {body}")
    if resp.status_code != 200 or not any(j["job_id"] == job_id for j in body["clothing_jobs"]):
        failures.append("Scenario 3: /api/clothing-jobs?status=pending did not list the test job")

    resp = client.get(f"/api/memory/persons/{person_id}/pipeline-status")
    body = resp.get_json()
    print(f"  GET /api/memory/persons/{person_id}/pipeline-status -> {resp.status_code} {body}")
    if resp.status_code != 200:
        failures.append("Scenario 3: pipeline-status endpoint failed")
    elif body["core_detection"]["status"] != "SUCCEEDED":
        failures.append(f"Scenario 3: expected core_detection.status SUCCEEDED, got {body['core_detection']}")
    elif body["clothing_enrichment"]["status"] != "pending":
        failures.append(f"Scenario 3: expected clothing_enrichment.status pending (not conflated with core_detection), got {body['clothing_enrichment']}")

    resp = client.get("/api/memory/persons/does-not-exist/pipeline-status")
    print(f"  GET /api/memory/persons/does-not-exist/pipeline-status -> {resp.status_code}")
    if resp.status_code != 404:
        failures.append(f"Scenario 3: expected 404 for unknown person, got {resp.status_code}")

    if getattr(service, "_gm", None) is not None:
        service._gm.close()

    print()


def main() -> int:
    print("=== Segment reliability state machine + monitoring API tests ===\n")
    if not _require_postgres_or_skip():
        return 1
    failures: list[str] = []
    test_segment_succeeds(failures)
    test_segment_fails_and_retries(failures)
    test_monitoring_endpoints(failures)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All segment/monitoring scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
