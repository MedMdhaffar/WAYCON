"""End-to-end test of realtime_main.RealtimeProcessor.run_forever(): a synthetic
frame source (no GStreamer/gi) feeds presence-gated ingestion, which closes
segments that flow through the queue into process_segment() and the (stubbed)
graph, with real segment-status bookkeeping in Postgres. This is the one level up
from test_segment_pipeline.py's test_segment_succeeds/fails, which drive
process_segment() directly -- this test instead verifies start()/run_forever()'s
wiring (ingestion -> queue -> consumer loop) actually works together.

Usage:
    python -m forensics.person_creation.tools.test_realtime_main
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _FakeFrame:
    frame_idx: int
    timestamp: str
    frame: Any


class ScriptedFrameSource:
    """Mimics gst_stream.GstFrameBuffer.get(timeout) without GStreamer -- same
    pattern as tools/test_presence_segmentation.py's ScriptedFrameSource.
    """

    def __init__(self, present_fn, frame_interval: float = 0.02, total_seconds: float = 3.0) -> None:
        self.present_fn = present_fn
        self.frame_interval = max(0.001, float(frame_interval))
        self.total_seconds = max(0.01, float(total_seconds))
        self._start = time.monotonic()
        self._idx = 0
        self._last_emit = 0.0

    def get(self, timeout: float = 0.5):
        elapsed = time.monotonic() - self._start
        if elapsed >= self.total_seconds:
            time.sleep(min(timeout, 0.02))
            return None
        remaining = self.frame_interval - (elapsed - self._last_emit)
        if remaining > 0:
            time.sleep(remaining)
        self._last_emit = time.monotonic() - self._start
        idx = self._idx
        self._idx += 1
        return _FakeFrame(frame_idx=idx, timestamp=str(self._last_emit), frame=self._last_emit)

    def current_present(self) -> bool:
        return bool(self.present_fn(time.monotonic() - self._start))

    def stop(self) -> None:
        pass  # RealtimeProcessor.stop() calls buffer.stop(); this fake has nothing to release


def _require_postgres_or_skip() -> bool:
    import psycopg
    import psycopg_pool
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


def test_run_forever_processes_segments(failures: list[str]) -> None:
    print("=== Scenario: run_forever() ingests, segments, and processes end to end ===")
    _reset_db()

    from forensics.global_memory.store import GlobalMemory
    from forensics.person_creation.realtime_main import RealtimeProcessor

    # Present continuously from t=0.1s to t=2.5s -- long enough for open debounce,
    # short segment caps (1s) to force >=2 rotations, and a close debounce near the end.
    source = ScriptedFrameSource(lambda t: 0.1 <= t < 2.5, frame_interval=0.02, total_seconds=3.0)

    class _FakeGraph:
        def stream(self, initial_state, stream_mode="updates"):
            yield {"finalize": {"per_cluster_profiles": {}, "profile": {}}}

    processor = RealtimeProcessor(
        "rtsp://user:pass@example.invalid/stream",
        camera_id="test-cam",
        process_every_n=1,
        sample_interval_seconds=0.05,
        open_debounce_count=2,
        close_debounce_seconds=0.3,
        max_segment_seconds=1.0,
        frame_source_factory=lambda: source,
        graph_factory=lambda: _FakeGraph(),
        gm_factory=GlobalMemory,
        detect_person_fn=lambda frame: source.current_present(),
    )
    # start() normally calls load_models() -- stub it out, no GPU/model needed
    # since detect_person_fn is injected above and process_segment doesn't touch
    # person_detector directly.
    processor._reid_result = {"reid_config": {}, "reid_available": False, "reid_unavailable_reason": "stubbed"}
    orig_start = RealtimeProcessor.start

    def fake_start(self):
        self._buffer = self._frame_source_factory()
        from forensics.person_creation.presence_segmentation import PresenceGatedIngestion

        self._ingestion = PresenceGatedIngestion(
            frame_source=self._buffer,
            detect_person_fn=self._detect_person_fn,
            sample_interval_seconds=self.sample_interval_seconds,
            open_debounce_count=self.open_debounce_count,
            close_debounce_seconds=self.close_debounce_seconds,
            max_segment_seconds=self.max_segment_seconds,
            codec=self.codec,
            on_segment_state_change=self._on_segment_state_change,
        )
        self._ingestion.start()

    processor.start = fake_start.__get__(processor, RealtimeProcessor)

    # run_forever() is meant to run forever (until externally stopped) -- it will
    # never exit on its own, that's not a bug. Wait for the scripted timeline to
    # finish producing segments, then stop it proactively, same as main() does on
    # SIGINT/SIGTERM.
    thread = threading.Thread(target=processor.run_forever, daemon=True)
    thread.start()

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and processor.segments_processed < 3:
        time.sleep(0.1)

    processor.stop()
    thread.join(timeout=3.0)
    if thread.is_alive():
        failures.append("Scenario: run_forever() did not stop cleanly after stop() was called")

    print(f"  segments_processed={processor.segments_processed} segments_failed={processor.segments_failed}")
    if processor.segments_processed < 1:
        failures.append(f"Scenario: expected at least 1 segment processed, got {processor.segments_processed}")
    if processor.segments_failed != 0:
        failures.append(f"Scenario: expected 0 failed segments, got {processor.segments_failed}")

    gm = GlobalMemory()
    segments = gm.list_segments(status="SUCCEEDED", limit=50)
    print(f"  Postgres segments with status=SUCCEEDED: {len(segments)}")
    for seg in segments:
        print(f"    {seg['segment_id']} seq={seg['seq_num']} incomplete={seg['segment_incomplete']}")
    if len(segments) != processor.segments_processed:
        failures.append(
            f"Scenario: processor counted {processor.segments_processed} but Postgres has "
            f"{len(segments)} SUCCEEDED rows"
        )
    gm.close()
    print()


def main() -> int:
    print("=== realtime_main.RealtimeProcessor end-to-end wiring test ===\n")
    if not _require_postgres_or_skip():
        return 1
    failures: list[str] = []
    test_run_forever_processes_segments(failures)

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All realtime_main scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
