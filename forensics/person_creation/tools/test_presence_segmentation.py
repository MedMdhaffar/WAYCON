"""Test ``forensics/person_creation/presence_segmentation.py``.

Modes
-----

1. Synthetic self-test (default, no camera/model/GPU required)

   Runs deterministic scenarios covering:

   * open and close debounce;
   * duration-cap rotation;
   * reconnect -> UNKNOWN and incomplete-segment handling;
   * bounded output-queue FIFO/drop-oldest behavior;
   * end-to-end segment rotation with an intentionally stalled consumer.

       python -m forensics.person_creation.tools.test_presence_segmentation

2. Live RTSP test (``--uri``)

   Runs the real GStreamer frame source and YOLO person detector. By default, the
   script consumes and prints each closed segment:

       python -m forensics.person_creation.tools.test_presence_segmentation \
           --uri rtsp://localhost:8554/test --duration 90

3. Live queue-retention/drop verification (``--verify-queue-drop``)

   The consumer is intentionally paused. Closed segments remain in the bounded
   output queue. Once the queue is full, the oldest unconsumed segment must be
   dropped and the newest segments retained:

       python -m forensics.person_creation.tools.test_presence_segmentation \
           --uri rtsp://localhost:8554/test \
           --duration 45 \
           --segment-seconds 10 \
           --queue-size 2 \
           --verify-queue-drop

   With a 10-second segment cap and queue size 2, use at least 35-45 seconds of
   continuous presence so at least three segments close and one drop is observable.

See ``forensics/person_creation/tools/test_gst_stream.py`` for instructions for a
local RTSP source (for example, mediamtx + ffmpeg).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from forensics.person_creation.presence_segmentation import (
    PresenceGatedIngestion,
    PresenceState,
    SegmentAccumulator,
    SegmentBatch,
)


# --------------------------------------------------------------------------- helpers


def _queue_snapshot(output_queue: "queue.Queue[SegmentBatch]") -> list[dict[str, Any]]:
    """Return a thread-safe, non-destructive test snapshot of the queue.

    This intentionally uses ``queue.Queue`` internals. It belongs only in this manual
    test utility; production code should consume through ``get()`` instead.
    """

    with output_queue.mutex:
        return [segment.as_summary() for segment in list(output_queue.queue)]


def _drain_queue(output_queue: "queue.Queue[SegmentBatch]") -> list[SegmentBatch]:
    drained: list[SegmentBatch] = []
    while True:
        try:
            drained.append(output_queue.get_nowait())
        except queue.Empty:
            return drained


def _active_segment_summary(ingestion: PresenceGatedIngestion) -> dict[str, Any] | None:
    """Return compact active-segment metadata without copying frame arrays."""

    accumulator = ingestion.accumulator
    with accumulator._lock:  # test-only inspection of the accumulator's private state
        current = accumulator._current
        return None if current is None else current.as_summary()


def _print_failures(failures: list[str]) -> int:
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All synthetic scenarios passed.")
    return 0


# ------------------------------------------------------------------ synthetic mode


@dataclass
class _FakeFrame:
    frame_idx: int
    timestamp: str
    frame: Any


class ScriptedFrameSource:
    """Emit frames on a scripted timeline without GStreamer.

    ``present_fn(elapsed)`` decides whether the fake detector should report a person.
    The class mimics ``GstFrameBuffer.get(timeout)``.
    """

    def __init__(
        self,
        present_fn: Callable[[float], bool],
        frame_interval: float = 0.02,
        total_seconds: float = 2.0,
    ) -> None:
        self.present_fn = present_fn
        self.frame_interval = max(0.001, float(frame_interval))
        self.total_seconds = max(0.01, float(total_seconds))
        self._start = time.monotonic()
        self._idx = 0
        self._last_emit = 0.0

    def get(self, timeout: float = 0.5) -> _FakeFrame | None:
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
        # The payload is elapsed time; the fake detector reads presence from the source.
        return _FakeFrame(frame_idx=idx, timestamp=str(self._last_emit), frame=self._last_emit)

    def current_present(self) -> bool:
        return bool(self.present_fn(time.monotonic() - self._start))


def _collect_until(
    ingestion: PresenceGatedIngestion,
    seconds: float,
) -> list[SegmentBatch]:
    segments: list[SegmentBatch] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            segments.append(ingestion.output_queue.get(timeout=0.05))
        except queue.Empty:
            pass
    return segments


def run_synthetic() -> int:
    print("=== Synthetic presence-segmentation tests ===\n")
    failures: list[str] = []

    # ---------------------------------------------------------------- Scenario 1
    print("=== Scenario 1: open/close debounce ===")

    def present_fn_1(t: float) -> bool:
        return 0.20 <= t < 0.80

    source1 = ScriptedFrameSource(present_fn_1, total_seconds=1.5)
    events1: list[tuple[float, str]] = []
    started1 = time.monotonic()
    ingestion1 = PresenceGatedIngestion(
        frame_source=source1,
        detect_person_fn=lambda _frame: source1.current_present(),
        sample_interval_seconds=0.05,
        open_debounce_count=2,
        close_debounce_seconds=0.25,
        max_segment_seconds=10.0,
    )

    original_transition1 = ingestion1.gate._transition

    def trace_transition1(new_state: PresenceState, reason: str) -> None:
        events1.append((time.monotonic() - started1, f"{new_state.value}:{reason}"))
        original_transition1(new_state, reason)

    ingestion1.gate._transition = trace_transition1  # type: ignore[method-assign]
    ingestion1.start()
    segments1 = _collect_until(ingestion1, 1.7)
    ingestion1.stop()
    segments1.extend(_drain_queue(ingestion1.output_queue))

    for timestamp, event in events1:
        print(f"  t={timestamp:5.2f}s presence -> {event}")
    print(f"  segments={ [segment.as_summary() for segment in segments1] }\n")

    states1 = [event.split(":", maxsplit=1)[0] for _, event in events1]
    if PresenceState.PRESENT.value not in states1:
        failures.append("Scenario 1: PRESENT transition was not observed")
    if PresenceState.ABSENT.value not in states1:
        failures.append("Scenario 1: ABSENT transition after close debounce was not observed")
    if len(segments1) != 1:
        failures.append(f"Scenario 1: expected exactly one segment, got {len(segments1)}")
    elif segments1[0].segment_incomplete:
        failures.append("Scenario 1: normal presence-loss close was incorrectly marked incomplete")
    elif not str(segments1[0].close_reason).startswith("presence_absent:"):
        failures.append(
            f"Scenario 1: expected presence_absent close reason, got {segments1[0].close_reason!r}"
        )

    # ---------------------------------------------------------------- Scenario 2
    print("=== Scenario 2: duration-cap rotation ===")
    source2 = ScriptedFrameSource(lambda t: t >= 0.10, total_seconds=1.75)
    ingestion2 = PresenceGatedIngestion(
        frame_source=source2,
        detect_person_fn=lambda _frame: source2.current_present(),
        sample_interval_seconds=0.05,
        open_debounce_count=2,
        close_debounce_seconds=1.0,
        max_segment_seconds=0.5,
    )
    ingestion2.start()
    segments2 = _collect_until(ingestion2, 1.9)
    ingestion2.stop()
    segments2.extend(_drain_queue(ingestion2.output_queue))

    summaries2 = [segment.as_summary() for segment in segments2]
    print(f"  segments={summaries2}\n")

    seq2 = [segment.seq_num for segment in segments2]
    if len(segments2) < 3:
        failures.append(f"Scenario 2: expected at least three segments, got {len(segments2)}")
    if seq2 != list(range(1, len(seq2) + 1)):
        failures.append(f"Scenario 2: expected contiguous sequence numbers, got {seq2}")
    if not all(segment.frames for segment in segments2):
        failures.append("Scenario 2: every emitted segment must contain at least one frame")
    if not all(
        segment.close_reason == "duration_cap" for segment in segments2[:-1]
    ):
        failures.append("Scenario 2: all rotated segments except the final shutdown segment must use duration_cap")

    # ---------------------------------------------------------------- Scenario 3
    print("=== Scenario 3: reconnect -> UNKNOWN + incomplete segment ===")
    source3 = ScriptedFrameSource(lambda _t: True, total_seconds=1.0)
    ingestion3 = PresenceGatedIngestion(
        frame_source=source3,
        detect_person_fn=lambda _frame: source3.current_present(),
        sample_interval_seconds=0.05,
        open_debounce_count=2,
        close_debounce_seconds=5.0,
        max_segment_seconds=100.0,
    )
    ingestion3.start()

    present_deadline = time.monotonic() + 0.6
    while ingestion3.gate.state is not PresenceState.PRESENT and time.monotonic() < present_deadline:
        time.sleep(0.01)

    if ingestion3.gate.state is not PresenceState.PRESENT:
        failures.append("Scenario 3 setup: presence never reached PRESENT")
    else:
        # The frame that satisfies open debounce is evaluated before the new segment exists.
        # Wait for at least one subsequent frame so reconnect closes a non-empty segment.
        frame_deadline = time.monotonic() + 0.3
        while time.monotonic() < frame_deadline:
            active = _active_segment_summary(ingestion3)
            if active is not None and active["frame_count"] > 0:
                break
            time.sleep(0.01)

        class _Reconnecting:
            value = "reconnecting"

        ingestion3.on_connection_state_change(_Reconnecting())
        try:
            segment3 = ingestion3.output_queue.get(timeout=0.5)
        except queue.Empty:
            segment3 = None

        if ingestion3.gate.state is not PresenceState.UNKNOWN:
            failures.append(
                f"Scenario 3: expected UNKNOWN after reconnect signal, got {ingestion3.gate.state}"
            )
        if segment3 is None:
            failures.append("Scenario 3: reconnect did not force-close the open segment")
        else:
            print(f"  segment={segment3.as_summary()}")
            if not segment3.segment_incomplete:
                failures.append("Scenario 3: reconnect-closed segment was not marked incomplete")
            if not str(segment3.close_reason).startswith("connection_lost:"):
                failures.append(
                    f"Scenario 3: expected connection_lost close reason, got {segment3.close_reason!r}"
                )

    ingestion3.stop()
    _drain_queue(ingestion3.output_queue)
    print()

    # ---------------------------------------------------------------- Scenario 4
    print("=== Scenario 4: deterministic bounded queue drops oldest ===")
    output4: "queue.Queue[SegmentBatch]" = queue.Queue(maxsize=2)
    dropped4: list[int] = []
    accumulator4 = SegmentAccumulator(
        output_queue=output4,
        max_segment_seconds=10.0,
        on_segment_dropped=lambda segment: dropped4.append(segment.seq_num),
    )

    for seq_num in range(1, 5):
        segment = SegmentBatch(
            segment_id=f"synthetic-{seq_num}",
            seq_num=seq_num,
            codec="h264",
            segment_start_ts=f"start-{seq_num}",
        )
        segment.add_frame(frame=f"frame-{seq_num}", timestamp=f"ts-{seq_num}")
        segment.close(reason="test")
        accumulator4._push(segment)  # test the real production drop-oldest implementation

    retained4 = _drain_queue(output4)
    retained_seq4 = [segment.seq_num for segment in retained4]
    print(f"  dropped={dropped4} retained={retained_seq4}\n")

    if dropped4 != [1, 2]:
        failures.append(f"Scenario 4: expected dropped [1, 2], got {dropped4}")
    if retained_seq4 != [3, 4]:
        failures.append(f"Scenario 4: expected retained FIFO [3, 4], got {retained_seq4}")

    # ---------------------------------------------------------------- Scenario 5
    print("=== Scenario 5: end-to-end queue overflow with stalled consumer ===")
    output5: "queue.Queue[SegmentBatch]" = queue.Queue(maxsize=2)
    dropped5: list[int] = []
    source5 = ScriptedFrameSource(lambda _t: True, frame_interval=0.02, total_seconds=2.25)
    ingestion5 = PresenceGatedIngestion(
        frame_source=source5,
        detect_person_fn=lambda _frame: source5.current_present(),
        output_queue=output5,
        sample_interval_seconds=0.05,
        open_debounce_count=2,
        close_debounce_seconds=5.0,
        max_segment_seconds=0.5,
        on_segment_dropped=lambda segment: dropped5.append(segment.seq_num),
    )
    ingestion5.start()
    # Deliberately do not call output_queue.get(): this simulates a stalled Tier 1 consumer.
    time.sleep(2.35)
    ingestion5.stop()  # force-closes the final active segment into the same bounded queue

    retained5 = _drain_queue(output5)
    retained_seq5 = [segment.seq_num for segment in retained5]
    all_seq5 = dropped5 + retained_seq5
    expected_retained5 = all_seq5[-2:] if len(all_seq5) >= 2 else all_seq5
    expected_dropped5 = all_seq5[:-2] if len(all_seq5) > 2 else []
    print(
        f"  produced={all_seq5} dropped={dropped5} retained={retained_seq5} "
        f"frames_seen={ingestion5.frames_seen}\n"
    )

    if len(all_seq5) < 3:
        failures.append(
            f"Scenario 5: expected at least three closed segments to overflow queue, got {all_seq5}"
        )
    if all_seq5 != list(range(1, len(all_seq5) + 1)):
        failures.append(f"Scenario 5: produced sequence is not contiguous: {all_seq5}")
    if dropped5 != expected_dropped5:
        failures.append(
            f"Scenario 5: expected dropped {expected_dropped5}, got {dropped5}"
        )
    if retained_seq5 != expected_retained5:
        failures.append(
            f"Scenario 5: expected newest two retained {expected_retained5}, got {retained_seq5}"
        )
    if len(retained_seq5) > output5.maxsize:
        failures.append(
            f"Scenario 5: queue retained {len(retained_seq5)} segments with maxsize={output5.maxsize}"
        )

    return _print_failures(failures)


# ------------------------------------------------------------------------ live mode


def run_live(
    uri: str,
    duration: float,
    codec: str,
    decoder: str | None,
    queue_size: int,
    segment_seconds: float,
    verify_queue_drop: bool,
    inspect_every: float,
) -> int:
    from forensics.person_creation.gst_stream import GstFrameBuffer
    from forensics.person_creation.live_stream import mask_camera_uri
    from forensics.person_creation.models.person_detector import get_person_detector

    queue_size = max(1, int(queue_size))
    segment_seconds = max(0.5, float(segment_seconds))
    inspect_every = max(0.2, float(inspect_every))

    print("Loading YOLO person detector...")
    detector = get_person_detector()
    detector.load("yolo26m.pt")

    def detect_person_fn(frame: Any) -> bool:
        return len(detector.detect(frame)) > 0

    print(f"Connecting to {mask_camera_uri(uri)}")
    buffer = GstFrameBuffer(uri, codec=codec, decoder=decoder)
    output_queue: "queue.Queue[SegmentBatch]" = queue.Queue(maxsize=queue_size)
    dropped_summaries: list[dict[str, Any]] = []
    closed_summaries: list[dict[str, Any]] = []
    event_lock = threading.Lock()

    def on_segment_dropped(segment: SegmentBatch) -> None:
        summary = segment.as_summary()
        with event_lock:
            dropped_summaries.append(summary)
        print(
            f"[{time.strftime('%H:%M:%S')}] SEGMENT DROPPED (oldest unconsumed): {summary}",
            flush=True,
        )

    ingestion = PresenceGatedIngestion(
        frame_source=buffer,
        detect_person_fn=detect_person_fn,
        output_queue=output_queue,
        sample_interval_seconds=0.3,
        open_debounce_count=2,
        close_debounce_seconds=2.5,
        max_segment_seconds=segment_seconds,
        codec=codec,
        on_segment_dropped=on_segment_dropped,
    )

    original_transition = ingestion.gate._transition

    def traced_transition(new_state: PresenceState, reason: str) -> None:
        print(
            f"[{time.strftime('%H:%M:%S')}] presence -> {new_state.value} ({reason})",
            flush=True,
        )
        original_transition(new_state, reason)

    ingestion.gate._transition = traced_transition  # type: ignore[method-assign]
    buffer._on_state_change = ingestion.on_connection_state_change

    # In verification mode, trace each close without consuming the queue. The production
    # _push() still performs the actual bounded queue insertion/drop-oldest operation.
    if verify_queue_drop:
        original_push = ingestion.accumulator._push

        def traced_push(segment: SegmentBatch) -> None:
            summary = segment.as_summary()
            with event_lock:
                closed_summaries.append(summary)
            print(f"[{time.strftime('%H:%M:%S')}] SEGMENT CLOSED: {summary}", flush=True)
            original_push(segment)

        ingestion.accumulator._push = traced_push  # type: ignore[method-assign]

    buffer.start()
    ingestion.start()

    if verify_queue_drop:
        print(
            f"Running for {duration:g}s in QUEUE VERIFICATION mode.\n"
            f"The consumer is paused; queue capacity={queue_size}, "
            f"segment cap={segment_seconds:g}s.\n",
            flush=True,
        )
    else:
        print(f"Running for {duration:g}s. Segments will print as they close.\n")

    deadline = time.monotonic() + duration
    consumed: list[SegmentBatch] = []
    next_inspection = time.monotonic()

    try:
        while time.monotonic() < deadline:
            if verify_queue_drop:
                now = time.monotonic()
                if now >= next_inspection:
                    queued = _queue_snapshot(output_queue)
                    active = _active_segment_summary(ingestion)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] QUEUE SNAPSHOT "
                        f"size={len(queued)}/{queue_size} "
                        f"stored_seq={[item['seq_num'] for item in queued]} "
                        f"active_seq={None if active is None else active['seq_num']} "
                        f"active_frames={0 if active is None else active['frame_count']} "
                        f"dropped_count={len(dropped_summaries)}",
                        flush=True,
                    )
                    next_inspection = now + inspect_every
                time.sleep(0.05)
                continue

            try:
                segment = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            consumed.append(segment)
            print(
                f"[{time.strftime('%H:%M:%S')}] SEGMENT CLOSED: {segment.as_summary()}",
                flush=True,
            )
    finally:
        ingestion.stop()  # force-close a non-empty active segment before final inspection
        buffer.stop()

    if not verify_queue_drop:
        final_segments = _drain_queue(output_queue)
        for segment in final_segments:
            consumed.append(segment)
            print(
                f"[{time.strftime('%H:%M:%S')}] FINAL SEGMENT: {segment.as_summary()}",
                flush=True,
            )
        print(
            f"\nDone. {len(consumed)} segment(s) consumed. "
            f"frames_seen={ingestion.frames_seen} samples_taken={ingestion.samples_taken}"
        )
        return 0 if ingestion.frames_seen > 0 else 1

    # Final non-destructive snapshot, then drain to verify FIFO order.
    retained_before_drain = _queue_snapshot(output_queue)
    retained_segments = _drain_queue(output_queue)
    retained_seq = [segment.seq_num for segment in retained_segments]
    with event_lock:
        closed_seq = [int(item["seq_num"]) for item in closed_summaries]
        dropped_seq = [int(item["seq_num"]) for item in dropped_summaries]

    expected_retained = closed_seq[-queue_size:]
    expected_dropped = closed_seq[:-queue_size] if len(closed_seq) > queue_size else []

    print("\n--- queue verification result ---")
    print(f"closed_seq:   {closed_seq}")
    print(f"dropped_seq:  {dropped_seq}")
    print(f"retained_seq: {retained_seq}")
    print(f"queue_snapshot_before_drain: {retained_before_drain}")
    print(f"frames_seen: {ingestion.frames_seen}")
    print(f"samples_taken: {ingestion.samples_taken}")

    failures: list[str] = []
    if ingestion.frames_seen <= 0:
        failures.append("No frames reached PresenceGatedIngestion")
    if not closed_seq:
        failures.append("No segment closed; keep a person visible or run longer")
    if closed_seq != list(range(1, len(closed_seq) + 1)):
        failures.append(f"Closed segment sequence is not contiguous: {closed_seq}")
    if len(closed_seq) <= queue_size:
        failures.append(
            "Queue never overflowed, so drop-oldest was not proven. "
            f"Need at least {queue_size + 1} closed segments; observed {len(closed_seq)}. "
            "Run longer or reduce --queue-size/--segment-seconds."
        )
    if dropped_seq != expected_dropped:
        failures.append(f"Expected dropped sequence {expected_dropped}, got {dropped_seq}")
    if retained_seq != expected_retained:
        failures.append(f"Expected newest retained sequence {expected_retained}, got {retained_seq}")
    if len(retained_seq) > queue_size:
        failures.append(
            f"Queue retained {len(retained_seq)} segments despite queue_size={queue_size}"
        )
    if retained_seq != sorted(retained_seq):
        failures.append(f"Retained queue was not drained in FIFO order: {retained_seq}")

    if failures:
        print(f"\nQUEUE VERIFICATION FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "\nQUEUE VERIFICATION PASSED: the queue stayed bounded, dropped the oldest "
        "unconsumed segments, and retained the newest segments in FIFO order."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="RTSP URI for live mode; omit for the synthetic self-test",
    )
    parser.add_argument("--duration", type=float, default=90.0, help="live run duration in seconds")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--decoder", default=None, help="GStreamer decoder override")
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="maximum number of closed, unconsumed SegmentBatch objects (default: 2)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="maximum segment duration before rotation (default: 10)",
    )
    parser.add_argument(
        "--verify-queue-drop",
        action="store_true",
        help=(
            "pause the segment consumer, inspect the bounded queue, and assert that "
            "overflow drops the oldest unconsumed segment"
        ),
    )
    parser.add_argument(
        "--inspect-every",
        type=float,
        default=2.0,
        help="seconds between queue snapshots in --verify-queue-drop mode (default: 2)",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.queue_size <= 0:
        parser.error("--queue-size must be greater than zero")
    if args.segment_seconds < 0.5:
        parser.error("--segment-seconds must be at least 0.5")

    if args.uri:
        return run_live(
            uri=args.uri,
            duration=args.duration,
            codec=args.codec,
            decoder=args.decoder,
            queue_size=args.queue_size,
            segment_seconds=args.segment_seconds,
            verify_queue_drop=args.verify_queue_drop,
            inspect_every=args.inspect_every,
        )

    if args.verify_queue_drop:
        parser.error("--verify-queue-drop requires --uri; synthetic mode already tests queue overflow")
    return run_synthetic()


if __name__ == "__main__":
    sys.exit(main())