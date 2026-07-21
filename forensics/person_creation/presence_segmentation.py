"""Tier 0 presence gate + Tier 1 segment accumulator (presence-gated micro-batching).

This is the subsystem between raw frame ingestion (gst_stream.GstFrameBuffer) and the
synchronous detection pipeline (YOLO -> face+embed -> filter -> cluster -> ReID ->
build_profile -> finalize). It replaces the current fixed-`duration_seconds` live-stream
window in nodes/process_live_stream.py with event-triggered, bounded-duration segments:

    GstFrameBuffer --(frames)--> PresenceGatedIngestion --(SegmentBatch)--> queue.Queue(maxsize=2)

Two tiers, both driven by the same dispatcher loop reading off the frame buffer:

  Tier 0 -- PresenceGate: a cheap, sampled (every `sample_interval_seconds`, default
  250-500ms) person-presence check reusing the existing YOLO person detector, with no
  downstream stages run on a negative check. Runs continuously regardless of whether a
  segment is currently open. Debounced both ways: `open_debounce_count` (>=2) consecutive
  positive checks before presence flips PRESENT (avoids opening a segment on one noisy
  frame); `close_debounce_seconds` (2-3s) of continuous absence before it flips back to
  ABSENT (avoids fragmenting one visit into many segments from a momentary miss).

  Tier 1 -- SegmentAccumulator: opens a SegmentBatch when presence flips PRESENT, appends
  every incoming frame (not just the sampled ones) plus its timestamp, caps segment
  duration at `max_segment_seconds` (default 10s) and rotates (closes + reopens) while
  presence continues, and closes on ABSENT. Closed segments are pushed to a bounded
  `queue.Queue(maxsize=2)` with drop-oldest-on-full semantics -- the sync pipeline (Tier 1
  consumer) is expected to drain this queue; if it falls behind, the oldest *unconsumed*
  segment is dropped rather than blocking ingestion.

A third state, UNKNOWN, exists for connection outages: `PresenceGatedIngestion` is meant to
be wired to `gst_stream.GstFrameBuffer`'s `on_state_change` callback (see
`PresenceGatedIngestion.on_connection_state_change`). On RECONNECTING, presence is forced to
UNKNOWN (not ABSENT) and any open segment is force-closed and tagged `segment_incomplete`
(never discarded). On reconnect, debounce always restarts from scratch -- presence is never
assumed to carry over a lost-frame gap.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PresenceState(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------- Tier 0


class PresenceGate:
    """Pure state machine: feed it (bool present, monotonic time) samples, it tells you
    when presence has flipped after the appropriate debounce. No threading/IO here --
    the driver (`PresenceGatedIngestion`) owns sampling cadence and frame access, which
    keeps this class trivially unit-testable without a real camera or model.
    """

    def __init__(
        self,
        open_debounce_count: int = 2,
        close_debounce_seconds: float = 2.5,
        on_change: Callable[[PresenceState, PresenceState, str], None] | None = None,
    ) -> None:
        self.open_debounce_count = max(1, int(open_debounce_count))
        self.close_debounce_seconds = max(0.0, float(close_debounce_seconds))
        self.on_change = on_change

        self.state: PresenceState = PresenceState.ABSENT
        self._positive_streak = 0
        self._absence_started_monotonic: float | None = None
        self._lock = threading.RLock()

    def observe(self, present: bool, now: float | None = None) -> None:
        """Feed one Tier 0 sample result. No-op while UNKNOWN (mid-reconnect) --
        the buffer isn't producing trustworthy frames, so sampling is meaningless
        until `reset_after_reconnect()` is called explicitly.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.state is PresenceState.UNKNOWN:
                return

            if present:
                self._positive_streak += 1
                self._absence_started_monotonic = None
                if self.state is PresenceState.ABSENT and self._positive_streak >= self.open_debounce_count:
                    self._transition(PresenceState.PRESENT, "open_debounce_satisfied")
            else:
                self._positive_streak = 0
                if self.state is PresenceState.PRESENT:
                    if self._absence_started_monotonic is None:
                        self._absence_started_monotonic = now
                    elif now - self._absence_started_monotonic >= self.close_debounce_seconds:
                        self._transition(PresenceState.ABSENT, "close_debounce_satisfied")

    def mark_unknown(self, reason: str) -> None:
        """Connection lost. Any open segment must be force-closed by the caller
        (SegmentAccumulator.on_presence_change handles that via the on_change callback).
        """
        with self._lock:
            self._positive_streak = 0
            self._absence_started_monotonic = None
            self._transition(PresenceState.UNKNOWN, reason)

    def reset_after_reconnect(self) -> None:
        """Connection restored. Debounce always restarts from scratch -- never assume
        presence carries across a lost-frame gap.
        """
        with self._lock:
            self._positive_streak = 0
            self._absence_started_monotonic = None
            self._transition(PresenceState.ABSENT, "reconnected_debounce_reset")

    def _transition(self, new_state: PresenceState, reason: str) -> None:
        old_state = self.state
        if old_state == new_state:
            return
        self.state = new_state
        if self.on_change is not None:
            try:
                self.on_change(old_state, new_state, reason)
            except Exception:
                pass


# ---------------------------------------------------------------------------- Tier 1


@dataclass
class SegmentBatch:
    segment_id: str
    seq_num: int
    codec: str
    segment_start_ts: str
    segment_end_ts: str | None = None
    frames: list[Any] = field(default_factory=list)         # frame tensor stack (np.ndarray per frame)
    frame_timestamps: list[str] = field(default_factory=list)
    segment_incomplete: bool = False
    close_reason: str | None = None
    _opened_monotonic: float = field(default_factory=time.monotonic, repr=False)

    def add_frame(self, frame: Any, timestamp: str) -> None:
        self.frames.append(frame)
        self.frame_timestamps.append(timestamp)

    def duration_seconds(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return now - self._opened_monotonic

    def close(self, *, reason: str, incomplete: bool = False) -> None:
        self.segment_end_ts = utc_now_iso()
        self.close_reason = reason
        self.segment_incomplete = incomplete

    def as_summary(self) -> dict:
        """Compact metadata without the frame stack -- useful for logging/tests."""
        return {
            "segment_id": self.segment_id,
            "seq_num": self.seq_num,
            "codec": self.codec,
            "segment_start_ts": self.segment_start_ts,
            "segment_end_ts": self.segment_end_ts,
            "frame_count": len(self.frames),
            "segment_incomplete": self.segment_incomplete,
            "close_reason": self.close_reason,
        }


class SegmentAccumulator:
    """Owns the currently-open SegmentBatch (if any) and the bounded output queue."""

    def __init__(
        self,
        output_queue: "queue.Queue[SegmentBatch]",
        codec: str = "h264",
        max_segment_seconds: float = 10.0,
        on_segment_dropped: Callable[[SegmentBatch], None] | None = None,
        on_segment_state_change: Callable[[SegmentBatch, str], None] | None = None,
    ) -> None:
        self.output_queue = output_queue
        self.codec = codec
        self.max_segment_seconds = max(0.5, float(max_segment_seconds))
        self._on_segment_dropped = on_segment_dropped
        # Optional hook for persisting the segment reliability state machine
        # (CAPTURING on open, READY on close) -- e.g. GlobalMemory.upsert_segment.
        # Kept as a callback rather than an import so this module has no DB
        # dependency and stays testable without one; called with (segment, status).
        self._on_segment_state_change = on_segment_state_change

        self._current: SegmentBatch | None = None
        self._seq_num = 0
        self._lock = threading.RLock()

    # -- presence-driven lifecycle -------------------------------------------------

    def on_presence_change(self, old_state: PresenceState, new_state: PresenceState, reason: str) -> None:
        with self._lock:
            if new_state is PresenceState.PRESENT and old_state is not PresenceState.PRESENT:
                self._open_segment()
            elif new_state is PresenceState.ABSENT and old_state is PresenceState.PRESENT:
                self._close_segment(reason=f"presence_absent:{reason}", incomplete=False)
            elif new_state is PresenceState.UNKNOWN:
                # Connection lost: force-close, tag incomplete, never discard.
                self._close_segment(reason=f"connection_lost:{reason}", incomplete=True)

    # -- frame-driven accumulation ---------------------------------------------------

    def add_frame(self, frame: Any, timestamp: str) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current.add_frame(frame, timestamp)
            if self._current.duration_seconds() >= self.max_segment_seconds:
                self._rotate_segment()

    def force_close_open(self, reason: str) -> None:
        """Called on shutdown so an in-progress segment isn't silently lost."""
        with self._lock:
            self._close_segment(reason=reason, incomplete=True)

    # -- internals --------------------------------------------------------------------

    def _open_segment(self) -> None:
        self._seq_num += 1
        self._current = SegmentBatch(
            segment_id=str(uuid.uuid4()),
            seq_num=self._seq_num,
            codec=self.codec,
            segment_start_ts=utc_now_iso(),
        )
        self._notify_state_change(self._current, "CAPTURING")

    def _rotate_segment(self) -> None:
        self._close_segment(reason="duration_cap", incomplete=False)
        self._open_segment()

    def _close_segment(self, *, reason: str, incomplete: bool) -> None:
        if self._current is None:
            return
        segment = self._current
        segment.close(reason=reason, incomplete=incomplete)
        self._current = None
        if segment.frames:
            self._notify_state_change(segment, "READY")
            self._push(segment)

    def _notify_state_change(self, segment: SegmentBatch, status: str) -> None:
        if self._on_segment_state_change is not None:
            try:
                self._on_segment_state_change(segment, status)
            except Exception:
                pass

    def _push(self, segment: SegmentBatch) -> None:
        if self.output_queue.full():
            try:
                dropped = self.output_queue.get_nowait()
                if self._on_segment_dropped is not None:
                    try:
                        self._on_segment_dropped(dropped)
                    except Exception:
                        pass
            except queue.Empty:
                pass
        try:
            self.output_queue.put_nowait(segment)
        except queue.Full:
            if self._on_segment_dropped is not None:
                try:
                    self._on_segment_dropped(segment)
                except Exception:
                    pass


# ---------------------------------------------------------------------------- driver


class PresenceGatedIngestion:
    """Ties a frame source (anything with `.get(timeout) -> BufferedFrame | None`, i.e.
    gst_stream.GstFrameBuffer) to PresenceGate + SegmentAccumulator via one dispatcher
    thread, and exposes `on_connection_state_change` to wire into the frame source's
    reconnect callbacks.
    """

    def __init__(
        self,
        frame_source: Any,
        detect_person_fn: Callable[[Any], bool],
        output_queue: "queue.Queue[SegmentBatch] | None" = None,
        sample_interval_seconds: float = 0.35,
        open_debounce_count: int = 2,
        close_debounce_seconds: float = 2.5,
        max_segment_seconds: float = 10.0,
        codec: str = "h264",
        on_segment_dropped: Callable[[SegmentBatch], None] | None = None,
        on_segment_state_change: Callable[[SegmentBatch, str], None] | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.detect_person_fn = detect_person_fn
        self.output_queue: "queue.Queue[SegmentBatch]" = output_queue or queue.Queue(maxsize=2)
        self.sample_interval_seconds = max(0.05, float(sample_interval_seconds))

        self.accumulator = SegmentAccumulator(
            self.output_queue,
            codec=codec,
            max_segment_seconds=max_segment_seconds,
            on_segment_dropped=on_segment_dropped,
            on_segment_state_change=on_segment_state_change,
        )
        self.gate = PresenceGate(
            open_debounce_count=open_debounce_count,
            close_debounce_seconds=close_debounce_seconds,
            on_change=self.accumulator.on_presence_change,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sample_monotonic = 0.0

        self.frames_seen = 0
        self.samples_taken = 0

    # -- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="presence-gated-ingestion", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.accumulator.force_close_open("ingestion_stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self.frame_source.get(timeout=0.5)
            if item is None:
                continue
            self.frames_seen += 1
            self.accumulator.add_frame(item.frame, item.timestamp)

            now = time.monotonic()
            if now - self._last_sample_monotonic >= self.sample_interval_seconds:
                self._last_sample_monotonic = now
                self.samples_taken += 1
                present = bool(self.detect_person_fn(item.frame))
                self.gate.observe(present, now=now)

    # -- reconnect hook, wire to GstFrameBuffer(on_state_change=...) --------------

    def on_connection_state_change(self, state) -> None:
        """`state` is a gst_stream.ConnectionState; typed loosely here to avoid a hard
        import dependency (this module has no other reason to import gst_stream, and
        stays usable with any frame source implementing the same `.get()` contract,
        e.g. a synthetic one in tests).
        """
        value = getattr(state, "value", state)
        if value == "reconnecting":
            self.gate.mark_unknown("connection_reconnecting")
        elif value == "connected":
            self.gate.reset_after_reconnect()
