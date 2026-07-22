"""Observational timing/timeline instrumentation for the realtime pipeline.

This module is never imported by production code paths (graph.py, realtime_main.py,
presence_segmentation.py are untouched). It exists purely so tests and the opt-in
profiler in tools/profile_realtime_timeline.py can answer "where did the time go"
questions about one camera's segment -> graph -> node timeline without changing any
detection/clustering/ReID/Global Memory/VLM behavior.

Two moving parts:
  - TimelineRecorder: a thread-safe, append-only event log. Every event carries both
    a `time.perf_counter_ns()` value (for durations -- monotonic, immune to wall-clock
    adjustments) and a `datetime.now(timezone.utc)` string (for human-readable
    reports only, never used to compute a duration).
  - Analysis functions (`segment_metrics_from_events`, `global_stats`,
    `overlap_analysis`) that turn a flat event list into per-segment timing rows and
    aggregate throughput/backlog statistics, plus renderers for a human-readable
    report, a compact table, and JSON/CSV export.

No global mutable state: every recorder instance is explicit and passed in by the
caller (test or profiler), never a module-level singleton.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TimelineEvent:
    event: str
    monotonic_ns: int
    timestamp_utc: str
    camera_id: str
    segment_id: str | None = None
    segment_seq_num: int | None = None
    node_name: str | None = None
    thread_name: str | None = None
    queue_depth: int | None = None
    details: dict[str, object] = field(default_factory=dict)


class TimelineRecorder:
    """Thread-safe append-only event log. Safe to call concurrently from the
    ingestion/capture thread, the graph-processing thread, and a test-control thread.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._events: list[TimelineEvent] = []

    def record(
        self,
        event: str,
        *,
        segment_id: str | None = None,
        segment_seq_num: int | None = None,
        node_name: str | None = None,
        queue_depth: int | None = None,
        **details: object,
    ) -> TimelineEvent:
        ev = TimelineEvent(
            event=event,
            monotonic_ns=time.perf_counter_ns(),
            timestamp_utc=_utc_now_iso(),
            camera_id=self.camera_id,
            segment_id=segment_id,
            segment_seq_num=segment_seq_num,
            node_name=node_name,
            thread_name=threading.current_thread().name,
            queue_depth=queue_depth,
            details=dict(details),
        )
        with self._lock:
            self._events.append(ev)
        return ev

    def events(self) -> list[TimelineEvent]:
        with self._lock:
            return list(self._events)


# ---------------------------------------------------------------------------- node timing


def timed_node(
    node_name: str,
    node_fn: Callable[[dict], dict],
    recorder: TimelineRecorder,
    *,
    segment_id_fn: Callable[[dict], str | None] | None = None,
) -> Callable[[dict], dict]:
    """Wrap a LangGraph node function with start/finish/failed timing events.

    Does not change the wrapped function's signature, return value, exception
    behavior, or how LangGraph merges its returned state update -- `wrapped` calls
    `node_fn(state)` exactly once and returns/raises exactly what it returns/raises.
    `stream_mode="updates"` alone only reveals when a node *finished*; this wrapper
    is what supplies the start timestamp.
    """

    def wrapped(state: dict) -> dict:
        segment_id = segment_id_fn(state) if segment_id_fn is not None else state.get("segment_id")
        seq_num = state.get("segment_seq_num")
        recorder.record("graph_node_started", segment_id=segment_id, segment_seq_num=seq_num, node_name=node_name)
        start_ns = time.perf_counter_ns()
        try:
            result = node_fn(state)
        except Exception as exc:
            end_ns = time.perf_counter_ns()
            recorder.record(
                "graph_node_failed",
                segment_id=segment_id,
                segment_seq_num=seq_num,
                node_name=node_name,
                details={"duration_seconds": (end_ns - start_ns) / 1_000_000_000, "error_type": type(exc).__name__},
            )
            raise
        end_ns = time.perf_counter_ns()
        recorder.record(
            "graph_node_finished",
            segment_id=segment_id,
            segment_seq_num=seq_num,
            node_name=node_name,
            details={"duration_seconds": (end_ns - start_ns) / 1_000_000_000},
        )
        return result

    return wrapped


# ---------------------------------------------------------------------------- analysis


@dataclass
class SegmentMetrics:
    segment_id: str
    seq_num: int | None = None

    segment_opened_ns: int | None = None
    segment_opened_iso: str | None = None
    first_frame_ns: int | None = None
    segment_closed_ns: int | None = None
    segment_closed_iso: str | None = None
    close_reason: str | None = None
    segment_incomplete: bool | None = None
    frame_count: int | None = None

    enqueued_ns: int | None = None
    queue_depth_at_enqueue: int | None = None
    dequeued_ns: int | None = None

    state_build_start_ns: int | None = None
    state_build_end_ns: int | None = None

    graph_start_ns: int | None = None
    graph_end_ns: int | None = None

    node_durations_seconds: dict[str, float] = field(default_factory=dict)

    result: str | None = None  # "SUCCEEDED" | "FAILED" | "DROPPED"
    error_type: str | None = None

    dropped: bool = False

    # -- derived (seconds) ------------------------------------------------------
    @property
    def capture_duration_s(self) -> float | None:
        if self.segment_opened_ns is None or self.segment_closed_ns is None:
            return None
        return (self.segment_closed_ns - self.segment_opened_ns) / 1_000_000_000

    @property
    def enqueue_delay_after_close_s(self) -> float | None:
        if self.segment_closed_ns is None or self.enqueued_ns is None:
            return None
        return (self.enqueued_ns - self.segment_closed_ns) / 1_000_000_000

    @property
    def queue_wait_s(self) -> float | None:
        if self.enqueued_ns is None or self.dequeued_ns is None:
            return None
        return (self.dequeued_ns - self.enqueued_ns) / 1_000_000_000

    @property
    def state_build_duration_s(self) -> float | None:
        if self.state_build_start_ns is None or self.state_build_end_ns is None:
            return None
        return (self.state_build_end_ns - self.state_build_start_ns) / 1_000_000_000

    @property
    def graph_duration_s(self) -> float | None:
        if self.graph_start_ns is None or self.graph_end_ns is None:
            return None
        return (self.graph_end_ns - self.graph_start_ns) / 1_000_000_000

    @property
    def total_processing_duration_s(self) -> float | None:
        if self.dequeued_ns is None or self.graph_end_ns is None:
            return None
        return (self.graph_end_ns - self.dequeued_ns) / 1_000_000_000

    @property
    def end_to_end_s(self) -> float | None:
        """First captured frame (falling back to segment_opened) through graph completion."""
        start = self.first_frame_ns if self.first_frame_ns is not None else self.segment_opened_ns
        if start is None or self.graph_end_ns is None:
            return None
        return (self.graph_end_ns - start) / 1_000_000_000

    @property
    def delay_behind_realtime_s(self) -> float | None:
        """graph_finish_time - segment_end_time: how far behind the live camera the
        graph's output trails once the segment itself has closed.
        """
        if self.segment_closed_ns is None or self.graph_end_ns is None:
            return None
        return (self.graph_end_ns - self.segment_closed_ns) / 1_000_000_000

    def as_dict(self) -> dict:
        d = asdict(self)
        for prop in (
            "capture_duration_s",
            "enqueue_delay_after_close_s",
            "queue_wait_s",
            "state_build_duration_s",
            "graph_duration_s",
            "total_processing_duration_s",
            "end_to_end_s",
            "delay_behind_realtime_s",
        ):
            d[prop] = getattr(self, prop)
        return d


def segment_metrics_from_events(events: Iterable[TimelineEvent]) -> list[SegmentMetrics]:
    """Group a flat TimelineRecorder event log into one SegmentMetrics row per
    segment_id, ordered by seq_num (falling back to first-seen order when seq_num is
    missing, e.g. for a segment that never made it past `segment_opened`).
    """
    by_segment: dict[str, SegmentMetrics] = {}
    order: list[str] = []

    def get(segment_id: str) -> SegmentMetrics:
        if segment_id not in by_segment:
            by_segment[segment_id] = SegmentMetrics(segment_id=segment_id)
            order.append(segment_id)
        return by_segment[segment_id]

    for ev in events:
        if ev.segment_id is None:
            continue
        m = get(ev.segment_id)
        if ev.segment_seq_num is not None:
            m.seq_num = ev.segment_seq_num

        if ev.event == "segment_opened":
            m.segment_opened_ns = ev.monotonic_ns
            m.segment_opened_iso = ev.timestamp_utc
        elif ev.event == "first_frame_added":
            m.first_frame_ns = ev.monotonic_ns
        elif ev.event == "segment_closed":
            m.segment_closed_ns = ev.monotonic_ns
            m.segment_closed_iso = ev.timestamp_utc
            m.close_reason = ev.details.get("close_reason")
            m.segment_incomplete = ev.details.get("segment_incomplete")
            m.frame_count = ev.details.get("frame_count")
        elif ev.event == "segment_enqueued":
            m.enqueued_ns = ev.monotonic_ns
            m.queue_depth_at_enqueue = ev.queue_depth
        elif ev.event == "segment_dequeued":
            m.dequeued_ns = ev.monotonic_ns
        elif ev.event == "initial_state_build_started":
            m.state_build_start_ns = ev.monotonic_ns
        elif ev.event == "initial_state_build_finished":
            m.state_build_end_ns = ev.monotonic_ns
        elif ev.event == "graph_started":
            m.graph_start_ns = ev.monotonic_ns
        elif ev.event == "graph_finished":
            m.graph_end_ns = ev.monotonic_ns
        elif ev.event == "graph_node_finished" and ev.node_name:
            m.node_durations_seconds[ev.node_name] = float(ev.details.get("duration_seconds", 0.0))
        elif ev.event == "graph_node_failed" and ev.node_name:
            m.node_durations_seconds[ev.node_name] = float(ev.details.get("duration_seconds", 0.0))
            m.error_type = ev.details.get("error_type")
        elif ev.event == "segment_processing_succeeded":
            m.result = "SUCCEEDED"
        elif ev.event == "segment_processing_failed":
            m.result = "FAILED"
            m.error_type = m.error_type or ev.details.get("error_type")
        elif ev.event == "segment_dropped":
            m.dropped = True
            m.result = m.result or "DROPPED"

    return [by_segment[sid] for sid in order]


@dataclass
class OverlapResult:
    seq_num: int | None
    next_seq_num: int | None
    overlapped: bool
    overlap_seconds: float


def overlap_analysis(metrics: list[SegmentMetrics]) -> list[OverlapResult]:
    """For consecutive segments (by list order, expected to already be seq_num-sorted),
    determine whether the next segment's capture began before the current segment's
    graph execution finished, and by how much its capture/graph windows overlap.
    """
    results: list[OverlapResult] = []
    for i in range(len(metrics) - 1):
        cur, nxt = metrics[i], metrics[i + 1]
        if cur.graph_start_ns is None or cur.graph_end_ns is None or nxt.segment_opened_ns is None:
            continue
        overlapped = nxt.segment_opened_ns < cur.graph_end_ns
        next_close = nxt.segment_closed_ns if nxt.segment_closed_ns is not None else nxt.segment_opened_ns
        overlap_start = max(cur.graph_start_ns, nxt.segment_opened_ns)
        overlap_end = min(cur.graph_end_ns, next_close)
        overlap_ns = max(0, overlap_end - overlap_start)
        results.append(
            OverlapResult(
                seq_num=cur.seq_num,
                next_seq_num=nxt.seq_num,
                overlapped=overlapped,
                overlap_seconds=overlap_ns / 1_000_000_000,
            )
        )
    return results


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def global_stats(metrics: list[SegmentMetrics]) -> dict:
    """Aggregate throughput/backlog statistics. `processing_utilization` is a
    throughput measure (mean graph duration vs. mean segment production interval),
    not a guarantee that every individual graph execution finishes under the segment
    cap -- see the docstring on the returned dict's "note" key.
    """
    succeeded = [m for m in metrics if m.result == "SUCCEEDED"]
    failed = [m for m in metrics if m.result == "FAILED"]
    dropped = [m for m in metrics if m.dropped]

    graph_durations = [m.graph_duration_s for m in metrics if m.graph_duration_s is not None]
    capture_durations = [m.capture_duration_s for m in metrics if m.capture_duration_s is not None]
    queue_waits = [m.queue_wait_s for m in metrics if m.queue_wait_s is not None]
    queue_depths = [m.queue_depth_at_enqueue for m in metrics if m.queue_depth_at_enqueue is not None]

    opened = sorted(m.segment_opened_ns for m in metrics if m.segment_opened_ns is not None)
    intervals_s = [(opened[i + 1] - opened[i]) / 1_000_000_000 for i in range(len(opened) - 1)]

    mean_graph = statistics.mean(graph_durations) if graph_durations else 0.0
    mean_interval = statistics.mean(intervals_s) if intervals_s else 0.0
    processing_utilization = (mean_graph / mean_interval) if mean_interval > 0 else 0.0
    required_speedup = max(1.0, processing_utilization)

    total_span_s = (opened[-1] - opened[0]) / 1_000_000_000 if len(opened) >= 2 else None
    throughput_per_min = (len(metrics) / total_span_s * 60.0) if total_span_s else None

    return {
        "segment_count": len(metrics),
        "successful_segment_count": len(succeeded),
        "failed_segment_count": len(failed),
        "dropped_segment_count": len(dropped),
        "mean_graph_duration_s": mean_graph,
        "median_graph_duration_s": statistics.median(graph_durations) if graph_durations else 0.0,
        "p95_graph_duration_s": _percentile(graph_durations, 0.95),
        "max_graph_duration_s": max(graph_durations) if graph_durations else 0.0,
        "mean_capture_duration_s": statistics.mean(capture_durations) if capture_durations else 0.0,
        "mean_queue_wait_s": statistics.mean(queue_waits) if queue_waits else 0.0,
        "max_queue_depth": max(queue_depths) if queue_depths else 0,
        "throughput_segments_per_minute": throughput_per_min,
        "mean_segment_production_interval_s": mean_interval,
        "processing_utilization": processing_utilization,
        "required_speedup": required_speedup,
        "note": (
            "processing_utilization is a throughput measure (mean graph duration / mean "
            "segment production interval): <1.0 keeps up on average, ==1.0 is at the limit, "
            ">1.0 means backlog grows during continuous presence. It does not guarantee any "
            "single graph execution finishes under the segment cap."
        ),
    }


# ---------------------------------------------------------------------------- rendering


def render_table(metrics: list[SegmentMetrics]) -> str:
    header = (
        "seq | capture_start_s | capture_end_s | capture_s | enqueue_s | dequeue_s | "
        "queue_wait_s | graph_start_s | graph_end_s | graph_s | end_to_end_s | result"
    )
    lines = [header, "-" * len(header)]
    t0 = min((m.segment_opened_ns for m in metrics if m.segment_opened_ns is not None), default=0)

    def rel(ns: int | None) -> str:
        return f"{(ns - t0) / 1_000_000_000:8.3f}" if ns is not None else "     n/a"

    for m in metrics:
        lines.append(
            f"{str(m.seq_num):>3} | {rel(m.segment_opened_ns)} | {rel(m.segment_closed_ns)} | "
            f"{(m.capture_duration_s or 0):9.3f} | {rel(m.enqueued_ns)} | {rel(m.dequeued_ns)} | "
            f"{(m.queue_wait_s or 0):12.3f} | {rel(m.graph_start_ns)} | {rel(m.graph_end_ns)} | "
            f"{(m.graph_duration_s or 0):7.3f} | {(m.end_to_end_s or 0):12.3f} | {m.result or 'PENDING'}"
        )
    return "\n".join(lines)


def render_human_summary(
    *,
    camera_id: str,
    profiling_start_iso: str,
    profiling_duration_s: float,
    max_segment_seconds: float,
    queue_capacity: int,
    metrics: list[SegmentMetrics],
    stats: dict,
    overlaps: list[OverlapResult],
) -> str:
    lines = [
        "Realtime Segment Timeline",
        "=" * 25,
        "",
        f"Camera: {camera_id}",
        f"Profiling start: {profiling_start_iso}",
        f"Profiling duration: {profiling_duration_s:.1f} s",
        f"Maximum segment duration: {max_segment_seconds:.1f} s",
        f"Queue capacity: {queue_capacity}",
        "Queue policy: drop-oldest",
        "",
    ]
    t0 = min((m.segment_opened_ns for m in metrics if m.segment_opened_ns is not None), default=0)

    def rel(ns: int | None) -> str | None:
        return None if ns is None else (ns - t0) / 1_000_000_000

    for m in metrics:
        lines.append(f"Segment {m.seq_num if m.seq_num is not None else '?'}")
        lines.append("-" * 9)
        lines.append(f"Segment ID: {m.segment_id}")
        if rel(m.segment_opened_ns) is not None:
            lines.append(f"Capture opened:            +{rel(m.segment_opened_ns):.3f} s")
        if rel(m.first_frame_ns) is not None:
            lines.append(f"First frame:               +{rel(m.first_frame_ns):.3f} s")
        if rel(m.segment_closed_ns) is not None:
            lines.append(f"Capture closed:            +{rel(m.segment_closed_ns):.3f} s")
        if m.capture_duration_s is not None:
            lines.append(f"Capture duration:           {m.capture_duration_s:.3f} s")
        if m.frame_count is not None:
            lines.append(f"Frames captured:            {m.frame_count}")
        if rel(m.enqueued_ns) is not None:
            lines.append(f"Enqueued:                  +{rel(m.enqueued_ns):.3f} s")
        if rel(m.dequeued_ns) is not None:
            lines.append(f"Dequeued:                  +{rel(m.dequeued_ns):.3f} s")
        if m.queue_wait_s is not None:
            lines.append(f"Queue wait:                 {m.queue_wait_s:.3f} s")
        if m.state_build_duration_s is not None:
            lines.append(f"State build duration:       {m.state_build_duration_s:.3f} s")
        if rel(m.graph_start_ns) is not None:
            lines.append(f"Graph started:             +{rel(m.graph_start_ns):.3f} s")
        if rel(m.graph_end_ns) is not None:
            lines.append(f"Graph finished:            +{rel(m.graph_end_ns):.3f} s")
        if m.graph_duration_s is not None:
            lines.append(f"Graph duration:             {m.graph_duration_s:.3f} s")
        if m.end_to_end_s is not None:
            lines.append(f"End-to-end latency:         {m.end_to_end_s:.3f} s")
        if m.delay_behind_realtime_s is not None:
            lines.append(f"Delay behind realtime:      {m.delay_behind_realtime_s:.3f} s")
        lines.append(f"Result: {m.result or 'PENDING'}{' (incomplete)' if m.segment_incomplete else ''}")
        if m.node_durations_seconds:
            lines.append("")
            lines.append("Node durations:")
            for name, dur in m.node_durations_seconds.items():
                lines.append(f"  {name:<28}{dur:8.3f} s")
        lines.append("")

    lines.append("Concurrency observations")
    lines.append("-" * 24)
    for ov in overlaps:
        lines.append(
            f"Segment {ov.next_seq_num} capture began while Segment {ov.seq_num} graph was running: "
            f"{'YES' if ov.overlapped else 'NO'}"
            + (f" (overlap {ov.overlap_seconds:.3f} s)" if ov.overlapped else "")
        )
    lines.append(f"Maximum queue depth: {stats['max_queue_depth']}")
    lines.append(f"Dropped segments: {stats['dropped_segment_count']}")
    lines.append("")
    lines.append("Global statistics")
    lines.append("-" * 18)
    lines.append(f"Segment count: {stats['segment_count']}")
    lines.append(f"Successful: {stats['successful_segment_count']}  Failed: {stats['failed_segment_count']}  Dropped: {stats['dropped_segment_count']}")
    lines.append(f"Mean graph duration: {stats['mean_graph_duration_s']:.3f} s")
    lines.append(f"Median graph duration: {stats['median_graph_duration_s']:.3f} s")
    lines.append(f"P95 graph duration: {stats['p95_graph_duration_s']:.3f} s")
    lines.append(f"Max graph duration: {stats['max_graph_duration_s']:.3f} s")
    lines.append(f"Mean capture duration: {stats['mean_capture_duration_s']:.3f} s")
    lines.append(f"Mean queue wait: {stats['mean_queue_wait_s']:.3f} s")
    if stats["throughput_segments_per_minute"] is not None:
        lines.append(f"Throughput: {stats['throughput_segments_per_minute']:.2f} segments/min")
    lines.append(f"Mean segment production interval: {stats['mean_segment_production_interval_s']:.3f} s")
    lines.append(
        f"Processing utilization: {stats['processing_utilization']:.2f} "
        f"(required speedup: {stats['required_speedup']:.2f}x)"
    )
    lines.append("")
    lines.append(render_table(metrics))
    return "\n".join(lines)


def to_json(
    *,
    camera_id: str,
    metrics: list[SegmentMetrics],
    stats: dict,
    overlaps: list[OverlapResult],
) -> dict:
    """JSON-safe report payload -- no frames/embeddings/credentials, only timestamps,
    counts, and durations.
    """
    return {
        "camera_id": camera_id,
        "segments": [m.as_dict() for m in metrics],
        "global_stats": stats,
        "overlaps": [asdict(o) for o in overlaps],
    }


def to_csv(metrics: list[SegmentMetrics]) -> str:
    cols = [
        "seq_num",
        "segment_id",
        "segment_opened_ns",
        "segment_closed_ns",
        "capture_duration_s",
        "enqueued_ns",
        "dequeued_ns",
        "queue_wait_s",
        "graph_start_ns",
        "graph_end_ns",
        "graph_duration_s",
        "end_to_end_s",
        "delay_behind_realtime_s",
        "result",
        "segment_incomplete",
    ]
    lines = [",".join(cols)]
    for m in metrics:
        d = m.as_dict()
        lines.append(",".join(str(d.get(c, "")) for c in cols))
    return "\n".join(lines)


def write_reports(
    *,
    camera_id: str,
    profiling_start_iso: str,
    profiling_duration_s: float,
    max_segment_seconds: float,
    queue_capacity: int,
    recorder: TimelineRecorder,
    output_json: str | None = None,
    output_csv: str | None = None,
    output_summary: str | None = None,
) -> dict:
    """Run the full analysis pipeline over `recorder`'s events and optionally write
    the three report formats to disk. Returns the JSON payload either way, so callers
    (tests) can assert on it without touching the filesystem.
    """
    events = recorder.events()
    metrics = segment_metrics_from_events(events)
    stats = global_stats(metrics)
    overlaps = overlap_analysis(metrics)
    payload = to_json(camera_id=camera_id, metrics=metrics, stats=stats, overlaps=overlaps)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    if output_csv:
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write(to_csv(metrics))
    if output_summary:
        summary = render_human_summary(
            camera_id=camera_id,
            profiling_start_iso=profiling_start_iso,
            profiling_duration_s=profiling_duration_s,
            max_segment_seconds=max_segment_seconds,
            queue_capacity=queue_capacity,
            metrics=metrics,
            stats=stats,
            overlaps=overlaps,
        )
        with open(output_summary, "w", encoding="utf-8") as f:
            f.write(summary)

    return payload
