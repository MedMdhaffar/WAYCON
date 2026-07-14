"""Unit tests for the real-time bounded queue (no GPU or camera required)."""

from __future__ import annotations

import threading

from forensics.person_creation.gpu.realtime import LatestFrameQueue, TimedFrame


def _item(i: int) -> TimedFrame:
    return TimedFrame(camera_id="cam0", capture_monotonic=float(i), capture_epoch=float(i), frame=None)


def test_drops_oldest_when_full():
    q = LatestFrameQueue(maxsize=2)
    for i in range(5):
        q.put(_item(i))
    assert len(q) == 2
    assert q.stats.put_count == 5
    assert q.stats.dropped_count == 3
    # Newest frames survive; the consumer never sees stale ones.
    assert q.get().capture_monotonic == 3.0
    assert q.get().capture_monotonic == 4.0
    assert q.stats.get_count == 2


def test_get_timeout_returns_none():
    q = LatestFrameQueue(maxsize=2)
    assert q.get(timeout=0.05) is None


def test_get_wakes_on_put():
    q = LatestFrameQueue(maxsize=1)
    got: list[TimedFrame] = []

    def consumer():
        got.append(q.get(timeout=2.0))

    t = threading.Thread(target=consumer)
    t.start()
    q.put(_item(7))
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert got and got[0].capture_monotonic == 7.0


def test_maxsize_one_keeps_only_newest():
    q = LatestFrameQueue(maxsize=1)
    q.put(_item(1))
    q.put(_item(2))
    q.put(_item(3))
    assert len(q) == 1
    assert q.get().capture_monotonic == 3.0
    assert q.stats.dropped_count == 2
