from __future__ import annotations

import json
import threading
import time
from collections import deque

import pytest

from forensics.person_creation import live_stream


_BLOCK = object()


class FakeCapture:
    def __init__(
        self,
        *,
        opened: bool = True,
        reads=(),
        default_read=(True, object()),
    ) -> None:
        self.opened = opened
        self.reads = deque(reads)
        self.default_read = default_read
        self.created_by: int | None = None
        self.read_started = threading.Event()
        self.allow_read = threading.Event()
        self.read_threads: list[int] = []
        self.release_threads: list[int] = []
        self.factory = None

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.read_threads.append(threading.get_ident())
        self.read_started.set()
        action = self.reads.popleft() if self.reads else self.default_read
        if action is _BLOCK:
            self.allow_read.wait()
            return False, None
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action()
        return action

    def release(self) -> None:
        self.release_threads.append(threading.get_ident())
        if self.factory is not None:
            self.factory.capture_released(self)


class CaptureFactory:
    def __init__(self, captures=(), *, fallback=None) -> None:
        self.pending = deque(captures)
        self.fallback = fallback
        self.calls: list[tuple[int, tuple]] = []
        self.created: list[FakeCapture] = []
        self.active_ids: set[int] = set()
        self.max_active = 0

    def __call__(self, *args):
        if self.pending:
            capture = self.pending.popleft()
        elif self.fallback is not None:
            capture = self.fallback()
        else:  # pragma: no cover - a failed assertion gives the useful context
            raise AssertionError("Unexpected VideoCapture construction")
        thread_id = threading.get_ident()
        self.calls.append((thread_id, args))
        self.created.append(capture)
        self.active_ids.add(id(capture))
        self.max_active = max(self.max_active, len(self.active_ids))
        capture.created_by = thread_id
        capture.factory = self
        return capture

    def capture_released(self, capture: FakeCapture) -> None:
        self.active_ids.discard(id(capture))


def _install_factory(monkeypatch, factory: CaptureFactory) -> CaptureFactory:
    monkeypatch.setattr(live_stream.cv2, "VideoCapture", factory)
    return factory


def _buffer(uri: str = "rtsp://camera.local/live", **kwargs):
    options = {
        "reconnect_initial_delay_seconds": 0.01,
        "reconnect_max_delay_seconds": 0.02,
        "startup_max_attempts": 3,
        "startup_timeout_seconds": 0.5,
        "shutdown_timeout_seconds": 0.2,
    }
    options.update(kwargs)
    return live_stream.LiveFrameBuffer(uri, **options)


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate(), "condition was not reached before timeout"


def test_failed_read_reconnects_and_later_frames_keep_reader_alive(monkeypatch):
    first = FakeCapture(reads=[(False, None)])
    second = FakeCapture()
    factory = _install_factory(monkeypatch, CaptureFactory([first, second]))
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: buffer.stream_reconnect_count == 1 and buffer.frames_read > 0)
    frame = buffer.get(timeout=0.2)

    assert frame is not None
    assert buffer.error is None
    assert buffer.ended is False
    assert len(factory.created) == 2
    assert factory.created[0] is not factory.created[1]
    buffer.stop()


def test_multiple_failed_reads_trigger_multiple_reconnects(monkeypatch):
    captures = [
        FakeCapture(reads=[(False, None)]),
        FakeCapture(reads=[RuntimeError("decoder failure")]),
        FakeCapture(),
    ]
    _install_factory(monkeypatch, CaptureFactory(captures))
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: buffer.stream_reconnect_count == 2 and buffer.frames_read > 0)

    assert buffer.error is None
    assert buffer.stream_state == "connected"
    buffer.stop()


def test_reader_thread_owns_open_read_and_release_across_reconnect(monkeypatch):
    controller_thread = threading.get_ident()
    first = FakeCapture(reads=[(False, None)])
    second = FakeCapture()
    factory = _install_factory(monkeypatch, CaptureFactory([first, second]))
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: buffer.frames_read > 0)
    buffer.stop()

    owner_threads = {thread_id for thread_id, _args in factory.calls}
    assert len(owner_threads) == 1
    owner_thread = next(iter(owner_threads))
    assert owner_thread != controller_thread
    assert all(capture.created_by == owner_thread for capture in factory.created)
    assert all(set(capture.read_threads) == {owner_thread} for capture in factory.created)
    assert all(capture.release_threads == [owner_thread] for capture in factory.created)


def test_stop_during_reconnect_interrupts_backoff(monkeypatch):
    first = FakeCapture(reads=[(False, None)])
    factory = _install_factory(
        monkeypatch,
        CaptureFactory([first], fallback=lambda: FakeCapture(opened=False)),
    )
    buffer = _buffer(
        reconnect_initial_delay_seconds=5.0,
        reconnect_max_delay_seconds=5.0,
        shutdown_timeout_seconds=0.2,
    )

    buffer.start()
    _wait_until(lambda: buffer.stream_state == "reconnecting" and first.release_threads)
    started = time.monotonic()
    buffer.stop()

    assert time.monotonic() - started < 0.2
    assert len(factory.created) == 1
    assert buffer.stream_state == "stopped"


def test_stop_during_active_capture_joins_cleanly(monkeypatch):
    capture = FakeCapture()
    _install_factory(monkeypatch, CaptureFactory([capture]))
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: buffer.frames_read > 0)
    buffer.stop()
    buffer.stop()

    assert buffer._thread is None
    assert capture.release_threads == capture.read_threads[:1]
    assert len(capture.release_threads) == 1


def test_initial_open_failure_uses_bounded_attempts(monkeypatch):
    factory = _install_factory(
        monkeypatch,
        CaptureFactory(fallback=lambda: FakeCapture(opened=False)),
    )
    buffer = _buffer(startup_max_attempts=3)

    started = time.monotonic()
    with pytest.raises(OSError, match="bounded startup retries"):
        buffer.start()

    assert time.monotonic() - started < 0.5
    assert len(factory.created) == 3
    assert all(len(capture.release_threads) == 1 for capture in factory.created)
    assert buffer._thread is None
    assert buffer.stream_state == "error"


def test_failure_after_a_frame_retries_indefinitely_by_default(monkeypatch):
    first = FakeCapture(reads=[(True, object()), (False, None)])
    factory = _install_factory(
        monkeypatch,
        CaptureFactory([first], fallback=lambda: FakeCapture(opened=False)),
    )
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: len(factory.created) >= 6)

    assert buffer.maximum_outage_seconds is None
    assert buffer.error is None
    assert buffer._thread is not None and buffer._thread.is_alive()
    assert buffer.stream_state == "reconnecting"
    buffer.stop()


def test_reconnect_failures_keep_one_reader_and_one_live_capture(monkeypatch):
    first = FakeCapture(reads=[(False, None)])
    factory = _install_factory(
        monkeypatch,
        CaptureFactory([first], fallback=lambda: FakeCapture(opened=False)),
    )
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: len(factory.created) >= 8)
    named_readers = [
        thread
        for thread in threading.enumerate()
        if thread.name == "person-creation-live-reader"
    ]

    assert len(named_readers) == 1
    assert factory.max_active == 1
    buffer.stop()
    assert not factory.active_ids
    assert all(len(capture.release_threads) == 1 for capture in factory.created)


def test_every_failed_capture_is_released_exactly_once(monkeypatch):
    read_failure = FakeCapture(reads=[(False, None)])
    open_failure = FakeCapture(opened=False)
    recovered = FakeCapture()
    factory = _install_factory(
        monkeypatch,
        CaptureFactory([read_failure, open_failure, recovered]),
    )
    buffer = _buffer()

    buffer.start()
    _wait_until(lambda: buffer.frames_read > 0)
    buffer.stop()

    assert factory.created == [read_failure, open_failure, recovered]
    assert [len(capture.release_threads) for capture in factory.created] == [1, 1, 1]


def test_status_transitions_from_reconnecting_to_connected(monkeypatch):
    first = FakeCapture(reads=[(False, None)])
    second = FakeCapture()
    _install_factory(monkeypatch, CaptureFactory([first, second]))
    buffer = _buffer(
        reconnect_initial_delay_seconds=0.1,
        reconnect_max_delay_seconds=0.1,
    )

    buffer.start()
    _wait_until(lambda: buffer.stream_reconnect_count == 1)
    reconnecting = buffer.stats(0)
    _wait_until(lambda: buffer.stream_state == "connected" and buffer.frames_read > 0)
    connected = buffer.stats(0)
    buffer.stop()

    assert reconnecting["stream_state"] == "reconnecting"
    assert reconnecting["stream_warning"] == "Temporary camera interruption; reconnecting."
    assert reconnecting["stream_reconnect_count"] == 1
    assert connected["stream_state"] == "connected"
    assert connected["stream_warning"] is None
    assert connected["last_frame_age_seconds"] is not None


def test_status_and_logs_never_expose_camera_credentials(monkeypatch, capsys):
    first = FakeCapture(reads=[RuntimeError("rtsp://leaked:secret@camera/live")])
    second = FakeCapture()
    _install_factory(monkeypatch, CaptureFactory([first, second]))
    raw_uri = "rtsp://private-user:private-password@camera.local/live?token=secret"
    buffer = _buffer(raw_uri)

    buffer.start()
    _wait_until(lambda: buffer.frames_read > 0)
    status_text = json.dumps(buffer.stats(0))
    buffer.stop()
    output = capsys.readouterr().out

    for secret in (raw_uri, "private-user", "private-password", "leaked", "secret"):
        assert secret not in status_text
        assert secret not in output


def test_reader_exiting_just_before_shutdown_timeout_succeeds(monkeypatch):
    capture = FakeCapture(reads=[_BLOCK])
    _install_factory(monkeypatch, CaptureFactory([capture]))
    buffer = _buffer(shutdown_timeout_seconds=0.2)
    buffer.start()
    assert capture.read_started.wait(timeout=1.0)

    releaser = threading.Thread(
        target=lambda: (time.sleep(0.05), capture.allow_read.set())
    )
    releaser.start()
    buffer.stop()
    releaser.join(1.0)

    assert buffer._thread is None
    assert len(capture.release_threads) == 1
    assert capture.release_threads[0] == capture.read_threads[0]


def test_permanently_blocked_reader_fails_closed_without_cross_thread_release(
    monkeypatch,
):
    controller_thread = threading.get_ident()
    capture = FakeCapture(reads=[_BLOCK])
    _install_factory(monkeypatch, CaptureFactory([capture]))
    buffer = _buffer(shutdown_timeout_seconds=0.02)
    buffer.start()
    assert capture.read_started.wait(timeout=1.0)

    started = time.monotonic()
    with pytest.raises(
        live_stream.LiveFrameBufferLifecycleError,
        match="staging must be preserved",
    ):
        buffer.stop()

    assert time.monotonic() - started < 0.5
    assert buffer._stop_event.is_set()
    assert capture.release_threads == []
    assert buffer._thread is not None and buffer._thread.is_alive()

    capture.allow_read.set()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()
    assert len(capture.release_threads) == 1
    assert capture.release_threads[0] == capture.read_threads[0]
    assert controller_thread not in capture.release_threads


def test_stop_before_start_is_safe_and_prevents_open(monkeypatch):
    factory = _install_factory(
        monkeypatch,
        CaptureFactory(fallback=lambda: FakeCapture()),
    )
    buffer = _buffer()

    buffer.stop()
    buffer.stop()
    buffer.start()

    assert factory.calls == []
    assert buffer._stop_event.is_set()
    assert buffer.stream_state == "stopped"
