from __future__ import annotations

import threading
import time

from forensics.person_creation import live_stream


class FakeCapture:
    def __init__(self, *, block_read: bool = False, fail_read: bool = False) -> None:
        self.created_by = threading.get_ident()
        self.block_read = block_read
        self.fail_read = fail_read
        self.read_started = threading.Event()
        self.allow_read = threading.Event()
        self.read_threads: list[int] = []
        self.release_threads: list[int] = []

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.read_threads.append(threading.get_ident())
        self.read_started.set()
        if self.block_read:
            self.allow_read.wait(timeout=2.0)
        if self.fail_read:
            raise RuntimeError("fake read failure")
        return False, None

    def release(self) -> None:
        self.release_threads.append(threading.get_ident())


def _install_capture(monkeypatch, capture: FakeCapture):
    calls = []

    def factory(*args):
        capture.created_by = threading.get_ident()
        calls.append((threading.get_ident(), args))
        return capture

    monkeypatch.setattr(live_stream.cv2, "VideoCapture", factory)
    return calls


def test_reader_thread_owns_open_read_and_release(monkeypatch):
    controller_thread = threading.get_ident()
    capture = FakeCapture()
    calls = _install_capture(monkeypatch, capture)
    buffer = live_stream.LiveFrameBuffer("rtsp://user:secret@camera.local/live")

    buffer.start()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()

    assert len(calls) == 1
    assert calls[0][0] != controller_thread
    assert capture.created_by == calls[0][0]
    assert capture.read_threads == [calls[0][0]]
    assert capture.release_threads == [calls[0][0]]


def test_stop_never_releases_while_reader_is_blocked(monkeypatch):
    controller_thread = threading.get_ident()
    capture = FakeCapture(block_read=True)
    _install_capture(monkeypatch, capture)
    buffer = live_stream.LiveFrameBuffer(
        "rtsp://camera.local/live",
        shutdown_timeout_seconds=0.02,
    )
    buffer.start()
    assert capture.read_started.wait(timeout=1.0)

    started = time.monotonic()
    buffer.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert buffer._stop_event.is_set()
    assert capture.release_threads == []
    assert buffer._thread is not None and buffer._thread.is_alive()

    capture.allow_read.set()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()
    assert capture.release_threads == capture.read_threads
    assert controller_thread not in capture.release_threads


def test_repeated_stop_after_normal_reader_exit_is_safe(monkeypatch):
    capture = FakeCapture()
    _install_capture(monkeypatch, capture)
    buffer = live_stream.LiveFrameBuffer("rtsp://camera.local/live")

    buffer.start()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()
    buffer.stop()

    assert len(capture.release_threads) == 1
    assert buffer._stop_event.is_set()


def test_stop_before_start_is_safe_and_prevents_open(monkeypatch):
    capture = FakeCapture()
    calls = _install_capture(monkeypatch, capture)
    buffer = live_stream.LiveFrameBuffer("rtsp://camera.local/live")

    buffer.stop()
    buffer.stop()
    buffer.start()

    assert calls == []
    assert capture.release_threads == []
    assert buffer._stop_event.is_set()


def test_reader_failure_releases_owned_capture(monkeypatch):
    capture = FakeCapture(fail_read=True)
    calls = _install_capture(monkeypatch, capture)
    buffer = live_stream.LiveFrameBuffer("rtsp://camera.local/live")

    buffer.start()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()

    assert buffer.error == "Camera stream stopped while reading frames."
    assert capture.release_threads == [calls[0][0]]
    assert capture.release_threads == capture.read_threads


def test_shutdown_logs_do_not_expose_rtsp_credentials(monkeypatch, capsys):
    capture = FakeCapture()
    _install_capture(monkeypatch, capture)
    raw_uri = "rtsp://private-user:private-password@camera.local/live"
    buffer = live_stream.LiveFrameBuffer(raw_uri)

    buffer.start()
    assert buffer._reader_stopped.wait(timeout=1.0)
    buffer.stop()
    output = capsys.readouterr().out

    assert raw_uri not in output
    assert "private-user" not in output
    assert "private-password" not in output
    assert "[LiveFrameBuffer] releasing capture" in output
    assert "[LiveFrameBuffer] reader joined" in output
