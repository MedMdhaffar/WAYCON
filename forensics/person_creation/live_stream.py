"""Bounded live-camera capture helpers with credential-safe reporting."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import cv2


class LiveFrameBufferLifecycleError(RuntimeError):
    """The reader may still own and access the camera capture."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_camera_uri(uri: str) -> str:
    """Mask a password in URI user-info while preserving the usable source label."""
    if not uri:
        return uri
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<invalid-camera-uri>"
    masked_netloc = parts.netloc
    if "@" in parts.netloc:
        _userinfo, hostinfo = parts.netloc.rsplit("@", 1)
        masked_netloc = f"****@{hostinfo}"

    query = parts.query
    pairs = parse_qsl(query, keep_blank_values=True)
    sensitive = {"password", "passwd", "pass", "token", "access_token", "api_key", "apikey"}
    if any(key.lower() in sensitive for key, _ in pairs):
        query = urlencode([
            (key, "****" if key.lower() in sensitive else value)
            for key, value in pairs
        ])
    return urlunsplit((parts.scheme, masked_netloc, parts.path, query, parts.fragment))


def capture_source_from_uri(uri: str) -> str:
    """Convert a file URI for OpenCV while leaving network URIs untouched."""
    parts = urlsplit(uri)
    if parts.scheme.lower() != "file":
        return uri
    if parts.netloc and parts.netloc not in {"", "localhost"}:
        return f"//{parts.netloc}{unquote(parts.path)}"
    return unquote(parts.path)


@dataclass
class BufferedFrame:
    frame_idx: int
    timestamp: str
    frame: Any
    captured_monotonic: float | None = None


class LiveFrameBuffer:
    """Continuously read a stream and retain at most the latest N frames."""

    def __init__(
        self,
        uri: str,
        max_size: int = 30,
        open_timeout_ms: int = 10_000,
        read_timeout_ms: int = 5_000,
        shutdown_timeout_seconds: float | None = None,
        reconnect_initial_delay_seconds: float = 0.5,
        reconnect_max_delay_seconds: float = 5.0,
        reconnect_backoff_multiplier: float = 2.0,
        startup_max_attempts: int = 3,
        startup_timeout_seconds: float | None = None,
        maximum_outage_seconds: float | None = None,
    ) -> None:
        self.uri = uri
        self.max_size = max(1, int(max_size))
        self.open_timeout_ms = max(1, int(open_timeout_ms))
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        default_shutdown_timeout = (
            max(self.open_timeout_ms, self.read_timeout_ms) / 1000.0
        ) + 1.0
        self.shutdown_timeout_seconds = max(
            0.01,
            float(
                default_shutdown_timeout
                if shutdown_timeout_seconds is None
                else shutdown_timeout_seconds
            ),
        )
        self.reconnect_initial_delay_seconds = max(
            0.01, float(reconnect_initial_delay_seconds)
        )
        self.reconnect_max_delay_seconds = max(
            self.reconnect_initial_delay_seconds,
            float(reconnect_max_delay_seconds),
        )
        self.reconnect_backoff_multiplier = max(
            1.0, float(reconnect_backoff_multiplier)
        )
        self.startup_max_attempts = max(1, int(startup_max_attempts))
        retry_budget = 0.0
        retry_delay = self.reconnect_initial_delay_seconds
        for _ in range(self.startup_max_attempts - 1):
            retry_budget += retry_delay
            retry_delay = min(
                self.reconnect_max_delay_seconds,
                retry_delay * self.reconnect_backoff_multiplier,
            )
        default_startup_timeout = (
            self.startup_max_attempts * ((self.open_timeout_ms / 1000.0) + 1.0)
            + retry_budget
        )
        self.startup_timeout_seconds = max(
            0.01,
            float(
                default_startup_timeout
                if startup_timeout_seconds is None
                else startup_timeout_seconds
            ),
        )
        self.maximum_outage_seconds = (
            None
            if maximum_outage_seconds is None
            else max(0.01, float(maximum_outage_seconds))
        )
        self.frames_read = 0
        self.frames_dropped = 0
        self.first_frame_time: str | None = None
        self.last_frame_time: str | None = None
        self.stream_opened = False
        self.stream_state = "stopped"
        self.stream_reconnect_count = 0
        self.stream_warning: str | None = None
        self.error: str | None = None
        self.ended = False
        self._last_frame_monotonic: float | None = None
        self._queue: queue.Queue[BufferedFrame] = queue.Queue(maxsize=self.max_size)
        self._stop_event = threading.Event()
        self._open_complete = threading.Event()
        self._reader_stopped = threading.Event()
        self._state_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cap = None
        self._started = False

    def _open_capture(self):
        source = capture_source_from_uri(self.uri)
        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms])
        try:
            if params:
                return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            pass
        return cv2.VideoCapture(source)

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError("Live frame buffer has already been started.")
            if self._stop_event.is_set():
                return
            self._started = True
            self.stream_state = "reconnecting"
            self._thread = threading.Thread(
                target=self._reader_loop,
                name="person-creation-live-reader",
                daemon=True,
            )
            thread = self._thread
        thread.start()

        startup_completed = self._open_complete.wait(
            timeout=self.startup_timeout_seconds
        )
        if not startup_completed:
            with self._state_lock:
                self.error = "Camera startup did not complete within its bounded timeout."
                self.stream_state = "error"
                self.stream_warning = self.error
            self._stop_event.set()
            thread.join(timeout=self.shutdown_timeout_seconds)
            if thread.is_alive():
                raise LiveFrameBufferLifecycleError(
                    "Live camera reader remained alive after startup timeout; "
                    "staging must be preserved."
                )
            raise OSError(self.error)
        if self.error and not self.stream_opened:
            thread.join(timeout=self.shutdown_timeout_seconds)
            if thread.is_alive():
                raise LiveFrameBufferLifecycleError(
                    "Live camera reader remained alive after startup failure; "
                    "staging must be preserved."
                )
            with self._state_lock:
                if self._thread is thread:
                    self._thread = None
            raise OSError(self.error)

    def _release_capture_owned(self, capture: Any) -> None:
        print("[LiveFrameBuffer] releasing capture", flush=True)
        try:
            capture.release()
        except Exception:
            with self._state_lock:
                self.stream_warning = "Camera capture cleanup failed; reconnecting."
            print("[LiveFrameBuffer] capture release failed", flush=True)
        else:
            print("[LiveFrameBuffer] capture released", flush=True)
        with self._state_lock:
            if self._cap is capture:
                self._cap = None

    def _set_reconnecting(self, *, increment: bool) -> None:
        with self._state_lock:
            if increment:
                self.stream_reconnect_count += 1
            self.stream_state = "reconnecting"
            self.stream_warning = "Temporary camera interruption; reconnecting."

    def _set_terminal_error(self, message: str) -> None:
        with self._state_lock:
            self.error = message
            self.stream_state = "error"
            self.stream_warning = message

    def _reader_loop(self) -> None:
        capture = None
        ever_connected = False
        startup_attempts = 0
        outage_started: float | None = None
        reconnect_delay = self.reconnect_initial_delay_seconds
        try:
            while not self._stop_event.is_set():
                if capture is None:
                    if (
                        outage_started is not None
                        and self.maximum_outage_seconds is not None
                    ):
                        if time.monotonic() - outage_started >= self.maximum_outage_seconds:
                            self._set_terminal_error(
                                "Camera reconnect outage exceeded the configured limit."
                            )
                            return
                    try:
                        candidate = self._open_capture()
                    except Exception:
                        candidate = None
                    if candidate is not None:
                        with self._state_lock:
                            self._cap = candidate
                    opened = False
                    if candidate is not None:
                        try:
                            opened = bool(candidate.isOpened())
                        except Exception:
                            opened = False
                    if not opened:
                        if candidate is not None:
                            self._release_capture_owned(candidate)
                        if not ever_connected:
                            startup_attempts += 1
                            if startup_attempts >= self.startup_max_attempts:
                                self._set_terminal_error(
                                    "Could not open camera stream after bounded startup retries."
                                )
                                self._open_complete.set()
                                return
                        else:
                            self._set_reconnecting(increment=False)
                        if self._stop_event.wait(reconnect_delay):
                            break
                        reconnect_delay = min(
                            self.reconnect_max_delay_seconds,
                            reconnect_delay * self.reconnect_backoff_multiplier,
                        )
                        continue

                    capture = candidate
                    if self._stop_event.is_set():
                        break
                    ever_connected = True
                    with self._state_lock:
                        self.stream_opened = True
                        if outage_started is None:
                            self.stream_state = "connected"
                            self.stream_warning = None
                    self._open_complete.set()

                try:
                    ok, frame = capture.read()
                except Exception:
                    ok, frame = False, None
                if self._stop_event.is_set():
                    break
                if not ok:
                    self._set_reconnecting(increment=True)
                    outage_started = outage_started or time.monotonic()
                    self._release_capture_owned(capture)
                    capture = None
                    if self._stop_event.wait(reconnect_delay):
                        break
                    reconnect_delay = min(
                        self.reconnect_max_delay_seconds,
                        reconnect_delay * self.reconnect_backoff_multiplier,
                    )
                    continue
                timestamp = utc_now_iso()
                frame_monotonic = time.monotonic()
                outage_started = None
                reconnect_delay = self.reconnect_initial_delay_seconds
                with self._state_lock:
                    frame_idx = self.frames_read
                    self.frames_read += 1
                    if self.first_frame_time is None:
                        self.first_frame_time = timestamp
                    self.last_frame_time = timestamp
                    self._last_frame_monotonic = frame_monotonic
                    self.stream_state = "connected"
                    self.stream_warning = None
                item = BufferedFrame(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    frame=frame,
                    captured_monotonic=frame_monotonic,
                )
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    self.frames_dropped += 1
        except BaseException:
            self._set_terminal_error("Live camera reader stopped unexpectedly.")
            raise
        finally:
            self._open_complete.set()
            print("[LiveFrameBuffer] reader loop exiting", flush=True)
            if capture is not None:
                self._release_capture_owned(capture)
            with self._state_lock:
                if self._stop_event.is_set() and self.error is None:
                    self.stream_state = "stopped"
                    self.stream_warning = None
                elif self.error is None:
                    self.error = "Live camera reader stopped unexpectedly."
                    self.stream_state = "error"
                    self.stream_warning = self.error
                self.ended = True
            self._reader_stopped.set()

    def get(self, timeout: float = 1.0) -> BufferedFrame | None:
        try:
            return self._queue.get(timeout=max(0.01, float(timeout)))
        except queue.Empty:
            return None

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    def stop(self) -> None:
        self.request_stop()
        self.join(self.shutdown_timeout_seconds)

    def request_stop(self) -> None:
        """Signal the sole reader owner without waiting for native I/O."""
        with self._stop_lock:
            print("[LiveFrameBuffer] stop requested", flush=True)
            self._stop_event.set()

    def join(self, timeout_seconds: float | None = None) -> None:
        """Join the reader for a bounded interval after ``request_stop``."""
        timeout = self.shutdown_timeout_seconds if timeout_seconds is None else max(
            0.0, float(timeout_seconds)
        )
        with self._stop_lock:
            with self._state_lock:
                thread = self._thread

            if thread is None or thread is threading.current_thread():
                return

            print("[LiveFrameBuffer] waiting for reader thread", flush=True)
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise LiveFrameBufferLifecycleError(
                    "Live camera reader did not stop within "
                    f"{timeout:g} seconds; "
                    "staging must be preserved."
                )

            print("[LiveFrameBuffer] reader joined", flush=True)
            with self._state_lock:
                if self._thread is thread:
                    self._thread = None

    def stats(self, frames_processed: int, warnings: list[str] | None = None) -> dict:
        with self._state_lock:
            stream_state = self.stream_state
            reconnect_count = self.stream_reconnect_count
            stream_warning = self.stream_warning
            last_frame_monotonic = self._last_frame_monotonic
        last_frame_age = (
            None
            if last_frame_monotonic is None
            else round(max(0.0, time.monotonic() - last_frame_monotonic), 3)
        )
        return {
            "stream_opened": self.stream_opened,
            "frames_read": self.frames_read,
            "frames_processed": int(frames_processed),
            "frames_dropped": self.frames_dropped,
            "buffer_max_size": self.max_size,
            "first_frame_time": self.first_frame_time,
            "last_frame_time": self.last_frame_time,
            "stream_state": stream_state,
            "stream_reconnect_count": reconnect_count,
            "stream_warning": stream_warning,
            "last_frame_age_seconds": last_frame_age,
            "warnings": list(warnings or []),
        }


def write_stream_report(
    output_dir: str | Path,
    *,
    camera_uri: str,
    camera_id: str | None,
    duration_seconds: int,
    stats: dict,
) -> str:
    """Write a report that never persists raw camera credentials."""
    path = Path(output_dir) / "stream_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_type": "live_camera",
        "camera_uri_masked": mask_camera_uri(camera_uri),
        "camera_id": camera_id,
        "duration_seconds": int(duration_seconds),
        **stats,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path)
