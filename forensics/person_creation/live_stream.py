"""Bounded live-camera capture helpers with credential-safe reporting."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import cv2


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


class LiveFrameBuffer:
    """Continuously read a stream and retain at most the latest N frames."""

    def __init__(
        self,
        uri: str,
        max_size: int = 30,
        open_timeout_ms: int = 10_000,
        read_timeout_ms: int = 5_000,
        shutdown_timeout_seconds: float | None = None,
    ) -> None:
        self.uri = uri
        self.max_size = max(1, int(max_size))
        self.open_timeout_ms = max(1, int(open_timeout_ms))
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        default_shutdown_timeout = (self.read_timeout_ms / 1000.0) + 1.0
        self.shutdown_timeout_seconds = max(
            0.01,
            float(
                default_shutdown_timeout
                if shutdown_timeout_seconds is None
                else shutdown_timeout_seconds
            ),
        )
        self.frames_read = 0
        self.frames_dropped = 0
        self.first_frame_time: str | None = None
        self.last_frame_time: str | None = None
        self.stream_opened = False
        self.error: str | None = None
        self.ended = False
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
            self._thread = threading.Thread(
                target=self._reader_loop,
                name="person-creation-live-reader",
                daemon=True,
            )
            thread = self._thread
        thread.start()

        # Opening belongs to the reader, but preserve the previous synchronous
        # startup error when the backend completes within its configured bound.
        self._open_complete.wait(timeout=(self.open_timeout_ms / 1000.0) + 1.0)
        if self.error and not self.stream_opened:
            raise OSError(self.error)

    def _reader_loop(self) -> None:
        capture = None
        try:
            if self._stop_event.is_set():
                return
            capture = self._open_capture()
            with self._state_lock:
                self._cap = capture
            if self._stop_event.is_set():
                return
            if not capture.isOpened():
                self.error = (
                    "Could not open camera stream. Check URI, network, credentials, "
                    "or FFmpeg/OpenCV support."
                )
                return
            self.stream_opened = True
            self._open_complete.set()

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if self._stop_event.is_set():
                    break
                if not ok:
                    break
                timestamp = utc_now_iso()
                frame_idx = self.frames_read
                self.frames_read += 1
                if self.first_frame_time is None:
                    self.first_frame_time = timestamp
                self.last_frame_time = timestamp
                item = BufferedFrame(frame_idx=frame_idx, timestamp=timestamp, frame=frame)
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
        except Exception:
            self.error = "Camera stream stopped while reading frames."
        finally:
            self._open_complete.set()
            print("[LiveFrameBuffer] reader loop exiting", flush=True)
            if capture is not None:
                print("[LiveFrameBuffer] releasing capture", flush=True)
                try:
                    capture.release()
                except Exception:
                    self.error = self.error or "Camera stream cleanup failed."
                    print("[LiveFrameBuffer] capture release failed", flush=True)
                else:
                    print("[LiveFrameBuffer] capture released", flush=True)
            with self._state_lock:
                if self._cap is capture:
                    self._cap = None
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
        with self._stop_lock:
            print("[LiveFrameBuffer] stop requested", flush=True)
            self._stop_event.set()
            with self._state_lock:
                thread = self._thread

            if thread is None or thread is threading.current_thread():
                return

            print("[LiveFrameBuffer] waiting for reader thread", flush=True)
            thread.join(timeout=self.shutdown_timeout_seconds)
            if thread.is_alive():
                print(
                    "[LiveFrameBuffer] warning: reader thread did not stop within "
                    f"{self.shutdown_timeout_seconds:.2f}s; capture remains owned "
                    "by the reader",
                    flush=True,
                )
                return

            print("[LiveFrameBuffer] reader joined", flush=True)
            with self._state_lock:
                if self._thread is thread:
                    self._thread = None

    def stats(self, frames_processed: int, warnings: list[str] | None = None) -> dict:
        return {
            "stream_opened": self.stream_opened,
            "frames_read": self.frames_read,
            "frames_processed": int(frames_processed),
            "frames_dropped": self.frames_dropped,
            "buffer_max_size": self.max_size,
            "first_frame_time": self.first_frame_time,
            "last_frame_time": self.last_frame_time,
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
