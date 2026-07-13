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
        userinfo, hostinfo = parts.netloc.rsplit("@", 1)
        if ":" in userinfo:
            username = userinfo.split(":", 1)[0]
            masked_netloc = f"{username}:****@{hostinfo}"

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
    ) -> None:
        self.uri = uri
        self.max_size = max(1, int(max_size))
        self.open_timeout_ms = max(1, int(open_timeout_ms))
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        self.frames_read = 0
        self.frames_dropped = 0
        self.first_frame_time: str | None = None
        self.last_frame_time: str | None = None
        self.stream_opened = False
        self.error: str | None = None
        self.ended = False
        self._queue: queue.Queue[BufferedFrame] = queue.Queue(maxsize=self.max_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None

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
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise OSError(
                "Could not open camera stream. Check URI, network, credentials, "
                "or FFmpeg/OpenCV support."
            )
        self.stream_opened = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="person-creation-live-reader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    self.ended = True
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
            self.ended = True
        finally:
            if self._cap is not None:
                self._cap.release()

    def get(self, timeout: float = 1.0) -> BufferedFrame | None:
        try:
            return self._queue.get(timeout=max(0.01, float(timeout)))
        except queue.Empty:
            return None

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    def stop(self) -> None:
        self._stop_event.set()
        if self._cap is not None:
            self._cap.release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

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
