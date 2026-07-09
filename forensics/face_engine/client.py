from __future__ import annotations

import json
import os
import uuid
from typing import Any
from urllib import error, request

from forensics.face_engine import DEFAULT_HOST, DEFAULT_PORT


class FaceEngineClientError(RuntimeError):
    pass


class FaceEngineClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        default_url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
        self.base_url = (base_url or os.getenv("FACE_ENGINE_URL") or default_url).rstrip("/")
        self.timeout = float(timeout)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                message = parsed.get("error", body)
            except json.JSONDecodeError:
                message = body or str(exc)
            raise FaceEngineClientError(f"face_engine HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise FaceEngineClientError(f"face_engine unavailable: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FaceEngineClientError("face_engine returned non-JSON response") from exc

        if parsed.get("ok") is False:
            raise FaceEngineClientError(str(parsed.get("error", "face_engine request failed")))
        return parsed

    def _request_multipart(
        self,
        path: str,
        field_name: str,
        filename: str,
        content_type: str,
        payload: bytes,
    ) -> dict[str, Any]:
        boundary = f"----WAYCON{uuid.uuid4().hex}"
        body = b"".join([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            payload,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ])
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
                message = parsed.get("error", body)
            except json.JSONDecodeError:
                message = body or str(exc)
            raise FaceEngineClientError(f"face_engine HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise FaceEngineClientError(f"face_engine unavailable: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FaceEngineClientError("face_engine returned non-JSON response") from exc

        if parsed.get("ok") is False:
            raise FaceEngineClientError(str(parsed.get("error", "face_engine request failed")))
        return parsed

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def detect(self, image_path: str) -> dict[str, Any]:
        return self._request("POST", "/detect", {"image_path": image_path})

    def detect_bytes(
        self,
        image_bytes: bytes,
        filename: str = "frame.png",
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        return self._request_multipart("/detect-bytes", "image", filename, content_type, image_bytes)

    def embed(self, face_crop_path: str) -> dict[str, Any]:
        return self._request("POST", "/embed", {"face_crop_path": face_crop_path})

    def detect_and_embed(self, image_path: str, save_crops_dir: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"image_path": image_path}
        if save_crops_dir is not None:
            payload["save_crops_dir"] = save_crops_dir
        return self._request("POST", "/detect-and-embed", payload)

    def recognize(
        self,
        face_crop_path: str | None = None,
        embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if bool(face_crop_path) == (embedding is not None):
            raise ValueError("provide exactly one of face_crop_path or embedding")
        payload: dict[str, Any] = {"top_k": int(top_k)}
        if face_crop_path:
            payload["face_crop_path"] = face_crop_path
        else:
            payload["embedding"] = embedding
        return self._request("POST", "/recognize", payload)
