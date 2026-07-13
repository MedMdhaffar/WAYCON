from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np

from forensics.face_engine.config import BASE_URL, REQUEST_TIMEOUT
from forensics.person_creation.utils.profiling import profile_measure


class FaceEngineConnectionError(ConnectionError):
    pass


class FaceEngineClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.timeout = REQUEST_TIMEOUT if timeout is None else float(timeout)

    def health(self) -> dict:
        return self._request("GET", "/health")

    def ensure_healthy(self) -> None:
        data = self.health()
        if not data.get("models_loaded"):
            raise FaceEngineConnectionError(
                f"Face engine at {self.base_url} is reachable but models are not loaded."
            )

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        metadata = {"input_shape": list(image_bgr.shape), "endpoint": "/detect"}
        with profile_measure("model.face_detector.total", metadata=metadata):
            with profile_measure("model.face_detector.jpeg_encode", metadata=metadata):
                files = self._image_files(image_bgr)
            with profile_measure("model.face_detector.remote_request", metadata=metadata):
                data = self._request("POST", "/detect", files=files)
            with profile_measure("model.face_detector.postprocess", metadata=metadata):
                faces = data.get("faces", [])
                result = [
                    {
                        "bbox": face.get("bbox", []),
                        "score": float(face.get("confidence", face.get("score", 0.0))),
                        "confidence": float(face.get("confidence", face.get("score", 0.0))),
                    }
                    for face in faces
                ]
                metadata["output_detection_count"] = len(result)
                return result

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        metadata = {"input_shape": list(crop_bgr.shape), "endpoint": "/embed"}
        with profile_measure("model.face_embedder.total", metadata=metadata):
            with profile_measure("model.face_embedder.jpeg_encode", metadata=metadata):
                files = self._image_files(crop_bgr)
            with profile_measure("model.face_embedder.remote_request", metadata=metadata):
                data = self._request("POST", "/embed", files=files)
            with profile_measure("model.face_embedder.postprocess", metadata=metadata):
                vec = np.asarray(data.get("embedding"), dtype=np.float32).reshape(-1)
                if vec.shape[0] != 512:
                    raise ValueError(f"face engine returned {vec.shape[0]}-d embedding, expected 512")
                return vec

    def recognize(self, embedding: list[float] | np.ndarray, top_k: int = 5, threshold: float | None = None) -> dict:
        payload: dict[str, Any] = {
            "embedding": np.asarray(embedding, dtype=np.float32).reshape(-1).astype(float).tolist(),
            "top_k": int(top_k),
        }
        if threshold is not None:
            payload["threshold"] = float(threshold)
        return self._request("POST", "/recognize", json=payload)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            import requests

            res = requests.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except Exception as exc:
            raise FaceEngineConnectionError(
                f"Face engine is not reachable at {self.base_url}. "
                "Start it with: python -m forensics.face_engine.app"
            ) from exc

        try:
            data = res.json()
        except ValueError as exc:
            raise RuntimeError(f"face engine returned non-JSON response from {path}: HTTP {res.status_code}") from exc

        if not res.ok:
            raise RuntimeError(data.get("error") or f"face engine request failed: HTTP {res.status_code}")
        return data

    @staticmethod
    def _image_files(image_bgr: np.ndarray) -> dict:
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise ValueError("failed to encode image for face engine request")
        return {"image": ("image.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")}

