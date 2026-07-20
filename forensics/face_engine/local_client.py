"""In-process face detection/embedding, replacing FaceEngineClient's HTTP round-trip.

Same call surface as `face_engine.client.FaceEngineClient` (detect/embed/recognize/
ensure_healthy/health) so every existing call site can swap the import with no other
change, but talks to the `FaceDetector`/`FaceEmbedder` singletons directly -- no HTTP
request, no JPEG re-encode/decode per crop, no separate face_engine process to keep
alive for the realtime path.

`face_engine.client.FaceEngineClient` (HTTP) is kept for anyone who wants to run
face_engine as a standalone service (e.g. a non-Python consumer, or querying it
remotely) -- it's just no longer on the person_creation pipeline's critical path.
"""

from __future__ import annotations

import numpy as np

from forensics.face_engine import config
from forensics.face_engine.models.detector import get_face_detector
from forensics.face_engine.models.embedder import get_face_embedder


class LocalFaceEngine:
    """Drop-in, in-process replacement for FaceEngineClient.

    Cheap to instantiate repeatedly (mirrors FaceEngineClient's usage pattern of one
    instance per call site) since it only holds references to the process-wide
    detector/embedder singletons -- `.load()` on those singletons is itself guarded
    against reloading, so calling `ensure_healthy()`/`load()` from multiple call sites
    is safe and only does real work once per process.
    """

    def __init__(self, device: str | None = None) -> None:
        self._detector = get_face_detector()
        self._embedder = get_face_embedder()
        self._device = device or config.DEVICE

    def load(self) -> None:
        if not self._detector.is_loaded():
            self._detector.load(device=self._device)
        if not self._embedder.is_loaded():
            self._embedder.load(device=self._device)

    def ensure_healthy(self) -> None:
        if not (self._detector.is_loaded() and self._embedder.is_loaded()):
            self.load()
        print(
            "[face_engine] using in-process LocalFaceEngine "
            f"(device={self._embedder.device if self._embedder.is_loaded() else self._device})"
        )

    def health(self) -> dict:
        models_loaded = self._detector.is_loaded() and self._embedder.is_loaded()
        return {
            "service": "waycon-face-engine-inprocess",
            "api_version": 1,
            "status": "ok" if models_loaded else "loading",
            "device": self._embedder.device if models_loaded else self._device,
            "models_loaded": models_loaded,
        }

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        faces = self._detector.detect(image_bgr)
        return [
            {
                "bbox": face.get("bbox", []),
                "score": float(face.get("confidence", 0.0)),
                "confidence": float(face.get("confidence", 0.0)),
            }
            for face in faces
        ]

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        vec = np.asarray(self._embedder.embed(crop_bgr), dtype=np.float32).reshape(-1)
        if vec.shape[0] != 512:
            raise ValueError(f"face embedder returned {vec.shape[0]}-d embedding, expected 512")
        return vec

    def recognize(self, embedding, top_k: int = 5, threshold: float | None = None) -> dict:
        from forensics.global_memory import GlobalMemory
        from forensics.global_memory.config import SIMILARITY_THRESHOLD

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        gm = GlobalMemory()
        try:
            matches = gm.query_by_face(
                vec.astype(float).tolist(),
                top_k=int(top_k),
                threshold=float(threshold) if threshold is not None else SIMILARITY_THRESHOLD,
            )
        finally:
            gm.close()

        compact = [
            {"person_id": item["person_id"], "name": item["name"], "similarity": item["similarity"]}
            for item in matches
        ]
        return {"matches": compact, "recognized": bool(compact)}
