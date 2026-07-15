"""Profile body ReID extractor using OSNet-x0.25.

This model is optional. If torchreid is missing or fails to load, the public
methods return None/empty results so enrollment can still complete.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from forensics.person_creation.models.device import model_parameter_device, resolve_device

DEFAULT_REID_CONFIG = {
    "model": "osnet_x0_25",
    "weights": "market1501",
    "input_size": [256, 128],
    "embedding_dim": 512,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_reid_config(config: dict | None = None) -> dict:
    merged = dict(DEFAULT_REID_CONFIG)
    merged.update(config or {})

    size = merged.get("input_size") or DEFAULT_REID_CONFIG["input_size"]
    if len(size) != 2:
        size = DEFAULT_REID_CONFIG["input_size"]
    merged["input_size"] = [int(size[0]), int(size[1])]
    merged["embedding_dim"] = int(merged.get("embedding_dim") or DEFAULT_REID_CONFIG["embedding_dim"])
    merged["model"] = str(merged.get("model") or DEFAULT_REID_CONFIG["model"])
    merged["weights"] = str(merged.get("weights") or DEFAULT_REID_CONFIG["weights"])
    return merged


class ReidExtractor:
    def __init__(self, config: dict | None = None, device: str = "auto") -> None:
        self.config = normalize_reid_config(config)
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._load_attempted = False
        self.unavailable_reason: str | None = None
        self.last_success_count = 0
        self.load(device=device)

    def is_available(self) -> bool:
        return self._model is not None and self._transform is not None

    @property
    def device(self) -> str:
        if not self.is_available():
            return "unavailable" if self._load_attempted else "not_loaded"
        return model_parameter_device(self._model, fallback=self._device)

    def load(self, device: str = "auto") -> None:
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            try:
                import torchreid
                from torchvision import transforms

                self._device = resolve_device(device)
                self._model = torchreid.models.build_model(
                    name=self.config["model"],
                    num_classes=1000,
                    pretrained=True,
                )
                self._model.eval().to(self._device)

                width, height = self.config["input_size"]
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ])
                print(
                    f"[ReidExtractor] {self.config['model']} loaded on {self.device} "
                    f"({self.config['weights']})"
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                self._model = None
                self._transform = None
                self.unavailable_reason = str(exc)
                print(f"[ReidExtractor] torchreid unavailable ({exc}) - profile ReID disabled")

    def _embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if not self.is_available() or image_rgb is None or image_rgb.size == 0:
            return None

        try:
            import torch

            tensor = self._transform(image_rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"[ReidExtractor] embedding failed: {exc}")
            return None

    def compute_embeddings(self, crop_paths: Iterable[str]) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        if not self.is_available():
            self.last_success_count = 0
            return embeddings

        for raw in crop_paths:
            try:
                path = Path(raw).resolve()
                image_bgr = cv2.imread(str(path))
                if image_bgr is None:
                    print(f"[ReidExtractor] crop unreadable: {raw}")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                emb = self._embed_rgb(image_rgb)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as exc:
                print(f"[ReidExtractor] crop failed {raw}: {exc}")

        self.last_success_count = len(embeddings)
        return embeddings

    def compute_mean_embedding(self, crop_paths: list[str]) -> np.ndarray | None:
        embeddings = self.compute_embeddings(crop_paths)
        if not embeddings:
            return None

        mean = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            return None
        return (mean / norm).astype(np.float32)


_instance: ReidExtractor | None = None
_instance_key: tuple | None = None
_instance_lock = threading.Lock()


def _config_key(config: dict) -> tuple:
    return (
        config.get("model"),
        config.get("weights"),
        tuple(config.get("input_size") or []),
        config.get("embedding_dim"),
    )


def get_reid_extractor(config: dict | None = None, device: str = "auto") -> ReidExtractor:
    """Return a process-local OSNet extractor without storing it in graph state."""
    global _instance, _instance_key
    normalized = normalize_reid_config(config)
    key = _config_key(normalized)
    with _instance_lock:
        if _instance is None or _instance_key != key:
            _instance = ReidExtractor(config=normalized, device=device)
            _instance_key = key
        return _instance
