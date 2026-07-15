from __future__ import annotations

import numpy as np

from forensics.face_engine.config import model_cache_path, resolve_device


class FaceEmbedder:
    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = "cpu"

    def load(self, device: str = "auto") -> None:
        import os

        import torch
        from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
        from torchvision import transforms

        cache_dir = model_cache_path()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", str(cache_dir))

        self._device = resolve_device(device)
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
        self._transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            fixed_image_standardization,
        ])
        print(f"[FaceEngine.FaceEmbedder] loaded InceptionResnetV1 vggface2 on {self._device}")

    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def embed(self, crop_bgr: np.ndarray) -> list[float]:
        import cv2
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if self._model is None or self._transform is None:
            raise RuntimeError("face embedder is not loaded")

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).tolist()


_instance = FaceEmbedder()


def get_face_embedder() -> FaceEmbedder:
    return _instance

