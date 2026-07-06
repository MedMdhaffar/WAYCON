import numpy as np


class FaceEmbedder:
    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = "cpu"

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str = "cpu") -> None:
        if self._model is not None:
            return
        from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
        from torchvision import transforms

        self._device = device
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self._transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            fixed_image_standardization,
        ])
        print(f"[FaceEmbedder] loaded InceptionResnetV1 vggface2 on {device}")

    def unload(self) -> None:
        if self._model is not None:
            from forensics.person_creation.utils.model_lifecycle import move_to_cpu
            move_to_cpu(self._model)
            self._model = None
        self._transform = None

    def embed(self, crop_bgr: np.ndarray) -> list[float]:
        import torch
        import torch.nn.functional as F
        import cv2
        from PIL import Image

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            emb = self._model(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu().tolist()


_instance = FaceEmbedder()


def get_face_embedder() -> FaceEmbedder:
    return _instance


def release_face_embedder() -> None:
    from forensics.person_creation.utils.memory import cleanup_memory
    _instance.unload()
    cleanup_memory("release_face_embedder")
