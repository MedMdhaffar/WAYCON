import numpy as np


class FaceEmbedder:
    def __init__(self) -> None:
        self._model = None
        self._transform = None

    def load(self) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
        from torchvision import transforms

        self._model = InceptionResnetV1(pretrained="vggface2").eval()
        self._transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            fixed_image_standardization,
        ])
        print("[FaceEmbedder] loaded InceptionResnetV1 vggface2")

    def embed(self, crop_bgr: np.ndarray) -> list[float]:
        import torch
        import torch.nn.functional as F
        import cv2
        from PIL import Image

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(crop_rgb)).unsqueeze(0)
        with torch.no_grad():
            emb = self._model(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).tolist()


_instance = FaceEmbedder()


def get_face_embedder() -> FaceEmbedder:
    return _instance
