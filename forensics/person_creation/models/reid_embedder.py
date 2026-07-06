import importlib.util
from importlib import import_module
from pathlib import Path
from typing import Sequence

import numpy as np


REID_MODEL_NAME = "OSNet_x1_0"
_TORCHREID_MODEL_NAME = "osnet_x1_0"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _load_feature_extractor_class():
    candidates = (
        "torchreid.utils",
        "torchreid.reid.utils",
        "torchreid.reid.utils.feature_extractor",
    )
    errors = []
    for module_name in candidates:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        feature_extractor = getattr(module, "FeatureExtractor", None)
        if feature_extractor is not None:
            return feature_extractor
        errors.append(f"{module_name}: FeatureExtractor not found")

    raise RuntimeError(
        "torchreid is installed, but no compatible FeatureExtractor import path "
        "was found. Tried: "
        + "; ".join(errors)
    )


class ReIDEmbedder:
    """OSNet person ReID wrapper.

    ReID is a same-day supporting body-appearance signal. Face embedding remains
    the permanent biometric identity anchor for this project.
    """

    MODEL_NAME = REID_MODEL_NAME

    def __init__(self) -> None:
        self._extractor = None
        self._device = _default_device()

    def load(self, device: str | None = None) -> None:
        if self._extractor is not None:
            return

        self._device = device or _default_device()
        if importlib.util.find_spec("torchreid") is None:
            raise RuntimeError(
                "torchreid is required for OSNet_x1_0 ReID embeddings. "
                "Install it with `pip install torchreid` or "
                "`pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`."
            )
        FeatureExtractor = _load_feature_extractor_class()

        self._extractor = FeatureExtractor(
            model_name=_TORCHREID_MODEL_NAME,
            model_path="",
            device=self._device,
        )
        print(f"[ReIDEmbedder] loaded {self.MODEL_NAME} on {self._device}")

    def _valid_paths(self, paths: Sequence[str]) -> list[str]:
        valid = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.exists():
                valid.append(str(path.resolve()))
            else:
                print(f"[ReIDEmbedder][warn] missing crop skipped: {raw_path}")
        return valid

    def embed_image(self, path: str) -> list[float] | None:
        embeddings = self.embed_batch([path])
        return embeddings[0] if embeddings else None

    def embed_batch(self, paths: Sequence[str]) -> list[list[float]]:
        self.load(self._device)
        valid_paths = self._valid_paths(paths)
        if not valid_paths:
            return []

        features = self._extractor(valid_paths)
        try:
            features_np = features.detach().cpu().numpy()
        except AttributeError:
            features_np = np.asarray(features)

        embeddings: list[list[float]] = []
        for raw_embedding in np.asarray(features_np, dtype=np.float32):
            normalized = _normalize(raw_embedding)
            if normalized is not None:
                embeddings.append(normalized.astype(float).tolist())
        return embeddings


_instance = ReIDEmbedder()


def get_reid_embedder() -> ReIDEmbedder:
    return _instance
