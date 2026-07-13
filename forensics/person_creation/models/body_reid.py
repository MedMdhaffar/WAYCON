"""Optional body re-identification embedder (OSNet-x0.25 via `torchreid`).

Used only as an auxiliary association cue in `nodes/auto_pair.py`. If
`torchreid` isn't installed, `load()` leaves the model unset and
`is_available()` reports False — auto_pair then drops the ReID cue for this
run and renormalizes the remaining cue weights. Nothing else in the pipeline
depends on this module, so its absence never breaks enrollment.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np

from forensics.person_creation.models.device import resolve_device
from forensics.person_creation.utils.profiling import cuda_event_measure, profile_measure

_INPUT_HW = (256, 128)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class BodyReId:
    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._load_attempted = False
        self.unavailable_reason: str | None = None

    def is_available(self) -> bool:
        return self._model is not None

    def load(self, device: str = "auto") -> None:
        """Best-effort load. Safe to call even when torchreid isn't installed —
        failures are caught and only disable the ReID cue, never the pipeline.
        """
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                # torchreid imports its optional Cython ranking evaluator even
                # though this adapter only uses the OSNet model. Suppress that
                # irrelevant performance warning without hiding other warnings.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Cython evaluation .* is unavailable.*",
                        category=UserWarning,
                        module=r"torchreid\.reid\.metrics\.rank",
                    )
                    import torchreid
                from torchvision import transforms

                self._device = resolve_device(device)
                self._model = torchreid.models.build_model(
                    name="osnet_x0_25", num_classes=1000, pretrained=True
                )
                self._model.eval().to(self._device)
                self._transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(_INPUT_HW),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ])
                print(f"[BodyReId] OSNet-x0.25 loaded on {self._device}")
            except Exception as exc:  # pragma: no cover - optional dependency
                self._model = None
                self.unavailable_reason = str(exc)
                print(f"[BodyReId] torchreid unavailable ({exc}) — body ReID cue disabled")

    def embed(self, body_crop_bgr: np.ndarray) -> list[float] | None:
        if self._model is None or body_crop_bgr is None or body_crop_bgr.size == 0:
            return None
        import torch
        import cv2

        try:
            metadata = {"device": self._device, "model": "osnet_x0_25", "batch_size": 1}
            with profile_measure("model.body_reid.total", metadata=metadata, synchronize_cuda=True):
                with profile_measure("model.body_reid.preprocess", metadata=metadata):
                    rgb = cv2.cvtColor(body_crop_bgr, cv2.COLOR_BGR2RGB)
                    tensor = self._transform(rgb).unsqueeze(0)
                    metadata["tensor_shape"] = list(tensor.shape)
                with profile_measure("model.body_reid.host_to_device", metadata=metadata, synchronize_cuda=True):
                    tensor = tensor.to(self._device)
                with profile_measure("model.body_reid.inference", metadata=metadata, synchronize_cuda=True):
                    with cuda_event_measure("model.body_reid.inference.cuda", metadata=metadata):
                        with torch.no_grad():
                            feat = self._model(tensor)
                with profile_measure("model.body_reid.postprocess", metadata=metadata, synchronize_cuda=True):
                    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                    return feat.squeeze(0).cpu().tolist()
        except Exception:
            return None


_instance = BodyReId()


def get_body_reid() -> BodyReId:
    return _instance
