"""Optional body re-identification embedder (OSNet-x0.25 via `torchreid`).

Used only as an auxiliary association cue in `nodes/auto_pair.py`. If
`torchreid` isn't installed, `load()` leaves the model unset and
`is_available()` reports False — auto_pair then drops the ReID cue for this
run and renormalizes the remaining cue weights. Nothing else in the pipeline
depends on this module, so its absence never breaks enrollment.
"""

from __future__ import annotations

import threading

import numpy as np

from forensics.person_creation.models.device import model_parameter_device, resolve_device

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

    @property
    def device(self) -> str:
        if not self.is_available():
            return "unavailable" if self._load_attempted else "not_loaded"
        return model_parameter_device(self._model, fallback=self._device)

    def load(self, device: str = "auto") -> None:
        """Best-effort load. Safe to call even when torchreid isn't installed —
        failures are caught and only disable the ReID cue, never the pipeline.
        """
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
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
                print(f"[BodyReId] OSNet-x0.25 loaded on {self.device}")
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
            rgb = cv2.cvtColor(body_crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(tensor)
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().tolist()
        except Exception:
            return None


_instance = BodyReId()


def get_body_reid() -> BodyReId:
    return _instance
