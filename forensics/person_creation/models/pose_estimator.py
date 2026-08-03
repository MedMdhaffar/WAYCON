"""Optional lightweight pose keypoint estimator (RTMPose via `rtmlib`).

Used only as an auxiliary association cue in `nodes/auto_pair.py`, to locate
the anatomical head position inside a body crop instead of assuming a fixed
"head is in the top 20% of the box" heuristic. `rtmlib` is a pure
onnxruntime inference package (no mmcv/mmdet framework install), which keeps
this an easy-to-drop optional dependency: if it isn't installed, `load()`
leaves the model unset and auto_pair disables the pose cue and renormalizes
the remaining cue weights.
"""

from __future__ import annotations

import threading

import numpy as np

from forensics.person_creation.models.device import (
    is_cuda_device,
    normalize_loaded_device,
    resolve_device,
)

# COCO-17 keypoint indices that describe the head.
_HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left/right eye, left/right ear
_MIN_KEYPOINT_SCORE = 0.3


class PoseEstimator:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self._device = "not_loaded"
        self.unavailable_reason: str | None = None

    def is_available(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        if not self.is_available():
            return "unavailable" if self._load_attempted else "not_loaded"
        reported = getattr(self._model, "device", None)
        return normalize_loaded_device(reported, default=self._device)

    def load(self, device: str = "auto") -> None:
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from rtmlib import RTMPose

                device_str = resolve_device(device)
                self._model = RTMPose(
                    model_input_size=(192, 256),
                    backend="onnxruntime",
                    device="cuda" if is_cuda_device(device_str) else "cpu",
                )
                self._device = device_str
                print(f"[PoseEstimator] RTMPose loaded on {self.device}")
            except Exception as exc:  # pragma: no cover - optional dependency
                self._model = None
                self.unavailable_reason = str(exc)
                print(f"[PoseEstimator] rtmlib/RTMPose unavailable ({exc}) — pose cue disabled")

    def head_center(self, body_crop_bgr: np.ndarray) -> tuple[float, float] | None:
        """Return the (x, y) head-keypoint centroid in `body_crop_bgr` pixel
        space, or None if the model is unavailable / no confident head
        keypoints were found.
        """
        if self._model is None or body_crop_bgr is None or body_crop_bgr.size == 0:
            return None
        try:
            keypoints, scores = self._model(body_crop_bgr)
            if keypoints is None or len(keypoints) == 0:
                return None
            kp = keypoints[0]
            kp_scores = scores[0] if scores is not None else None
            pts = []
            for i in _HEAD_KEYPOINTS:
                if i >= len(kp):
                    continue
                if kp_scores is not None and i < len(kp_scores) and kp_scores[i] < _MIN_KEYPOINT_SCORE:
                    continue
                pts.append(kp[i])
            if not pts:
                return None
            arr = np.asarray(pts, dtype=np.float64)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
        except Exception:
            return None


_instance = PoseEstimator()


def get_pose_estimator() -> PoseEstimator:
    return _instance
