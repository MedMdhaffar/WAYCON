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

from forensics.person_creation.models.device import resolve_device
from forensics.person_creation.utils.profiling import profile_measure

# COCO-17 keypoint indices that describe the head.
_HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left/right eye, left/right ear
_MIN_KEYPOINT_SCORE = 0.3


class PoseEstimator:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self.unavailable_reason: str | None = None

    def is_available(self) -> bool:
        return self._model is not None

    def load(self, device: str = "auto") -> None:
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from rtmlib import Body

                device_str = resolve_device(device)
                # Body is rtmlib's high-level image -> (keypoints, scores)
                # API. The low-level RTMPose class requires an explicit
                # onnx_model and bounding boxes.
                self._model = Body(
                    mode="lightweight",
                    to_openpose=False,
                    backend="onnxruntime",
                    device="cuda" if device_str == "cuda" else "cpu",
                )
                print(f"[PoseEstimator] RTMPose Body loaded on {device_str}")
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
            with profile_measure(
                "model.pose_estimator.total",
                metadata={"input_shape": list(body_crop_bgr.shape)},
                synchronize_cuda=True,
            ):
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
