import sys
import types

import numpy as np

from forensics.person_creation.models import pose_estimator


def test_pose_estimator_uses_rtmlib_high_level_body(monkeypatch):
    constructed = {}

    class FakeBody:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def __call__(self, image):
            keypoints = np.zeros((1, 17, 2), dtype=np.float32)
            scores = np.ones((1, 17), dtype=np.float32)
            return keypoints, scores

    monkeypatch.setitem(sys.modules, "rtmlib", types.SimpleNamespace(Body=FakeBody))
    monkeypatch.setattr(pose_estimator, "resolve_device", lambda _device: "cpu")

    estimator = pose_estimator.PoseEstimator()
    estimator.load()

    assert estimator.is_available()
    assert constructed == {
        "mode": "lightweight",
        "to_openpose": False,
        "backend": "onnxruntime",
        "device": "cpu",
    }
    assert estimator.head_center(np.zeros((64, 32, 3), dtype=np.uint8)) == (0.0, 0.0)
