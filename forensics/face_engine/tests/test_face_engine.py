from __future__ import annotations

import io

import numpy as np
import pytest

from forensics.face_engine.app import create_app
from forensics.face_engine.client import FaceEngineClient, FaceEngineConnectionError


class FakeDetector:
    device = "cpu"

    def __init__(self, faces=None):
        self.faces = faces if faces is not None else [
            {"bbox": [1.0, 2.0, 20.0, 22.0], "confidence": 0.91},
        ]

    def is_loaded(self):
        return True

    def detect(self, image):
        return self.faces


class FakeEmbedder:
    device = "cpu"

    def is_loaded(self):
        return True

    def embed(self, image):
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()


def _jpg_bytes() -> bytes:
    import cv2

    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def client():
    app = create_app(load_models=False)
    app.config["FACE_DETECTOR"] = FakeDetector()
    app.config["FACE_EMBEDDER"] = FakeEmbedder()
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["models_loaded"] is True
    assert data["device"] == "cpu"


def test_detect_single_face(client):
    res = client.post(
        "/detect",
        data={"image": (io.BytesIO(_jpg_bytes()), "face.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["count"] == 1
    face = data["faces"][0]
    assert len(face["bbox"]) == 4
    assert 0.0 < face["confidence"] <= 1.0


def test_detect_no_face():
    app = create_app(load_models=False)
    app.config["FACE_DETECTOR"] = FakeDetector(faces=[])
    app.config["FACE_EMBEDDER"] = FakeEmbedder()
    app.config["TESTING"] = True
    client = app.test_client()

    res = client.post(
        "/detect",
        data={"image": (io.BytesIO(_jpg_bytes()), "blank.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    assert res.get_json() == {"faces": [], "count": 0}


def test_embed_shape_and_norm(client):
    res = client.post(
        "/embed",
        data={"image": (io.BytesIO(_jpg_bytes()), "crop.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["dim"] == 512
    assert np.linalg.norm(np.asarray(data["embedding"], dtype=np.float32)) == pytest.approx(1.0, abs=1e-5)


def test_embed_bad_input(client):
    res = client.post(
        "/embed",
        data={"image": (io.BytesIO(b"not an image"), "bad.jpg")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_recognize_wrong_dim(client):
    res = client.post("/recognize", json={"embedding": [0.1] * 128})
    assert res.status_code == 422
    assert "expected 512" in res.get_json()["error"]


def test_recognize_unknown(client, tmp_path, monkeypatch):
    from forensics.global_memory import config as gm_config

    monkeypatch.setattr(gm_config, "DB_PATH", str(tmp_path / "memory.db"))
    vec = [0.0] * 512
    vec[0] = 1.0
    res = client.post("/recognize", json={"embedding": vec, "threshold": 0.9})
    assert res.status_code == 200
    assert res.get_json() == {"matches": [], "recognized": False}


def test_client_engine_down():
    client = FaceEngineClient(base_url="http://127.0.0.1:9", timeout=0.1)
    with pytest.raises(FaceEngineConnectionError, match="Start it with"):
        client.health()

