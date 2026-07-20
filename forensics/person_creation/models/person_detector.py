import threading
import numpy as np

from forensics.person_creation.models.device import resolve_device


class PersonDetector:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def load(self, model_path: str, device: str = "auto") -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        device = resolve_device(device)
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device
        print(f"[PersonDetector] loaded {model_path} on {device}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        if self._model is None:
            return []
        with self._lock:
            results = self._model.predict(frame_bgr, verbose=False, device=self._device)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            if int(boxes.cls[i].item()) != 0:
                continue
            detections.append({
                "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                "score": float(boxes.conf[i].item()),
            })
        return detections


_instance = PersonDetector()


def get_person_detector() -> PersonDetector:
    return _instance
