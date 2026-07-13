import threading
import numpy as np

from forensics.person_creation.models.device import resolve_device
from forensics.person_creation.utils.profiling import cuda_event_measure, profile_measure


class PersonDetector:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def load(self, model_path: str, device: str = "auto") -> None:
        from ultralytics import YOLO
        device = resolve_device(device)
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device
        print(f"[PersonDetector] loaded {model_path} on {device}")

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        if self._model is None:
            return []
        metadata = {
            "device": self._device,
            "input_shape": list(frame_bgr.shape),
            "batch_size": 1,
        }
        with profile_measure(
            "model.person_detector.total", metadata=metadata, synchronize_cuda=True
        ):
            with self._lock:
                with profile_measure(
                    "model.person_detector.predict", metadata=metadata, synchronize_cuda=True
                ):
                    with cuda_event_measure("model.person_detector.predict.cuda", metadata=metadata):
                        results = self._model.predict(
                            frame_bgr, verbose=False, device=self._device
                        )
            with profile_measure("model.person_detector.postprocess", metadata=metadata):
                if not results or results[0].boxes is None:
                    metadata["output_detection_count"] = 0
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
                metadata["output_detection_count"] = len(detections)
                return detections


_instance = PersonDetector()


def get_person_detector() -> PersonDetector:
    return _instance
