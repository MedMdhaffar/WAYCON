import threading
import numpy as np

from forensics.person_creation.models.device import resolve_device


def _patch_fuse():
    try:
        from ultralytics.nn.modules.conv import Conv
        Conv.default_act = __import__("torch").nn.SiLU()
    except Exception:
        pass

    try:
        import ultralytics.nn.tasks as _tasks
        _orig_fuse = _tasks.DetectionModel.fuse

        def _safe_fuse(self, verbose=True):
            try:
                return _orig_fuse(self, verbose)
            except Exception:
                return self

        _tasks.DetectionModel.fuse = _safe_fuse
    except Exception:
        pass


class FaceDetector:
    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._lock = threading.Lock()
        self._conf = 0.5

<<<<<<< HEAD
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str = "cuda") -> None:
        if self._model is not None:
            return
=======
    def load(self, device: str = "auto") -> None:
>>>>>>> Khalifa_branch
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        device = resolve_device(device)
        _patch_fuse()
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
        )
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device
        print(f"[FaceDetector] loaded YOLOv8-Face on {device}")

    def unload(self) -> None:
        if self._model is not None:
            from forensics.person_creation.utils.model_lifecycle import move_to_cpu
            move_to_cpu(self._model)
            self._model = None
        self._device = None

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        if self._model is None:
            return []
        with self._lock:
            results = self._model.predict(
                frame_bgr, conf=self._conf, verbose=False, device=self._device
            )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                "score": float(boxes.conf[i].item()),
            })
        return detections


_instance = FaceDetector()


def get_face_detector() -> FaceDetector:
    return _instance


def release_face_detector() -> None:
    from forensics.person_creation.utils.memory import cleanup_memory
    _instance.unload()
    cleanup_memory("release_face_detector")
