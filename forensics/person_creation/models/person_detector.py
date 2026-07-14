import threading
import numpy as np

from forensics.person_creation.models.device import resolve_device
from forensics.person_creation.utils.profiling import cuda_event_measure, profile_measure


class PersonDetector:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._imgsz = 640
        self._stride = 32

    def load(self, model_path: str, device: str = "auto") -> None:
        from ultralytics import YOLO
        device = resolve_device(device)
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device
        try:
            self._stride = max(int(self._model.model.stride.max()), 32)
        except Exception:
            self._stride = 32
        imgsz = self._model.overrides.get("imgsz", 640)
        self._imgsz = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
        print(f"[PersonDetector] loaded {model_path} on {device}")

    def warmup(self) -> None:
        """One dummy inference so the first real frame does not pay for CUDA
        context/cuDNN autotune. No-op when the model is not loaded."""
        if self._model is None or self._device == "cpu":
            return
        import torch

        dummy = torch.zeros(3, 384, 640, dtype=torch.uint8, device=self._device)
        self.detect_cuda(dummy)
        torch.cuda.synchronize()
        print("[PersonDetector] warm-up complete")

    def detect_cuda(self, frame_rgb_chw) -> list[dict]:
        """GPU-native detection on a CUDA-resident frame.

        Input: (3, H, W) uint8 RGB torch tensor already on the model device.
        The frame is letterboxed on the GPU and passed to Ultralytics as a
        tensor, which skips its CPU letterbox + host-to-device upload. Output
        format is identical to detect(): [{"bbox": [x1,y1,x2,y2], "score": s}].
        """
        if self._model is None:
            return []
        import torch
        from ultralytics.utils import ops as yolo_ops

        from forensics.person_creation.gpu.preprocess import letterbox_gpu

        metadata = {
            "device": self._device,
            "input_shape": list(frame_rgb_chw.shape),
            "batch_size": 1,
            "input_kind": "cuda_tensor",
        }
        with profile_measure(
            "model.person_detector.total", metadata=metadata, synchronize_cuda=True
        ):
            with torch.inference_mode():
                im, orig_shape = letterbox_gpu(
                    frame_rgb_chw, self._imgsz, stride=self._stride
                )
                with self._lock:
                    with profile_measure(
                        "model.person_detector.predict", metadata=metadata, synchronize_cuda=True
                    ):
                        with cuda_event_measure("model.person_detector.predict.cuda", metadata=metadata):
                            results = self._model.predict(
                                im, verbose=False, device=self._device
                            )
            with profile_measure("model.person_detector.postprocess", metadata=metadata):
                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    metadata["output_detection_count"] = 0
                    return []
                boxes = results[0].boxes
                # Tensor input bypasses Ultralytics' own rescaling: boxes are in
                # letterbox coordinates. Map them back to the original frame.
                xyxy = yolo_ops.scale_boxes(
                    im.shape[2:], boxes.xyxy.clone(), orig_shape
                ).cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                detections = []
                for i in range(len(boxes)):
                    if int(cls[i]) != 0:
                        continue
                    detections.append({
                        "bbox": xyxy[i].tolist(),
                        "score": float(conf[i]),
                    })
                metadata["output_detection_count"] = len(detections)
                return detections

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
