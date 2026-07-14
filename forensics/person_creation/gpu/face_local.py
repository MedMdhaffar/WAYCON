"""Layer 5 — in-process CUDA face detection.

The current path JPEG-encodes the full frame and POSTs it to the Face Engine
(HTTP), which JPEG-decodes and uploads it to the GPU a second time. This
module loads the *same* YOLOv8-Face weights (same HF repo, same confidence
threshold) in the pipeline process and accepts CUDA tensors directly, so a
GPU-resident frame never leaves the device for face detection.

The HTTP FaceEngineClient remains the compatibility fallback — selected by
`PERSON_CREATION_LOCAL_FACE=0` or automatically when this detector cannot
load (the caller logs the reason).

Output format matches FaceEngineClient.detect():
    [{"bbox": [x1, y1, x2, y2], "score": s, "confidence": s}]
"""

from __future__ import annotations

import threading

import torch

from forensics.face_engine.models.detector import FaceDetector
from forensics.person_creation.gpu.preprocess import letterbox_gpu
from forensics.person_creation.utils.profiling import cuda_event_measure, profile_measure


class LocalFaceDetector(FaceDetector):
    """FaceDetector (YOLOv8-face) with a CUDA-tensor input path."""

    def warmup(self) -> None:
        if self._model is None or self._device == "cpu":
            return
        dummy = torch.zeros(3, 384, 640, dtype=torch.uint8, device=self._device)
        self.detect_cuda(dummy)
        torch.cuda.synchronize()
        print("[LocalFaceDetector] warm-up complete")

    def detect_cuda(self, frame_rgb_chw: torch.Tensor) -> list[dict]:
        """Full-frame face detection on a (3, H, W) uint8 RGB CUDA tensor."""
        if self._model is None:
            raise RuntimeError("face detector is not loaded")
        metadata = {
            "device": self._device,
            "input_shape": list(frame_rgb_chw.shape),
            "input_kind": "cuda_tensor",
        }
        with profile_measure("model.face_detector.total", metadata=metadata, synchronize_cuda=True):
            from ultralytics.utils import ops as yolo_ops

            with torch.inference_mode():
                im, orig_shape = letterbox_gpu(frame_rgb_chw, 640, stride=32)
                with self._lock:
                    with cuda_event_measure("model.face_detector.predict.cuda", metadata=metadata):
                        results = self._model.predict(
                            im, conf=self._conf, verbose=False, device=self._device
                        )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                metadata["output_detection_count"] = 0
                return []
            boxes = results[0].boxes
            xyxy = yolo_ops.scale_boxes(im.shape[2:], boxes.xyxy.clone(), orig_shape).cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            detections = [
                {"bbox": xyxy[i].tolist(), "score": float(conf[i]), "confidence": float(conf[i])}
                for i in range(len(boxes))
            ]
            metadata["output_detection_count"] = len(detections)
            return detections

    def detect_rois_cuda(
        self,
        frame_rgb_chw: torch.Tensor,
        person_boxes: list[list[float]],
        padding: int = 8,
    ) -> list[dict]:
        """ROI-batched detection: run the face model only inside person boxes,
        batched as one letterboxed 640x640 batch. Returned bboxes are in
        full-frame coordinates.

        NOTE: this changes recall characteristics vs full-frame detection
        (faces outside person boxes are missed; small faces gain resolution).
        It is provided for the real-time path and is NOT used by default in
        the batch pipeline, which must stay equivalent to the CPU path.
        """
        if self._model is None:
            raise RuntimeError("face detector is not loaded")
        if not person_boxes:
            return []
        _, h, w = frame_rgb_chw.shape
        rois, origins, scales = [], [], []
        with torch.inference_mode():
            for bbox in person_boxes:
                x1 = max(0, int(bbox[0]) - padding)
                y1 = max(0, int(bbox[1]) - padding)
                x2 = min(w, int(bbox[2]) + padding)
                y2 = min(h, int(bbox[3]) + padding)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = frame_rgb_chw[:, y1:y2, x1:x2]
                im, orig_shape = letterbox_gpu(roi, 640, auto=False)
                rois.append(im)
                origins.append((x1, y1))
                scales.append(orig_shape)
            if not rois:
                return []
            batch = torch.cat(rois, dim=0)
            with self._lock:
                results = self._model.predict(
                    batch, conf=self._conf, verbose=False, device=self._device
                )
        from ultralytics.utils import ops as yolo_ops

        detections: list[dict] = []
        for result, (ox, oy), orig_shape in zip(results, origins, scales):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = yolo_ops.scale_boxes((640, 640), result.boxes.xyxy.clone(), orig_shape)
            conf = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(xyxy.cpu().numpy()):
                detections.append({
                    "bbox": [float(box[0] + ox), float(box[1] + oy),
                             float(box[2] + ox), float(box[3] + oy)],
                    "score": float(conf[i]),
                    "confidence": float(conf[i]),
                })
        return detections


_instance = LocalFaceDetector()
_load_lock = threading.Lock()


def get_local_face_detector() -> LocalFaceDetector:
    return _instance


def load_local_face_detector(device: str = "auto") -> LocalFaceDetector:
    """Load (once) and return the singleton. Raises on failure — callers decide
    whether to fall back to the HTTP Face Engine and must log that decision."""
    with _load_lock:
        if not _instance.is_loaded():
            _instance.load(device=device)
        return _instance
