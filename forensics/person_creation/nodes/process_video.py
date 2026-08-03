import json
import re

import cv2
import numpy as np
from pathlib import Path
from typing import Any


_REQUIRED_QUALITY_FIELDS = (
    "path",
    "frame_idx",
    "bbox",
    "confidence",
    "sharpness",
)
_CAMERA_URI_RE = re.compile(r"rtsps?://\S+", re.IGNORECASE)


def _sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop(frame: np.ndarray, bbox: list[float], padding: int = 2) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame[y1:y2, x1:x2]


def prepare_staging_dirs(output_dir: str | Path) -> tuple[Path, Path]:
    """Create and return the crop staging directories shared by all sources."""
    staging = Path(output_dir) / "_staging"
    body_dir = staging / "body_crops"
    face_dir = staging / "face_crops"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    return body_dir, face_dir


def _write_crop(recorded_path: str, crop: np.ndarray) -> str:
    """Persist one crop and verify that its record names the written file."""
    imwrite_path = str(Path(recorded_path))
    write_ok = bool(cv2.imwrite(imwrite_path, crop))
    exists = Path(imwrite_path).exists()
    if not write_ok:
        raise OSError(f"cv2.imwrite failed for crop {Path(imwrite_path).name}")
    if recorded_path != imwrite_path:
        raise RuntimeError("Recorded crop path differs from the cv2.imwrite path.")
    if not exists:
        raise OSError(
            f"Crop write reported success but file is missing: {Path(imwrite_path).name}"
        )
    return recorded_path


def log_crop_record_example(label: str, record: dict | None) -> None:
    """Log one credential-safe crop schema example for source comparison."""
    if not record:
        return
    example = {
        key: (Path(str(record[key])).name if key == "path" else record.get(key))
        for key in _REQUIRED_QUALITY_FIELDS
        if key in record
    }
    example["source_type"] = record.get("source_type")
    for key in ("video", "video_path", "source_uri"):
        if key in record:
            example[key] = _CAMERA_URI_RE.sub("<camera-source>", str(record.get(key)))
    missing = [key for key in _REQUIRED_QUALITY_FIELDS if key not in record]
    null = [key for key in _REQUIRED_QUALITY_FIELDS if record.get(key) is None]
    print(
        f"[crop_schema] {label} crop record: "
        f"{json.dumps(example, sort_keys=True)} missing={missing} null={null}"
    )


def detect_and_save_frame(
    frame: np.ndarray,
    *,
    frame_idx: int,
    source_stem: str,
    source_metadata: dict[str, Any],
    body_dir: Path,
    face_dir: Path,
    person_detector,
    face_detector,
) -> tuple[list[dict], list[dict]]:
    """Run the shared detectors and persist crop records in the video format."""
    body_crops: list[dict] = []
    face_crops: list[dict] = []
    persons = person_detector.detect(frame)
    faces = face_detector.detect(frame)

    for det_idx, det in enumerate(persons):
        crop = _crop(frame, det["bbox"])
        if crop.size == 0:
            continue
        fname = f"{source_stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg"
        path = str(body_dir / fname)
        _write_crop(path, crop)
        body_crops.append({
            "path": path,
            "frame_idx": frame_idx,
            "bbox": det["bbox"],
            "confidence": float(det.get("confidence", det.get("score", 0.0))),
            "sharpness": _sharpness(crop),
            **source_metadata,
        })

    for det_idx, det in enumerate(faces):
        crop = _crop(frame, det["bbox"])
        if crop.size == 0:
            continue
        fname = f"{source_stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg"
        path = str(face_dir / fname)
        _write_crop(path, crop)
        face_crops.append({
            "path": path,
            "frame_idx": frame_idx,
            "bbox": det["bbox"],
            "confidence": float(det.get("confidence", det.get("score", 0.0))),
            "sharpness": _sharpness(crop),
            **source_metadata,
        })

    return body_crops, face_crops


def process_video(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.face_engine.client import FaceEngineClient

    person_det = get_person_detector()
    face_det = FaceEngineClient()
    every_n = state.get("process_every_n", 5)
    body_dir, face_dir = prepare_staging_dirs(state["output_dir"])

    body_crops: list[dict] = []
    face_crops: list[dict] = []

    for video_path in state["video_paths"]:
        stem = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"cv2 could not open video: {video_path!r}")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                bodies, detected_faces = detect_and_save_frame(
                    frame,
                    frame_idx=frame_idx,
                    source_stem=stem,
                    source_metadata={
                        "source_type": "video_file",
                        "video": video_path,
                        "video_path": video_path,
                    },
                    body_dir=body_dir,
                    face_dir=face_dir,
                    person_detector=person_det,
                    face_detector=face_det,
                )
                body_crops.extend(bodies)
                face_crops.extend(detected_faces)

            frame_idx += 1

        cap.release()
        print(f"[process_video] {stem}: {frame_idx} frames → {len(body_crops)} body, {len(face_crops)} face crops")
        if frame_idx == 0:
            raise OSError(f"cv2 opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt")

    log_crop_record_example("video face", face_crops[0] if face_crops else None)
    return {
        "body_crops": body_crops,
        "face_crops": face_crops,
        "source_type": "video_file",
    }
