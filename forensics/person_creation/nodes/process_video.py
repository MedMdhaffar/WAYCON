import cv2
import numpy as np
from pathlib import Path
from typing import Any


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
        if not cv2.imwrite(path, crop):
            continue
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
        if not cv2.imwrite(path, crop):
            continue
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
    from forensics.face_engine.local_client import LocalFaceEngine

    person_det = get_person_detector()
    face_det = LocalFaceEngine()
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

    return {
        "body_crops": body_crops,
        "face_crops": face_crops,
        "source_type": "video_file",
    }
