import cv2
import numpy as np
from pathlib import Path


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


def process_video(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector
    from forensics.person_creation.models.face_detector import get_face_detector

    person_det = get_person_detector()
    face_det = get_face_detector()
    every_n = state.get("process_every_n", 5)
    output_dir = Path(state["output_dir"])

    # Detection writes go to _staging/. promote_crops moves only confirmed
    # crops to body_crops/ and face_crops/ after human-in-the-loop pairing.
    body_dir = output_dir / "_staging" / "body_crops"
    face_dir = output_dir / "_staging" / "face_crops"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

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
                persons = person_det.detect(frame)
                faces = face_det.detect(frame)

                for det_idx, det in enumerate(persons):
                    crop = _crop(frame, det["bbox"])
                    if crop.size == 0:
                        continue
                    sharp = _sharpness(crop)
                    fname = f"{stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg"
                    path = str(body_dir / fname)
                    cv2.imwrite(path, crop)
                    body_crops.append({
                        "path": path,
                        "frame_idx": frame_idx,
                        "video": video_path,
                        "bbox": det["bbox"],
                        "sharpness": sharp,
                    })

                for det_idx, det in enumerate(faces):
                    crop = _crop(frame, det["bbox"])
                    if crop.size == 0:
                        continue
                    sharp = _sharpness(crop)
                    fname = f"{stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg"
                    path = str(face_dir / fname)
                    cv2.imwrite(path, crop)
                    face_crops.append({
                        "path": path,
                        "frame_idx": frame_idx,
                        "video": video_path,
                        "bbox": det["bbox"],
                        "sharpness": sharp,
                    })

            frame_idx += 1

        cap.release()
        print(f"[process_video] {stem}: {frame_idx} frames → {len(body_crops)} body, {len(face_crops)} face crops")
        if frame_idx == 0:
            raise OSError(f"cv2 opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt")

    return {"body_crops": body_crops, "face_crops": face_crops}
