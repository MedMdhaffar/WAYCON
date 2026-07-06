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


def _body_name(stem: str, frame_idx: int, det_idx: int) -> str:
    return f"{stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg"


def _face_name(stem: str, frame_idx: int, det_idx: int) -> str:
    return f"{stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg"


def _run_detection_pass(video_paths, every_n, out_dir, name_fn, detector) -> list[dict]:
    """Single detector over all videos. Used in low-memory mode so only one
    heavy model is resident at a time."""
    crops: list[dict] = []
    for video_path in video_paths:
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
                dets = detector.detect(frame)
                for det_idx, det in enumerate(dets):
                    crop = _crop(frame, det["bbox"])
                    if crop.size == 0:
                        continue
                    sharp = _sharpness(crop)
                    fname = name_fn(stem, frame_idx, det_idx)
                    path = str(out_dir / fname)
                    cv2.imwrite(path, crop)
                    crops.append({
                        "path": path,
                        "frame_idx": frame_idx,
                        "video": video_path,
                        "bbox": det["bbox"],
                        "sharpness": sharp,
                    })

            frame_idx += 1

        cap.release()
        if frame_idx == 0:
            raise OSError(f"cv2 opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt")
    return crops


def _run_combined_pass(video_paths, every_n, body_dir, face_dir, person_det, face_det) -> tuple[list[dict], list[dict]]:
    """Both detectors over each frame in one read pass. Faster but keeps both
    models resident on GPU simultaneously — only used when LOW_MEMORY_MODE
    is disabled."""
    body_crops: list[dict] = []
    face_crops: list[dict] = []

    for video_path in video_paths:
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
                    path = str(body_dir / _body_name(stem, frame_idx, det_idx))
                    cv2.imwrite(path, crop)
                    body_crops.append({
                        "path": path, "frame_idx": frame_idx, "video": video_path,
                        "bbox": det["bbox"], "sharpness": sharp,
                    })

                for det_idx, det in enumerate(faces):
                    crop = _crop(frame, det["bbox"])
                    if crop.size == 0:
                        continue
                    sharp = _sharpness(crop)
                    path = str(face_dir / _face_name(stem, frame_idx, det_idx))
                    cv2.imwrite(path, crop)
                    face_crops.append({
                        "path": path, "frame_idx": frame_idx, "video": video_path,
                        "bbox": det["bbox"], "sharpness": sharp,
                    })

            frame_idx += 1

        cap.release()
        if frame_idx == 0:
            raise OSError(f"cv2 opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt")

    return body_crops, face_crops


def process_video(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector, release_person_detector
    from forensics.person_creation.models.face_detector import get_face_detector, release_face_detector
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom
    from forensics.person_creation import config

    _YOLO_MODEL_PATH = str(Path(__file__).parents[4] / "yolo26m.pt")

    every_n = state.get("process_every_n", 5)
    output_dir = Path(state["output_dir"])

    # Detection writes go to _staging/. promote_crops moves only confirmed
    # crops to body_crops/ and face_crops/ after human-in-the-loop pairing.
    body_dir = output_dir / "_staging" / "body_crops"
    face_dir = output_dir / "_staging" / "face_crops"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    video_paths = state["video_paths"]

    if config.LOW_MEMORY_MODE:
        # Safest mode: one detector resident on GPU at a time.
        person_det = get_person_detector()
        log_memory("before loading person_detector")
        try:
            person_det.load(model_path=_YOLO_MODEL_PATH, device=config.DETECTOR_DEVICE)
            log_memory("after loading person_detector")
            body_crops = _run_detection_pass(video_paths, every_n, body_dir, _body_name, person_det)
        except Exception as exc:
            raise clarify_oom(exc, "process_video/person_detector") from exc
        finally:
            release_person_detector()
            cleanup_memory("process_video/person_detector")

        face_det = get_face_detector()
        log_memory("before loading face_detector")
        try:
            face_det.load(device=config.DETECTOR_DEVICE)
            log_memory("after loading face_detector")
            face_crops = _run_detection_pass(video_paths, every_n, face_dir, _face_name, face_det)
        except Exception as exc:
            raise clarify_oom(exc, "process_video/face_detector") from exc
        finally:
            release_face_detector()
            cleanup_memory("process_video/face_detector")
    else:
        person_det = get_person_detector()
        face_det = get_face_detector()
        log_memory("before loading detectors")
        try:
            person_det.load(model_path=_YOLO_MODEL_PATH, device=config.DETECTOR_DEVICE)
            face_det.load(device=config.DETECTOR_DEVICE)
            log_memory("after loading detectors")
            body_crops, face_crops = _run_combined_pass(
                video_paths, every_n, body_dir, face_dir, person_det, face_det
            )
        except Exception as exc:
            raise clarify_oom(exc, "process_video") from exc
        finally:
            release_person_detector()
            release_face_detector()
            cleanup_memory("process_video")

    print(f"[process_video] {len(video_paths)} videos → {len(body_crops)} body, {len(face_crops)} face crops")
    return {"body_crops": body_crops, "face_crops": face_crops}
