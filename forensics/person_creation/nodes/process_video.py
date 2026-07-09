import os
from pathlib import Path

import cv2
import numpy as np


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


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _face_engine_detect(client, frame: np.ndarray) -> list[dict]:
    success, buf = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("failed to encode frame as PNG")

    result = client.detect_bytes(buf.tobytes())
    faces = result.get("faces")
    if faces is None:
        raise ValueError("face_engine returned no faces list")

    normalized = []
    for face in faces:
        bbox = face.get("bbox")
        if not bbox:
            raise ValueError("face_engine returned face without bbox")
        normalized.append({
            "bbox": [float(v) for v in bbox],
            "score": float(face.get("confidence", face.get("score", 0.0))),
        })
    return normalized


def process_video(state: dict) -> dict:
    from forensics.person_creation.models.person_detector import get_person_detector

    person_det = get_person_detector()
    face_det = None
    use_face_engine = _enabled("PERSON_CREATION_USE_FACE_ENGINE")
    fallback_local = _enabled("FACE_ENGINE_FALLBACK_LOCAL")
    face_engine_client = None
    fallback_warned = False
    service_failed = False

    if use_face_engine:
        from forensics.face_engine.client import FaceEngineClient

        face_engine_client = FaceEngineClient()
        print("[process_video] using face_engine for face detection")

    def local_face_detect(frame: np.ndarray) -> list[dict]:
        nonlocal face_det
        if face_det is None:
            from forensics.person_creation.models.face_detector import get_face_detector

            face_det = get_face_detector()
        return face_det.detect(frame)

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
                if use_face_engine and not service_failed:
                    try:
                        faces = _face_engine_detect(face_engine_client, frame)
                    except Exception as exc:
                        if not fallback_local:
                            raise RuntimeError(
                                f"face_engine face detection failed and local fallback is disabled: {exc}"
                            ) from exc
                        service_failed = True
                        if not fallback_warned:
                            print(
                                "[process_video] warning: face_engine face detection failed; "
                                f"falling back to local detector: {exc}"
                            )
                            fallback_warned = True
                        faces = local_face_detect(frame)
                else:
                    faces = local_face_detect(frame)

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
