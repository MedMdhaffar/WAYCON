import math

import cv2
import numpy as np
from pathlib import Path

from forensics.person_creation.utils.profiling import get_active_profiler, profile_measure


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
    from forensics.face_engine.client import FaceEngineClient

    person_det = get_person_detector()
    face_det = FaceEngineClient()
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
    profiler = get_active_profiler()
    counters = {
        "total_frames_reported_by_video": 0,
        "total_frames_read": 0,
        "total_frames_decoded": 0,
        "selected_frames_processed": 0,
        "expected_selected_frames": 0,
        "skipped_frames": 0,
        "decode_failures": 0,
        "person_detections": 0,
        "face_detections": 0,
        "body_crops_written": 0,
        "face_crops_written": 0,
        "crop_write_failures": 0,
    }
    total_duration_seconds = 0.0
    resolutions: list[str] = []
    source_fps_values: list[float] = []

    for video_path in state["video_paths"]:
        stem = Path(video_path).stem
        video_name = Path(video_path).name
        video_metadata = {"video": video_name}
        bodies_before = len(body_crops)
        faces_before = len(face_crops)
        with profile_measure("video.complete", metadata=video_metadata):
            with profile_measure("video.open", metadata={"video": video_name}):
                cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise OSError(f"cv2 could not open video: {video_path!r}")

            with profile_measure("video.metadata", metadata={"video": video_name}):
                reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = reported_frames / source_fps if source_fps > 0 else 0.0
            counters["total_frames_reported_by_video"] += reported_frames
            if every_n > 0:
                counters["expected_selected_frames"] += math.ceil(reported_frames / every_n)
            total_duration_seconds += duration_seconds
            if source_fps > 0:
                source_fps_values.append(source_fps)
            if width and height:
                resolutions.append(f"{width}x{height}")
            video_metadata.update(
                total_frames_reported=reported_frames,
                source_fps=source_fps,
                width=width,
                height=height,
                source_duration_seconds=duration_seconds,
            )
            frame_idx = 0
            processed_frame_idx = 0

            try:
                while True:
                    frame_meta = {"video": video_name}
                    if profiler is not None and profiler.config.frame_level:
                        frame_meta["frame_index"] = frame_idx
                    with profile_measure("frame.decode", metadata=frame_meta):
                        ret, frame = cap.read()
                    counters["total_frames_read"] += 1
                    if not ret:
                        break
                    counters["total_frames_decoded"] += 1

                    selected = False
                    with profile_measure("frame.skip_decision", metadata=frame_meta):
                        selected = frame_idx % every_n == 0
                    if not selected:
                        counters["skipped_frames"] += 1
                        frame_idx += 1
                        continue

                    counters["selected_frames_processed"] += 1
                    frame_meta.update(
                        processed_frame_index=processed_frame_idx,
                        frame_width=int(frame.shape[1]),
                        frame_height=int(frame.shape[0]),
                    )
                    processed_frame_idx += 1

                    person_meta = dict(frame_meta)
                    with profile_measure(
                        "frame.person_detection",
                        metadata=person_meta,
                        synchronize_cuda=True,
                    ):
                        persons = person_det.detect(frame)
                        person_meta["detection_count"] = len(persons)
                    face_meta = dict(frame_meta)
                    with profile_measure("frame.face_detection", metadata=face_meta):
                        faces = face_det.detect(frame)
                        face_meta["detection_count"] = len(faces)
                    counters["person_detections"] += len(persons)
                    counters["face_detections"] += len(faces)

                    for crop_type, detections, destination, target in (
                        ("body", persons, body_dir, body_crops),
                        ("face", faces, face_dir, face_crops),
                    ):
                        for det_idx, det in enumerate(detections):
                            crop_meta = {**frame_meta, "crop_type": crop_type}
                            with profile_measure("frame.crop_extraction", metadata=crop_meta):
                                crop = _crop(frame, det["bbox"])
                            if crop.size == 0:
                                continue
                            with profile_measure("frame.crop_sharpness", metadata=crop_meta):
                                sharp = _sharpness(crop)
                            if crop_type == "body":
                                fname = f"{stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg"
                            else:
                                fname = f"{stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg"
                            path = str(destination / fname)
                            with profile_measure("frame.crop_write", metadata=crop_meta):
                                write_ok = cv2.imwrite(path, crop)
                            if write_ok:
                                counters[f"{crop_type}_crops_written"] += 1
                            else:
                                counters["crop_write_failures"] += 1
                            with profile_measure("frame.metadata_construction", metadata=crop_meta):
                                target.append({
                                    "path": path,
                                    "frame_idx": frame_idx,
                                    "video": video_path,
                                    "bbox": det["bbox"],
                                    "sharpness": sharp,
                                })
                    frame_idx += 1
            finally:
                cap.release()

            video_metadata.update(
                total_frames_decoded=frame_idx,
                selected_frames_processed=processed_frame_idx,
                body_crops_written=len(body_crops) - bodies_before,
                face_crops_written=len(face_crops) - faces_before,
            )
        print(f"[process_video] {stem}: {frame_idx} frames → {len(body_crops)} body, {len(face_crops)} face crops")
        if frame_idx == 0:
            raise OSError(f"cv2 opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt")

    if profiler is not None:
        profiler.update_run_metadata(
            video_duration_seconds=total_duration_seconds,
            resolutions=sorted(set(resolutions)),
            source_fps_values=source_fps_values,
            total_source_frames=counters["total_frames_reported_by_video"],
            expected_selected_frames=counters["expected_selected_frames"],
            actual_selected_frames=counters["selected_frames_processed"],
            video_counters=counters,
        )

    return {"body_crops": body_crops, "face_crops": face_crops}
