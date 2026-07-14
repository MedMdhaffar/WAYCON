"""GPU-first process_video node (behind PERSON_CREATION_GPU_PIPELINE=1).

Data path per selected frame:
    NVDEC decode (GPU, NV12) → NV12→RGB (GPU, one D2D copy)
    → person detect_cuda (GPU letterbox + tensor inference)
    → face detect_cuda (in-process, same weights as the Face Engine)
    → GPU crop views → GPU sharpness → nvJPEG encode
    → device-to-host of *compressed bytes only* → write to _staging/

Device-to-host transfers that remain (by design, all measured):
    - detection boxes/scores (a few hundred bytes per frame);
    - compressed JPEG bytes of persisted crops;
    - inside Ultralytics predict: the letterboxed network input is copied back
      to host once per call to build Results.orig_img (~0.9 MB at 384×640) —
      removing it requires bypassing predict() and is documented follow-up;
    - full-frame D2H only when a CPU fallback is active (logged).

The state contract is identical to the CPU node: crops land in
_staging/{body,face}_crops/ and entries are {path, frame_idx, video, bbox,
sharpness}, so filter_quality → … → finalize run unchanged.

With PERSON_CREATION_FINAL_ONLY_CROPS=1 the filter_quality thresholds are
applied on the GPU *before* persistence, so crops that filter_quality would
discard are never written to disk (they never appear in state either — the
post-filter_quality state is identical because the same thresholds are used).
"""

from __future__ import annotations

import math
from pathlib import Path

from forensics.person_creation.gpu import env_flag
from forensics.person_creation.utils.profiling import get_active_profiler, profile_measure

_warmed_up = False


def _get_face_backend():
    """Return (mode, backend). mode is 'local' or 'http'. Never silent."""
    from forensics.face_engine.client import FaceEngineClient

    if env_flag("PERSON_CREATION_LOCAL_FACE", default=True):
        try:
            from forensics.person_creation.gpu.face_local import load_local_face_detector

            detector = load_local_face_detector(device="auto")
            print("[process_video_gpu] face backend: local CUDA (in-process YOLOv8-face)")
            return "local", detector
        except Exception as exc:
            print(
                f"[process_video_gpu] local face detector failed to load ({exc!r}) — "
                "falling back to HTTP Face Engine"
            )
    else:
        print("[process_video_gpu] face backend: HTTP Face Engine (PERSON_CREATION_LOCAL_FACE=0)")
    return "http", FaceEngineClient()


def process_video_gpu(state: dict) -> dict:
    import torch

    from forensics.person_creation.gpu.crops import crop_gpu, sharpness_gpu, write_crop_jpeg
    from forensics.person_creation.gpu.decoder import create_decoder
    from forensics.person_creation.models.person_detector import get_person_detector

    global _warmed_up

    person_det = get_person_detector()
    face_mode, face_backend = _get_face_backend()
    use_gpu_jpeg = env_flag("PERSON_CREATION_GPU_JPEG", default=True)
    final_only = env_flag("PERSON_CREATION_FINAL_ONLY_CROPS", default=False)
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        print("[process_video_gpu] WARNING: CUDA unavailable — running with CPU tensors")

    quality_gate = None
    if final_only:
        from forensics.person_creation.nodes.filter_quality import _body_ok, _face_ok

        quality_gate = {"body": _body_ok, "face": _face_ok}
        print("[process_video_gpu] final-only crops: filter_quality gate applied before write")

    if not _warmed_up and cuda_ok:
        with profile_measure("model.warmup"):
            person_det.warmup()
            if face_mode == "local":
                face_backend.warmup()
        _warmed_up = True

    every_n = state.get("process_every_n", 5)
    output_dir = Path(state["output_dir"])
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
        "crops_rejected_by_quality_gate": 0,
        "gpu_frames": 0,
        "cpu_fallback_frames": 0,
    }
    total_duration_seconds = 0.0
    resolutions: list[str] = []
    source_fps_values: list[float] = []
    decoder_backends: list[str] = []

    for video_path in state["video_paths"]:
        stem = Path(video_path).stem
        video_name = Path(video_path).name
        video_metadata = {"video": video_name, "pipeline": "gpu"}
        bodies_before = len(body_crops)
        faces_before = len(face_crops)
        with profile_measure("video.complete", metadata=video_metadata):
            with profile_measure("video.open", metadata={"video": video_name}):
                decoder = create_decoder(video_path)
            decoder_backends.append(decoder.backend)
            video_metadata["decoder_backend"] = decoder.backend

            reported_frames = decoder.frame_count
            source_fps = decoder.fps
            duration_seconds = reported_frames / source_fps if source_fps > 0 else 0.0
            counters["total_frames_reported_by_video"] += reported_frames
            if every_n > 0:
                counters["expected_selected_frames"] += math.ceil(reported_frames / every_n)
            total_duration_seconds += duration_seconds
            if source_fps > 0:
                source_fps_values.append(source_fps)
            if decoder.width and decoder.height:
                resolutions.append(f"{decoder.width}x{decoder.height}")
            video_metadata.update(
                total_frames_reported=reported_frames,
                source_fps=source_fps,
                width=decoder.width,
                height=decoder.height,
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
                        frame = decoder.read()
                    counters["total_frames_read"] += 1
                    if frame is None:
                        break
                    counters["total_frames_decoded"] += 1

                    selected = frame_idx % every_n == 0
                    if not selected:
                        counters["skipped_frames"] += 1
                        frame_idx += 1
                        continue

                    counters["selected_frames_processed"] += 1
                    frame_meta.update(
                        processed_frame_index=processed_frame_idx,
                        frame_width=frame.width,
                        frame_height=frame.height,
                    )
                    processed_frame_idx += 1

                    # GPU-resident RGB tensor. NVDEC frames convert on-device;
                    # CPU-decoded fallback frames pay one recorded H2D upload.
                    if frame.device == "cuda":
                        counters["gpu_frames"] += 1
                        with profile_measure("frame.nv12_to_rgb", metadata=frame_meta, synchronize_cuda=True):
                            rgb = frame.to_rgb_chw()
                    else:
                        counters["cpu_fallback_frames"] += 1
                        with profile_measure("frame.h2d_upload", metadata=frame_meta, synchronize_cuda=True):
                            rgb = frame.to_rgb_chw()
                            if cuda_ok:
                                rgb = rgb.cuda(non_blocking=True)

                    person_meta = dict(frame_meta)
                    with profile_measure(
                        "frame.person_detection", metadata=person_meta, synchronize_cuda=True
                    ):
                        persons = person_det.detect_cuda(rgb)
                        person_meta["detection_count"] = len(persons)

                    face_meta = dict(frame_meta)
                    with profile_measure("frame.face_detection", metadata=face_meta):
                        if face_mode == "local":
                            faces = face_backend.detect_cuda(rgb)
                        else:
                            with profile_measure("frame.face_d2h_fallback", metadata=face_meta):
                                frame_bgr = frame.to_bgr_numpy()
                            faces = face_backend.detect(frame_bgr)
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
                                crop = crop_gpu(rgb, det["bbox"])
                            if crop.numel() == 0:
                                continue
                            with profile_measure("frame.crop_sharpness", metadata=crop_meta, synchronize_cuda=True):
                                sharp = sharpness_gpu(crop)
                            record = {
                                "frame_idx": frame_idx,
                                "video": video_path,
                                "bbox": det["bbox"],
                                "sharpness": sharp,
                            }
                            if quality_gate is not None and not quality_gate[crop_type](record):
                                counters["crops_rejected_by_quality_gate"] += 1
                                continue
                            if crop_type == "body":
                                fname = f"{stem}_f{frame_idx:06d}_b{det_idx:02d}.jpg"
                            else:
                                fname = f"{stem}_face_f{frame_idx:06d}_f{det_idx:02d}.jpg"
                            path = str(destination / fname)
                            with profile_measure("frame.crop_write", metadata=crop_meta):
                                write_ok = write_crop_jpeg(path, crop, use_gpu_jpeg=use_gpu_jpeg)
                            if write_ok:
                                counters[f"{crop_type}_crops_written"] += 1
                            else:
                                counters["crop_write_failures"] += 1
                            with profile_measure("frame.metadata_construction", metadata=crop_meta):
                                target.append({"path": path, **record})
                    frame_idx += 1
            finally:
                decoder.close()

            video_metadata.update(
                total_frames_decoded=frame_idx,
                selected_frames_processed=processed_frame_idx,
                body_crops_written=len(body_crops) - bodies_before,
                face_crops_written=len(face_crops) - faces_before,
            )
        print(
            f"[process_video_gpu] {stem} ({decoder.backend}): {frame_idx} frames → "
            f"{len(body_crops)} body, {len(face_crops)} face crops"
        )
        if frame_idx == 0:
            raise OSError(
                f"decoder opened but decoded 0 frames from {video_path!r} — codec / file may be corrupt"
            )

    if profiler is not None:
        profiler.update_run_metadata(
            video_duration_seconds=total_duration_seconds,
            resolutions=sorted(set(resolutions)),
            source_fps_values=source_fps_values,
            total_source_frames=counters["total_frames_reported_by_video"],
            expected_selected_frames=counters["expected_selected_frames"],
            actual_selected_frames=counters["selected_frames_processed"],
            video_counters=counters,
            gpu_pipeline=True,
            decoder_backends=decoder_backends,
            face_backend=face_mode,
            gpu_jpeg=use_gpu_jpeg,
            final_only_crops=final_only,
        )

    return {"body_crops": body_crops, "face_crops": face_crops}
