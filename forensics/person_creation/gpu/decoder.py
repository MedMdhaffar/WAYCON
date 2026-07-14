"""Layer 1 — video decoder abstraction.

Backends:
    NvdecDecoder   — NVDEC hardware decode via PyNvVideoCodec; frames are
                     GPU-resident NV12 surfaces exposed to torch through DLPack
                     (zero host copies).
    OpenCVDecoder  — the existing cv2.VideoCapture CPU path, wrapped in the
                     same interface (fallback).

`create_decoder()` performs capability detection and **always logs** which
backend was selected and why — no silent fallback.

Frame lifetime contract (NVDEC): `DecodedFrame.tensor` for pixel_format
"nv12" is a DLPack *view* of a decoder-owned surface pool and is only valid
until the next `read()` call. Convert it to an owned tensor immediately with
`to_rgb_chw()` (one device-to-device copy) before reading further frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch

from forensics.person_creation.gpu import env_flag
from forensics.person_creation.gpu.preprocess import (
    bgr_hwc_to_rgb_chw,
    default_matrix_for,
    nv12_to_rgb,
)


@dataclass
class DecodedFrame:
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    pixel_format: str  # "nv12" (GPU) | "bgr24" (CPU)
    device: str  # "cuda" | "cpu"
    tensor: torch.Tensor | None
    cuda_surface: object | None = field(default=None, repr=False)
    color_matrix: str = "bt709"

    def to_rgb_chw(self) -> torch.Tensor:
        """Owned, contiguous (3, H, W) uint8 RGB tensor on this frame's device."""
        if self.pixel_format == "nv12":
            return nv12_to_rgb(self.tensor, self.height, self.width, self.color_matrix)
        if self.pixel_format == "bgr24":
            return bgr_hwc_to_rgb_chw(self.tensor)
        raise ValueError(f"unsupported pixel_format {self.pixel_format!r}")

    def to_bgr_numpy(self) -> np.ndarray:
        """CPU BGR ndarray (H, W, 3) — for CPU-only consumers. Device-to-host
        transfer when the frame is GPU-resident; free for CPU frames."""
        if self.pixel_format == "bgr24":
            return self.tensor.numpy()
        rgb = self.to_rgb_chw()
        return rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


class VideoDecoder(ABC):
    backend: str = "abstract"

    @abstractmethod
    def open(self, source: str) -> None: ...

    @abstractmethod
    def read(self) -> DecodedFrame | None: ...

    @abstractmethod
    def close(self) -> None: ...

    # Stream metadata — populated by open().
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0  # 0 when unknown (e.g. live streams)

    def __enter__(self) -> "VideoDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class NvdecDecoder(VideoDecoder):
    """NVDEC via PyNvVideoCodec SimpleDecoder (sequential access)."""

    backend = "nvdec"

    def __init__(self, gpu_id: int = 0) -> None:
        self._gpu_id = gpu_id
        self._decoder = None
        self._index = 0
        self._matrix = "bt709"

    def open(self, source: str) -> None:
        import PyNvVideoCodec as nvc

        self._decoder = nvc.SimpleDecoder(
            str(source), gpu_id=self._gpu_id, use_device_memory=True
        )
        meta = self._decoder.get_stream_metadata()
        self.width = int(meta.width)
        self.height = int(meta.height)
        self.fps = float(meta.average_fps)
        self.frame_count = int(meta.num_frames)
        self.codec = str(getattr(meta, "codec_name", "unknown"))
        self._matrix = default_matrix_for(self.height)
        self._index = 0

    def read(self) -> DecodedFrame | None:
        if self._decoder is None:
            raise RuntimeError("decoder is not open")
        if self.frame_count and self._index >= self.frame_count:
            return None
        try:
            surface = self._decoder[self._index]
        except (IndexError, RuntimeError):
            return None
        tensor = torch.from_dlpack(surface)
        timestamp = getattr(surface, "timestamp", None)
        if not timestamp or timestamp < 0:
            timestamp = self._index / self.fps if self.fps > 0 else 0.0
        frame = DecodedFrame(
            frame_index=self._index,
            timestamp_seconds=float(timestamp),
            width=self.width,
            height=self.height,
            pixel_format="nv12",
            device="cuda",
            tensor=tensor,
            cuda_surface=surface,
            color_matrix=self._matrix,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._decoder = None


class OpenCVDecoder(VideoDecoder):
    """The existing CPU decode path behind the common interface."""

    backend = "opencv-cpu"

    def __init__(self) -> None:
        self._cap = None
        self._index = 0

    def open(self, source: str) -> None:
        import cv2

        self._cap = cv2.VideoCapture(str(source))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise OSError(f"cv2 could not open video: {source!r}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._index = 0

    def read(self) -> DecodedFrame | None:
        if self._cap is None:
            raise RuntimeError("decoder is not open")
        ret, frame_bgr = self._cap.read()
        if not ret:
            return None
        frame = DecodedFrame(
            frame_index=self._index,
            timestamp_seconds=self._index / self.fps if self.fps > 0 else 0.0,
            width=int(frame_bgr.shape[1]),
            height=int(frame_bgr.shape[0]),
            pixel_format="bgr24",
            device="cpu",
            tensor=torch.from_numpy(frame_bgr),
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def nvdec_available() -> tuple[bool, str]:
    """Capability probe with a human-readable reason."""
    try:
        import torch as _torch

        if not _torch.cuda.is_available():
            return False, "torch.cuda is not available"
    except Exception as exc:  # pragma: no cover
        return False, f"torch import failed: {exc}"
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception as exc:
        return False, f"PyNvVideoCodec import failed: {exc} (pip install PyNvVideoCodec)"
    return True, "ok"


def create_decoder(source: str, use_nvdec: bool | None = None) -> VideoDecoder:
    """Open `source` with the best available backend. Logs the decision."""
    if use_nvdec is None:
        use_nvdec = env_flag("PERSON_CREATION_USE_NVDEC", default=True)

    if use_nvdec:
        ok, reason = nvdec_available()
        if ok:
            decoder = NvdecDecoder()
            try:
                decoder.open(source)
                print(
                    f"[gpu.decoder] backend=nvdec (PyNvVideoCodec) source={source} "
                    f"{decoder.width}x{decoder.height}@{decoder.fps:.2f} "
                    f"frames={decoder.frame_count} codec={decoder.codec}"
                )
                return decoder
            except Exception as exc:
                print(
                    f"[gpu.decoder] NVDEC open failed for {source!r}: {exc!r} — "
                    "falling back to CPU decode"
                )
        else:
            print(f"[gpu.decoder] NVDEC unavailable ({reason}) — falling back to CPU decode")
    else:
        print("[gpu.decoder] NVDEC disabled by PERSON_CREATION_USE_NVDEC=0 — CPU decode")

    decoder = OpenCVDecoder()
    decoder.open(source)
    print(
        f"[gpu.decoder] backend=opencv-cpu source={source} "
        f"{decoder.width}x{decoder.height}@{decoder.fps:.2f} frames={decoder.frame_count}"
    )
    return decoder
