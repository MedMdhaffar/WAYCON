"""Layer 6/7 — GPU crop extraction, GPU sharpness, GPU JPEG encode.

Crops are CUDA tensor views of the decoded frame until the moment they are
persisted; the only device-to-host transfer is the *compressed* JPEG byte
stream (via torchvision's nvJPEG binding), not raw pixels.

Numerical compatibility notes:
 - sharpness replicates cv2 exactly in structure: BGR→gray rec.601 weights
   with rounding to uint8, 3×3 Laplacian [[0,1,0],[1,-4,1],[0,1,0]] with
   REFLECT_101 border (torch 'reflect'), population variance. Differences vs
   cv2 come only from float32 vs float64 accumulation and are ~1e-3 relative
   (asserted in tests).
 - JPEG bytes differ from cv2.imwrite output (different encoder); quality is
   pinned to 95 to match cv2's default. Equivalence is defined on decoded
   content, not file bytes.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

_LAPLACIAN_KERNEL: dict[tuple, torch.Tensor] = {}


def crop_gpu(frame_rgb_chw: torch.Tensor, bbox: list[float], padding: int = 2) -> torch.Tensor:
    """CUDA equivalent of process_video._crop: integer-cast, padded, clamped
    slice. Returns a view (no copy) of shape (3, h, w)."""
    _, h, w = frame_rgb_chw.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return frame_rgb_chw[:, y1:y2, x1:x2]


def sharpness_gpu(crop_rgb_chw_u8: torch.Tensor) -> float:
    """Variance of the 3×3 Laplacian of the gray crop — cv2-compatible."""
    if crop_rgb_chw_u8.numel() == 0:
        return 0.0
    r, g, b = crop_rgb_chw_u8[0].float(), crop_rgb_chw_u8[1].float(), crop_rgb_chw_u8[2].float()
    gray = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp_(0, 255)
    gray = gray.unsqueeze(0).unsqueeze(0)
    key = (str(gray.device), gray.dtype)
    kernel = _LAPLACIAN_KERNEL.get(key)
    if kernel is None:
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device, dtype=gray.dtype,
        ).reshape(1, 1, 3, 3)
        _LAPLACIAN_KERNEL[key] = kernel
    if gray.shape[2] < 2 or gray.shape[3] < 2:
        return 0.0
    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    lap = F.conv2d(padded, kernel)
    return float(lap.var(unbiased=False).item())


def encode_jpeg_gpu(crop_rgb_chw_u8: torch.Tensor, quality: int = 95) -> bytes:
    """nvJPEG encode on the GPU; only the compressed bytes cross to the host."""
    from torchvision.io import encode_jpeg

    data = encode_jpeg(crop_rgb_chw_u8.contiguous(), quality=quality)
    return data.cpu().numpy().tobytes()


def write_crop_jpeg(path: str | Path, crop_rgb_chw_u8: torch.Tensor, use_gpu_jpeg: bool = True) -> bool:
    """Persist one accepted crop. GPU path: nvJPEG encode + byte write.
    Fallback: device-to-host copy + cv2.imwrite (identical to CPU pipeline)."""
    path = Path(path)
    try:
        if use_gpu_jpeg and crop_rgb_chw_u8.is_cuda:
            data = encode_jpeg_gpu(crop_rgb_chw_u8)
            path.write_bytes(data)
            return True
        import cv2

        bgr_hwc = crop_rgb_chw_u8.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
        return bool(cv2.imwrite(str(path), bgr_hwc))
    except Exception as exc:
        print(f"[gpu.crops] failed to write {path}: {exc!r}")
        return False
