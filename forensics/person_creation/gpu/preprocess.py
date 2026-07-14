"""CUDA-resident preprocessing: NV12→RGB, letterbox, normalization.

All functions operate on torch tensors and never touch host memory when the
input lives on the GPU. No custom CUDA kernels are required; every operation
maps to a handful of cuDNN/elementwise launches (see architecture analysis §6).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Limited-range YUV→RGB coefficients. H.264 HD streams (height >= 720) are
# BT.709 unless the stream says otherwise; SD defaults to BT.601. Validated
# against cv2.VideoCapture output in tools/gpu_decoder_prototype.py.
_YUV_MATRICES = {
    "bt601": ((1.164, 0.000, 1.596), (1.164, -0.392, -0.813), (1.164, 2.017, 0.000)),
    "bt709": ((1.164, 0.000, 1.793), (1.164, -0.213, -0.533), (1.164, 2.112, 0.000)),
}


def default_matrix_for(height: int) -> str:
    return "bt709" if height >= 720 else "bt601"


def nv12_to_rgb(nv12: torch.Tensor, height: int, width: int, matrix: str = "bt709") -> torch.Tensor:
    """Convert an NV12 surface tensor of shape (height*3/2, width) uint8 into an
    owned contiguous RGB tensor of shape (3, height, width) uint8 on the same
    device. This is the single device-to-device copy that detaches the frame
    from the decoder-owned surface pool.
    """
    if nv12.dim() != 2 or nv12.shape[0] != height * 3 // 2 or nv12.shape[1] < width:
        raise ValueError(f"unexpected NV12 shape {tuple(nv12.shape)} for {width}x{height}")
    coef = _YUV_MATRICES[matrix]
    # NVDEC may pad the pitch; slice to the visible width.
    y = nv12[:height, :width].to(torch.float32)
    uv = nv12[height:, :width].reshape(height // 2, width // 2, 2).to(torch.float32)
    # Chroma upsample ×2 (nearest, matching standard NV12→RGB converters).
    u = uv[..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1)
    v = uv[..., 1].repeat_interleave(2, 0).repeat_interleave(2, 1)

    y = y - 16.0
    u = u - 128.0
    v = v - 128.0
    r = coef[0][0] * y + coef[0][2] * v
    g = coef[1][0] * y + coef[1][1] * u + coef[1][2] * v
    b = coef[2][0] * y + coef[2][1] * u
    rgb = torch.stack((r, g, b), dim=0)
    return rgb.round_().clamp_(0.0, 255.0).to(torch.uint8).contiguous()


def rgb_to_bgr(rgb_chw: torch.Tensor) -> torch.Tensor:
    return rgb_chw.flip(0)


def bgr_hwc_to_rgb_chw(frame_bgr_hwc: torch.Tensor) -> torch.Tensor:
    """CPU-decoded fallback frames (H, W, 3) BGR uint8 → (3, H, W) RGB uint8."""
    return frame_bgr_hwc.permute(2, 0, 1).flip(0).contiguous()


def letterbox_gpu(
    img_chw_u8: torch.Tensor,
    new_shape: int | tuple[int, int] = 640,
    stride: int = 32,
    auto: bool = True,
    center: bool = True,
    pad_value: float = 114.0,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """GPU replica of ultralytics LetterBox (scaleup=True, centered padding).

    Input: (3, H, W) uint8 on any device. Returns (1, 3, h, w) float32 in
    [0, 1] ready for Ultralytics tensor-input predict, plus the original
    (H, W) needed to rescale boxes back with ultralytics scale_boxes.
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = int(img_chw_u8.shape[1]), int(img_chw_u8.shape[2])
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (max(1, round(w * r)), max(1, round(h * r)))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride

    img = img_chw_u8.unsqueeze(0).to(torch.float32)
    if (new_unpad[1], new_unpad[0]) != (h, w):
        img = F.interpolate(img, size=(new_unpad[1], new_unpad[0]), mode="bilinear", align_corners=False)
    if center:
        top, left = round(dh / 2 - 0.1), round(dw / 2 - 0.1)
    else:
        top, left = 0, 0
    bottom, right = dh - top, dw - left
    if any(v > 0 for v in (top, bottom, left, right)):
        img = F.pad(img, (left, right, top, bottom), value=pad_value)
    return img.clamp_(0.0, 255.0).div_(255.0), (h, w)
