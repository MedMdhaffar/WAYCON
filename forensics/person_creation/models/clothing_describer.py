import json
import numpy as np
from pathlib import Path

from forensics.person_creation.models.device import resolve_device

_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "clothing.yaml"
_INPUT_SIZE = 448
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_prompt() -> str:
    import yaml
    with open(_PROMPTS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["clothing_description"]
    return " ".join([
        cfg["context"],
        cfg["instruction"],
        cfg["constraints"],
        cfg["detail"],
        cfg["output_format"],
    ])


def _pad_to_square(img):
    from PIL import Image as PILImage
    w, h = img.size
    size = max(w, h)
    canvas = PILImage.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


class ClothingDescriber:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._dtype = None

    def load(self, model_id: str = "OpenGVLab/InternVL3_5-2B", device: str = "auto") -> None:
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._device = resolve_device(device)
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        try:
            import flash_attn
            use_flash = self._device == "cuda"
        except ImportError:
            use_flash = False

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            use_flash_attn=use_flash,
            device_map=self._device,
            trust_remote_code=True,
        ).eval()
        print(f"[ClothingDescriber] loaded {model_id} on {self._device}")

    def _preprocess(self, crop_bgr: np.ndarray):
        import torch
        import cv2
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        t = transforms.Compose([
            transforms.Lambda(_pad_to_square),
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return t(Image.fromarray(rgb)).unsqueeze(0).to(self._device, dtype=self._dtype)

    def describe(self, crops_bgr: list[np.ndarray]) -> tuple[str, dict]:
        import torch

        n = len(crops_bgr)
        pixel_values = torch.cat([self._preprocess(c) for c in crops_bgr], dim=0)

        base_prompt = _load_prompt()
        if n == 1:
            question = f"<image>\n{base_prompt}"
        else:
            img_tags = "".join(f"Image-{i+1}: <image>\n" for i in range(n))
            question = f"{img_tags}{base_prompt}"

        gen_cfg = {"max_new_tokens": 150, "do_sample": False}
        if n == 1:
            response = self._model.chat(self._tokenizer, pixel_values, question, gen_cfg)
        else:
            response = self._model.chat(
                self._tokenizer, pixel_values, question, gen_cfg,
                num_patches_list=[1] * n,
            )

        structured = self._parse(response)
        return response, structured

    @staticmethod
    def _parse(response: str) -> dict:
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        try:
            result = json.loads(text)
            return {
                "top": str(result.get("top", "unknown")),
                "bottom": str(result.get("bottom", "unknown")),
                "shoes": str(result.get("shoes", "unknown")),
                "full": str(result.get("full", response.strip())),
            }
        except Exception:
            return {"top": "unknown", "bottom": "unknown", "shoes": "unknown", "full": response.strip()}


_instance = ClothingDescriber()


def get_clothing_describer() -> ClothingDescriber:
    return _instance
