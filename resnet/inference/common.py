# Shared by resnet.inference.pytorch and resnet.inference.trt_adapter.
# Only numpy + Pillow here so it's cheap to import from F´'s process.
from __future__ import annotations

import os
from typing import List

import numpy as np
from PIL import Image

IMAGE_EXT = (".png", ".jpg", ".jpeg")

# microsoft/resnet-* preprocessor_config.json (ConvNextImageProcessor):
#   size.shortest_edge=224, crop_pct=0.875, resample=3 (bicubic),
#   image_mean/std = ImageNet, rescale_factor=1/255
CROP = 224
CROP_PCT = 0.875
RESIZE_SHORT = int(CROP / CROP_PCT)  # 256
RESAMPLE = Image.BICUBIC
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_images(folder: str) -> List[str]:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def preprocess_image(im: Image.Image) -> np.ndarray:
    """PIL image -> float32 [1, 3, 224, 224].

    Bit-for-bit re-implementation of the Hugging Face PIL image processor for
    microsoft/resnet-* (transformers ConvNextImageProcessorPil):
      1. shortest edge -> 256, long edge -> int(256 * long / short), bicubic
      2. center crop 224 x 224, offsets (size - 224) // 2
      3. * 1/255 in float64, cast to float32
      4. (x - mean) / std in float32
    Verified against the HF processor by resnet.inference.selftest.
    """
    im = im.convert("RGB")
    w, h = im.size
    short, long = (w, h) if w <= h else (h, w)
    new_long = int(RESIZE_SHORT * long / short)
    new_w, new_h = (RESIZE_SHORT, new_long) if w <= h else (new_long, RESIZE_SHORT)
    im = im.resize((new_w, new_h), resample=RESAMPLE)
    a = np.asarray(im)  # H, W, 3 uint8
    top, left = (new_h - CROP) // 2, (new_w - CROP) // 2
    a = a[top : top + CROP, left : left + CROP]
    a = (a.astype(np.float64) * (1.0 / 255.0)).astype(np.float32)
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def preprocess_path(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return preprocess_image(im)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64).reshape(-1)
    z = np.exp(z - z.max())
    return z / z.sum()


def label_of(labels: dict, i: int) -> str:
    return labels.get(str(i), labels.get(i, f"class_{i}"))


def format_result(label: str, prob: float) -> str:
    """'cello, violoncello (99.6%)'. SCALES_ML_CONFIDENCE=0 -> label only."""
    if os.environ.get("SCALES_ML_CONFIDENCE", "1") == "0":
        return label
    return f"{label} ({prob * 100:.1f}%)"


def decode(logits: np.ndarray, labels: dict) -> tuple:
    """logits [1, 1000] -> (index, label, probability)."""
    p = softmax(logits)
    i = int(p.argmax())
    return i, label_of(labels, i), float(p[i])
