# ResNet adapter for scales_trt.worker: tag, export_onnx, preprocess, postprocess
from __future__ import annotations

import json
import os

import numpy as np

from resnet.inference.common import CROP, decode, format_result, preprocess_path


def tag(model: str) -> str:
    return model.replace("/", "--")


def preprocess(path: str) -> np.ndarray:
    # Same pixels the HF processor gives resnet.inference.pytorch (bicubic resize)
    return preprocess_path(path)


def export_onnx(model: str, onnx_path: str, labels_path: str) -> None:
    import torch
    from transformers import ResNetForImageClassification

    hf = ResNetForImageClassification.from_pretrained(model).eval()
    with open(labels_path, "w") as f:
        json.dump(hf.config.id2label, f)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).logits

    w = Wrap(hf).eval()
    dummy = torch.randn(1, 3, CROP, CROP)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    kw = dict(opset_version=17, input_names=["input"], output_names=["logits"])
    with torch.no_grad():
        try:
            # TorchScript exporter: single self-contained file, no onnxscript needed
            torch.onnx.export(w, dummy, onnx_path, dynamo=False, **kw)
        except TypeError:  # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(w, dummy, onnx_path, **kw)


def postprocess(y: np.ndarray, labels: dict) -> str:
    _, label, prob = decode(y, labels)
    return format_result(label, prob)
