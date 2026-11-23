'''
F' drop-in ResNet image classification (Hugging Face Transformers + PyTorch, CUDA).

    SET_ML_PATH "resnet.inference.pytorch"

Contract:
  main(folder_path, model_name=...) -> list[(filename, "label (confidence%)")]

F' runs in fprime-venv (Python 3.12), where no CUDA build of torch exists for
Jetson. main() therefore runs this same module as a child process under the
GPU interpreter (ml-venv, Python 3.10, NVIDIA's Jetson torch) via
scales_trt.fprime_wrap, unless it is already running there.

Command line:
  python -m resnet.inference.pytorch <folder>            # from any interpreter
  python -m resnet.inference.pytorch --worker <folder>   # in-process, JSON on stdout
'''
from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Tuple

from scales_trt.fprime_wrap import is_gpu_interpreter, run_module

MODEL = os.environ.get("SCALES_RESNET_MODEL", "microsoft/resnet-18")


def load_model_and_processor(model_name=MODEL):
    from transformers import AutoImageProcessor, ResNetForImageClassification

    try:
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_name, backend="pil")
        except (TypeError, ValueError):  # transformers < 5
            image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        model = ResNetForImageClassification.from_pretrained(model_name).eval()
    except OSError as e:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError(
                f"{model_name} is not in the Hugging Face cache of user {os.environ.get('USER', '?')} "
                "and HF_HUB_OFFLINE=1. Run `make jetson-setup` (or `make ml-setup`) as the service user, "
                "or set HF_HUB_OFFLINE=0 to allow downloading."
            ) from e
        raise
    return model, image_processor


def _print_benchmark(device, n, total_s, per_image_s, filenames, load_s):
    if os.environ.get("SCALES_ML_BENCH", "1") == "0" or n == 0 or total_s <= 0:
        return
    ms = [t * 1000.0 for t in per_image_s]
    lines = [
        "--- ResNet-PyTorch ---",
        "  device: %s   images: %d   model load: %.2f s" % (device, n, load_s),
        "  total: %.3f s   avg FPS: %.1f" % (total_s, n / total_s),
        "  per-image time (ms):  min=%.1f  max=%.1f  mean=%.1f" % (min(ms), max(ms), sum(ms) / n),
    ]
    if n <= 20:
        lines.append("  per file:")
        for name, t in zip(filenames, ms):
            name = name if len(name) <= 40 else name[:37] + "..."
            lines.append("    %-42s  %6.1f FPS  (%5.1f ms)" % (name, 1000.0 / t if t else 0.0, t))
    lines.append("---")
    print("\n".join(lines), file=sys.stderr, flush=True)


def run_local(folder_path: str, model_name: str = MODEL, return_details: bool = False):
    """Classify every image in folder_path in this process."""
    import torch
    from PIL import Image

    from resnet.inference.common import decode, format_result, list_images
    from scales_trt.cuda_check import select_device

    filenames = list_images(folder_path)
    if not filenames:
        print("[ResNet-PyTorch] No images in: " + folder_path, file=sys.stderr, flush=True)
        return ([], []) if return_details else []

    device = select_device()  # raises on a Jetson without working CUDA
    t_load = time.perf_counter()
    model, image_processor = load_model_and_processor(model_name)
    model = model.to(device)
    labels = model.config.id2label
    with torch.inference_mode():  # warm-up: CUDA context, cuDNN kernel selection
        model(pixel_values=torch.zeros(1, 3, 224, 224, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load

    output, details, per_image_s = [], [], []
    for filename in filenames:
        t0 = time.perf_counter()
        with Image.open(os.path.join(folder_path, filename)) as im:
            pixel_values = image_processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            logits = model(pixel_values=pixel_values.to(device)).logits
        logits = logits.float().cpu().numpy()  # implicit sync
        idx, label, prob = decode(logits, labels)
        output.append((filename, format_result(label, prob)))
        details.append({"file": filename, "index": idx, "label": label, "prob": prob})
        per_image_s.append(time.perf_counter() - t0)

    _print_benchmark(device, len(output), sum(per_image_s), per_image_s, filenames, load_s)
    return (output, details) if return_details else output


def main(folder_path, model_name=MODEL) -> List[Tuple[str, str]]:
    folder_path = os.path.abspath(folder_path)
    if is_gpu_interpreter():
        return run_local(folder_path, model_name)
    p = run_module("resnet.inference.pytorch", ["--worker", folder_path, "--model", model_name])
    return [(str(a), str(b)) for a, b in p.get("outputs", [])]


def _worker(folder: str, model_name: str) -> int:
    try:
        t0 = time.time()
        outputs = run_local(folder, model_name)
        dt = time.time() - t0
        print(json.dumps({"ok": True, "outputs": outputs, "stats": {"images": len(outputs), "seconds": dt}}), flush=True)
        return 0
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)
        return 2


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m resnet.inference.pytorch")
    ap.add_argument("folder", nargs="?", default="../test-imagery")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--worker", action="store_true", help="run in-process, print one JSON line")
    a = ap.parse_args()
    if a.worker:
        raise SystemExit(_worker(os.path.abspath(a.folder), a.model))
    for n, c in main(a.folder, a.model):
        print(f"{n}: {c}")
