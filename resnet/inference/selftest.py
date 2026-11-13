'''
ResNet accuracy/consistency self-test. Run under the GPU interpreter:

    ml-venv/bin/python -m resnet.inference.selftest <image_folder> [--no-trt] [--json]

Checks, per image:
  1. preprocess  - resnet.inference.common.preprocess_image (used by TensorRT)
                   matches the Hugging Face image processor (used by PyTorch)
  2. cuda        - PyTorch on CUDA gives the same top-1 as PyTorch on CPU
  3. golden      - top-1 matches the known-good labels for test-imagery/
  4. tensorrt    - the cached TensorRT engine gives the same top-1 as PyTorch
                   (skipped with --no-trt or when no engine is built)
Exit code 0 only if every check that ran passed.
'''
from __future__ import annotations

import json
import os
import sys
import time

# Known-good top-1 ImageNet indices for fprime-scales-ref/test-imagery with
# microsoft/resnet-18 + the HF reference processor (recorded 2026-11-13, CPU).
GOLDEN = {
    "ANYA_01-07-24.jpg": "Dandie Dinmont, Dandie Dinmont terrier",
    "Mars_helicopter_on_sol_46.png": "folding chair",
    "SOFIA.jpg": "wing",
    "astronaut.png": "neck brace",
    "cello.jpg": "cello, violoncello",
    "columbia.jpg": "space shuttle",
    "europa_clipper.jpeg": "airship, dirigible",
    "nasa_perseverance_selfie.jpg": "moped",
    "sushi.jpg": "pizza, pizza pie",
    "takoyaki.jpeg": "meat loaf, meatloaf",
    "whopper.jpg": "cheeseburger",
}
PREPROCESS_TOL = 1e-4


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m resnet.inference.selftest")
    ap.add_argument("folder")
    ap.add_argument("--model", default=os.environ.get("SCALES_RESNET_MODEL", "microsoft/resnet-18"))
    ap.add_argument("--no-trt", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from PIL import Image

    from resnet.inference.common import decode, list_images, preprocess_image
    from resnet.inference.pytorch import load_model_and_processor

    folder = os.path.abspath(a.folder)
    files = list_images(folder)
    if not files:
        print(f"no images in {folder}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False in " + sys.executable, file=sys.stderr)
        return 1

    model, proc = load_model_and_processor(a.model)
    labels = model.config.id2label
    cpu_model = model
    gpu_model = type(model).from_pretrained(a.model).eval().cuda()

    trt_sess = None
    trt_note = "skipped (--no-trt)" if a.no_trt else None
    if not a.no_trt:
        try:
            from scales_trt import runtime
            from resnet.inference import trt_adapter

            paths = runtime.cache_paths(os.path.expanduser("~/.scales_trt_cache"), trt_adapter.tag(a.model))
            eng = paths["engine_fp16"] if os.environ.get("TRT_FP16", "1") != "0" else paths["engine_fp32"]
            if os.path.isfile(eng):
                with open(eng, "rb") as f:
                    trt_sess = runtime.load_session(f.read())
                trt_note = os.path.basename(eng)
            else:
                trt_note = f"skipped (no engine at {eng})"
        except Exception as e:
            trt_note = f"skipped ({type(e).__name__}: {e})"

    rows, fails = [], []
    t_gpu = 0.0
    for name in files:
        with Image.open(os.path.join(folder, name)) as im:
            im = im.convert("RGB")
            ref = proc(images=im, return_tensors="pt")["pixel_values"]
            mine = torch.from_numpy(preprocess_image(im))
        pre_diff = float((ref - mine).abs().max())
        with torch.inference_mode():
            cpu_logits = cpu_model(pixel_values=ref).logits.numpy()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_logits = gpu_model(pixel_values=ref.cuda()).logits.float().cpu().numpy()
            t_gpu += time.perf_counter() - t0
        ci, cl, cp = decode(cpu_logits, labels)
        gi, gl, gp = decode(gpu_logits, labels)
        row = {
            "file": name, "label": gl, "prob": round(gp, 4), "preprocess_maxdiff": pre_diff,
            "cpu_label": cl, "cuda_vs_cpu_logit_maxdiff": float(np.abs(cpu_logits - gpu_logits).max()),
        }
        if pre_diff > PREPROCESS_TOL:
            fails.append(f"{name}: preprocess differs from HF by {pre_diff:.3g}")
        if gi != ci:
            fails.append(f"{name}: CUDA '{gl}' != CPU '{cl}'")
        if name in GOLDEN:
            row["golden"] = GOLDEN[name]
            if gl != GOLDEN[name]:
                fails.append(f"{name}: '{gl}' != golden '{GOLDEN[name]}'")
        if trt_sess is not None:
            from scales_trt import runtime

            y = runtime.run(trt_sess, preprocess_image(im))
            ti, tl, tp = decode(y, labels)
            row.update(trt_label=tl, trt_prob=round(tp, 4),
                       trt_vs_cuda_logit_maxdiff=float(np.abs(y.reshape(-1) - gpu_logits.reshape(-1)).max()))
            if ti != gi:
                fails.append(f"{name}: TensorRT '{tl}' != PyTorch '{gl}'")
        rows.append(row)

    summary = {
        "ok": not fails, "images": len(files), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "tensorrt": trt_note, "fails": fails,
        "cuda_ms_per_image": round(1000 * t_gpu / len(files), 2), "rows": rows,
    }
    if a.json:
        print(json.dumps(summary), flush=True)
    else:
        print(f"GPU {summary['gpu']}  torch {summary['torch']}  TensorRT: {trt_note}")
        print(f"{'file':32s} {'pre Δ':>8s} {'CUDA top-1':28s} {'p':>5s}  {'TensorRT top-1':28s}")
        for r in rows:
            print(f"{r['file'][:32]:32s} {r['preprocess_maxdiff']:8.1e} {r['label'][:28]:28s} {r['prob']:5.2f}  "
                  f"{r.get('trt_label', '-')[:28]:28s}")
        print(f"CUDA forward pass: {summary['cuda_ms_per_image']} ms/image")
        for f in fails:
            print("FAIL: " + f)
        print("SELFTEST PASSED" if not fails else f"SELFTEST FAILED ({len(fails)} problems)")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
