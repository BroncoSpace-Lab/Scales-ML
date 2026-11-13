# CUDA device selection + environment report.
#
#   python -m scales_trt.cuda_check            # human-readable report, exit 1 if CUDA unusable
#   python -m scales_trt.cuda_check --json     # same, as one JSON line
#   python -m scales_trt.cuda_check --no-trt   # don't require TensorRT
#
# SCALES_ML_DEVICE selects the inference device:
#   cuda  -> require the GPU; raise if torch can't use it (default on Jetson)
#   auto  -> GPU if available, else CPU with a warning (default elsewhere)
#   cpu   -> force CPU
from __future__ import annotations

import json
import os
import platform
import sys

TEGRA_RELEASE = "/etc/nv_tegra_release"


def is_jetson() -> bool:
    return os.path.isfile(TEGRA_RELEASE)


def l4t_release() -> str | None:
    # "# R36 (release), REVISION: 4.4, ..." -> "36.4.4"
    try:
        head = open(TEGRA_RELEASE).readline()
        major = head.split("R", 1)[1].split()[0]
        rev = head.split("REVISION:", 1)[1].split(",")[0].strip()
        return f"{major}.{rev}"
    except Exception:
        return None


def requested_device() -> str:
    d = os.environ.get("SCALES_ML_DEVICE", "").strip().lower()
    if d in ("cuda", "cpu", "auto"):
        return d
    return "cuda" if is_jetson() else "auto"


def select_device():
    """Return a torch.device according to SCALES_ML_DEVICE, failing loudly when CUDA is required."""
    import torch

    want = requested_device()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    msg = (
        f"CUDA unavailable in {sys.executable} (torch {torch.__version__}, built for CUDA "
        f"{torch.version.cuda}). On the Jetson run `make jetson-setup`; "
        "set SCALES_ML_DEVICE=cpu to allow CPU inference."
    )
    if want == "cuda":
        raise RuntimeError(msg)
    print("[scales-ml] WARNING: " + msg + " Falling back to CPU.", file=sys.stderr, flush=True)
    return torch.device("cpu")


def report(require_trt: bool = True) -> dict:
    r: dict = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "jetson": is_jetson(),
        "l4t": l4t_release(),
        "problems": [],
    }
    try:
        import torch

        r["torch"] = torch.__version__
        r["torch_cuda"] = torch.version.cuda
        r["cuda_available"] = bool(torch.cuda.is_available())
        if r["cuda_available"]:
            r["gpu"] = torch.cuda.get_device_name(0)
            r["cudnn"] = torch.backends.cudnn.version()
            # A real kernel launch, not just device enumeration
            x = torch.randn(256, 256, device="cuda")
            r["matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
            conv = torch.nn.Conv2d(3, 8, 3).cuda()
            r["cudnn_conv_ok"] = bool(torch.isfinite(conv(torch.randn(1, 3, 32, 32, device="cuda"))).all().item())
        else:
            r["problems"].append("torch.cuda.is_available() is False")
    except Exception as e:  # ImportError or a CUDA runtime failure
        r["problems"].append(f"torch: {e}")
    try:
        import tensorrt as trt

        r["tensorrt"] = trt.__version__
    except Exception as e:
        if require_trt:
            r["problems"].append(f"tensorrt: {e}")
    for mod in ("numpy", "PIL", "transformers", "onnx"):
        try:
            m = __import__(mod)
            r[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            r["problems"].append(f"{mod}: {e}")
    r["ok"] = not r["problems"]
    return r


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    r = report(require_trt="--no-trt" not in argv)
    if "--json" in argv:
        print(json.dumps(r), flush=True)
    else:
        for k, v in r.items():
            if k != "problems":
                print(f"  {k:15s} {v}")
        for p in r["problems"]:
            print(f"  PROBLEM         {p}")
        print("CUDA stack OK" if r["ok"] else "CUDA stack NOT OK")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
