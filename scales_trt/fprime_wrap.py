# F´ (fprime-venv, Python 3.12) -> GPU interpreter (JetPack Python 3.10) bridge.
#
# NVIDIA only ships CUDA-enabled torch and TensorRT Python bindings for
# JetPack's system Python, so anything that needs the GPU runs as a child
# process under that interpreter. The child prints diagnostics to stderr and a
# single JSON object as the last line of stdout.
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple

DEFAULT_SYSTEM_PYTHON = "/usr/bin/python3"


def _repo_ml_venv() -> Optional[str]:
    # ml-venv created by `make jetson-setup` in the fprime-scales-ref root
    roots = [os.environ.get("FPRIME_SCALES_ROOT")]
    here = os.path.dirname(os.path.abspath(__file__))
    # Scales-ML/scales_trt -> Scales-ML -> MLComponent -> Components -> repo root
    roots.append(os.path.abspath(os.path.join(here, "..", "..", "..", "..")))
    for root in roots:
        if root:
            py = os.path.join(root, "ml-venv", "bin", "python")
            if os.access(py, os.X_OK):
                return py
    return None


def ensure_env() -> None:
    # Defaults so F´ needs no manual exports; override before F´ starts if needed
    os.environ.setdefault("SCALES_TRT_PYTHON", _repo_ml_venv() or DEFAULT_SYSTEM_PYTHON)
    os.environ.setdefault("TRT_FP16", "1")
    os.environ.setdefault("SCALES_TRT_MODEL", "microsoft/resnet-18")
    os.environ.setdefault("SCALES_YOLO_WEIGHTS", "yolov8n.pt")


def jetson_python() -> str:
    """Interpreter that has CUDA torch + TensorRT (ml-venv, else /usr/bin/python3)."""
    ensure_env()
    return os.environ.get("SCALES_TRT_PYTHON") or os.environ.get("TRT_PYTHON") or DEFAULT_SYSTEM_PYTHON


WORKER_FLAG = "SCALES_ML_WORKER"


def is_gpu_interpreter() -> bool:
    """True when this process is a worker child, or already runs under jetson_python()."""
    if os.environ.get(WORKER_FLAG) == "1":
        return True
    # <prefix>/bin/python -> <prefix>; compare without resolving symlinks, since a
    # venv's python is a symlink to the system interpreter.
    gpu_prefix = os.path.dirname(os.path.dirname(os.path.abspath(jetson_python())))
    return os.path.abspath(sys.prefix) == gpu_prefix


def run_module(module: str, argv: List[str], extra_env: Optional[dict] = None) -> dict:
    """Run `python -m module argv...` under the GPU interpreter, return its JSON reply.

    The child's stderr (device line, timings, warnings) is forwarded to this
    process's stderr so it shows up in the F´ journal.
    """
    ensure_env()
    env = os.environ.copy()
    env[WORKER_FLAG] = "1"
    # Inference must not depend on the network: Hugging Face's online file
    # checks add ~80 s per call on the testbench. `make jetson-setup` caches the
    # weights; set HF_HUB_OFFLINE=0 to allow downloads.
    env.setdefault("HF_HUB_OFFLINE", "1")
    if extra_env:
        env.update(extra_env)
    py = jetson_python()
    if not os.access(py, os.X_OK):
        raise RuntimeError(f"GPU Python {py} not found; run `make jetson-setup`")
    r = subprocess.run(
        [py, "-m", module, *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
    )
    if r.stderr:
        sys.stderr.write(r.stderr if r.stderr.endswith("\n") else r.stderr + "\n")
        sys.stderr.flush()
    lines = (r.stdout or "").strip().splitlines()
    line = lines[-1] if lines else ""
    try:
        payload = json.loads(line) if line else None
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        tail = (r.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(f"{module} under {py} exited {r.returncode} without a JSON reply: " + " | ".join(tail))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", f"{module} failed"))
    return payload


def worker(argv: List[str]) -> dict:
    return run_module("scales_trt.worker", argv)


def trt_infer(adapter: str, folder: str, model: str, tag: str) -> List[Tuple[str, str]]:
    # folder = absolute path from SET_INFERENCE_PATH
    fp16 = "1" if os.environ.get("TRT_FP16", "1") != "0" else "0"
    p = worker(["infer", "--adapter", adapter, "--folder", folder, "--model", model, "--fp16", fp16])
    stats = p.get("stats") or {}
    fps = stats.get("fps")
    if fps:
        print(
            f"[{tag}] {stats.get('device', 'cuda')} | {stats.get('images')} images | "
            f"{stats.get('seconds', 0):.3f}s | {fps:.2f} FPS",
            file=sys.stderr,
            flush=True,
        )
    return [(str(a), str(b)) for a, b in p.get("outputs", [])]
