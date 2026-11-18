"""Seeding, RNG state capture and restore, and an environment fingerprint.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few upsampling backward kernels have no deterministic CUDA implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True        # fastest kernels; runs are seeded but not bitwise-reproducible
        torch.use_deterministic_algorithms(False)


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def environment() -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for mod in ("segmentation_models_pytorch", "timm", "albumentations", "numpy", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        env["git"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                             cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        env["git"] = None
    return env
