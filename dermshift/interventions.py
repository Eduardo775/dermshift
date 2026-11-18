"""Test-time transforms (analysis G3); the model is unchanged.

All functions take and return a uint8 RGB image at network resolution.

* ``sog``       Shades of Gray colour constancy (Finlayson and Trezzi, 2004), p = 6.
* ``reinhard``  Reinhard colour transfer in CIELAB to the statistics of the model's training images.
* ``hairless``  black-hat hair detection (17x17 cross, grey level > 10), 1-pixel dilation, Telea inpainting.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np

TRANSFORMS = ("sog", "reinhard", "hairless")


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    x = img.astype(np.float64) + 1e-6
    e = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)        # illuminant per channel
    gain = e.mean() / e
    return np.clip(img.astype(np.float64) * gain, 0, 255).astype(np.uint8)


def _to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def reinhard_to(img: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    lab = _to_lab(img)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    out = (lab - mu) / sd * np.asarray(std, np.float32) + np.asarray(mean, np.float32)
    return _from_lab(out)


def hair_mask(img: np.ndarray, k: int = 17, thr: int = 10) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k)))
    return (bh > thr).astype(np.uint8)


def remove_hair(img: np.ndarray, k: int = 17, thr: int = 10, radius: int = 5) -> np.ndarray:
    m = cv2.dilate(hair_mask(img, k, thr), np.ones((3, 3), np.uint8))
    if not m.any():
        return img.copy()
    return cv2.inpaint(img, m, radius, cv2.INPAINT_TELEA)


def lab_stats(cache_dir: Path, indices) -> dict:
    """Mean and std (per CIELAB channel, pooled over all pixels) of a set of cached images."""
    from .data.cache import CacheArrays

    arr = CacheArrays(Path(cache_dir))
    s1, s2, n = np.zeros(3), np.zeros(3), 0
    for i in indices:
        lab = _to_lab(arr.image(int(i))).reshape(-1, 3).astype(np.float64)
        s1 += lab.sum(0); s2 += (lab ** 2).sum(0); n += len(lab)
    mean = s1 / n
    return {"mean": mean.tolist(), "std": np.sqrt(np.maximum(s2 / n - mean ** 2, 0)).tolist(), "n_images": len(indices)}


def domain_color_stats(manifests: Path, caches: Path, domain: str) -> dict:
    f = Path(manifests) / f"color_stats_{domain}.json"
    if f.exists():
        return json.loads(f.read_text())
    from .domains import domain_indices

    cache, tr, _ = domain_indices(manifests, caches, domain)
    st = lab_stats(Path(caches) / cache, tr)
    f.write_text(json.dumps(st, indent=2))
    return st


def get_transform(name: str, color_stats: dict | None = None):
    if name == "sog":
        return shades_of_gray
    if name == "hairless":
        return remove_hair
    if name == "reinhard":
        if color_stats is None:
            raise ValueError("reinhard needs the colour statistics of the training domain")
        return partial(reinhard_to, mean=tuple(color_stats["mean"]), std=tuple(color_stats["std"]))
    raise ValueError(f"unknown transform '{name}' (expected one of {TRANSFORMS})")
