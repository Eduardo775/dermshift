"""Pre-resized array caches.

Images are resized to 192x256 before augmentation, so they are cached once as memory-mappable arrays.
Evaluation caches also keep the reference masks at original resolution.

Layout of ``<cache>/<name>/``::

    images.npy        uint8 (N, H, W, 3), INTER_AREA (INTER_LINEAR when upsampling)
    masks.npy         uint8 (N, H, W), nearest neighbour
    meta.csv          image_id, orig_h, orig_w and the manifest columns
    gt_original.npz   evaluation caches only: bit-packed reference masks at original size
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .manifest import read_gray, read_rgb


def resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    interp = cv2.INTER_AREA if (img.shape[0] >= h and img.shape[1] >= w) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def resize_mask(m: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def _load_pair(args):
    """Read, check and resize one image/mask pair (runs in a thread pool)."""
    r, datasets_root, h, w, threshold, keep = args
    img = read_rgb(datasets_root / r.image_path)
    if not getattr(r, "mask_path", ""):                     # unlabelled domain (BCN20000): no mask
        qa = dict(image_id=r.image_id, height=img.shape[0], width=img.shape[1], lesion_fraction=float("nan"),
                  n_unique_mask_values=0, touches_border=False)
        return (resize_image(img, h, w), np.zeros((h, w), np.uint8), None,
                dict(r._asdict(), orig_h=img.shape[0], orig_w=img.shape[1]), qa)
    raw = read_gray(datasets_root / r.mask_path)
    if raw.shape != img.shape[:2]:
        raise ValueError(f"{r.image_id}: image {img.shape[:2]} vs mask {raw.shape}")
    m = (raw > threshold).astype(np.uint8)
    qa = dict(image_id=r.image_id, height=img.shape[0], width=img.shape[1], lesion_fraction=float(m.mean()),
              n_unique_mask_values=int(len(np.unique(raw))),
              touches_border=bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()))
    gt = (np.packbits(m.astype(bool)), np.array(m.shape)) if keep else None   # 1 bit/pixel in RAM and on disk
    return (resize_image(img, h, w), resize_mask(m, h, w), gt,
            dict(r._asdict(), orig_h=img.shape[0], orig_w=img.shape[1]), qa)


def build_cache(manifest: pd.DataFrame, datasets_root: Path, out_dir: Path, h: int, w: int,
                threshold: int = 127, keep_original_gt: bool = False, overwrite: bool = False,
                work_dir: Path | None = None, threads: int = 16) -> Path:
    """Single pass over the raw files: integrity checks (-> qa.csv), resize, write arrays.
    Files are read by ``threads`` parallel readers; arrays are written to ``work_dir`` and then moved to
    ``out_dir``."""
    from concurrent.futures import ThreadPoolExecutor

    out_dir = Path(out_dir)
    if (out_dir / "_COMPLETE").exists() and not overwrite:
        print(f"[cache] {out_dir} already complete; skipping")
        return out_dir
    tmp = Path(work_dir or out_dir.parent / "_tmp") / out_dir.name
    if tmp.resolve() == out_dir.resolve():                  # workstation: caches read in place
        tmp = out_dir.parent / "_tmp" / out_dir.name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(manifest)
    imgs = np.lib.format.open_memmap(tmp / "images.npy", mode="w+", dtype=np.uint8, shape=(n, h, w, 3))
    msks = np.lib.format.open_memmap(tmp / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, h, w))
    gt_orig, meta, qa = {}, [], []
    t0 = time.time()
    jobs = ((r, Path(datasets_root), h, w, threshold, keep_original_gt) for r in manifest.itertuples(index=False))
    with ThreadPoolExecutor(threads) as ex:
        for i, (im, mk, gt, me, q) in enumerate(ex.map(_load_pair, jobs)):
            imgs[i], msks[i] = im, mk
            if gt is not None:
                gt_orig[me["image_id"]] = gt[0]
                gt_orig["__shape__" + me["image_id"]] = gt[1]
            meta.append(me); qa.append(q)
            if (i + 1) % 250 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"[cache] {out_dir.name}: {i + 1}/{n}  {el / 60:.1f} min, ~{el / (i + 1) * (n - i - 1) / 60:.1f} min left",
                      flush=True)
    imgs.flush(); msks.flush()
    del imgs, msks
    import gc
    gc.collect()                       # Windows: release the memmap file handles before moving the folder
    pd.DataFrame(meta).to_csv(tmp / "meta.csv", index=False)
    pd.DataFrame(qa).to_csv(tmp / "qa.csv", index=False)
    if keep_original_gt:
        np.savez_compressed(tmp / "gt_original.npz", **gt_orig)
    (tmp / "_COMPLETE").write_text(f"{n} items, {h}x{w}\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.move(str(tmp), str(out_dir))
    print(f"[cache] {out_dir.name} written ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out_dir


def copy_to_local(src: Path, dst: Path) -> Path:
    """Copy a finished cache to a faster disk (``paths.local_root``)."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        if not (src / "_COMPLETE").exists():
            raise FileNotFoundError(f"Cache {src} is incomplete; rebuild it")
        return dst
    if not (src / "_COMPLETE").exists():
        raise FileNotFoundError(f"Cache {src} is incomplete; rebuild it")
    if (dst / "_COMPLETE").exists() and (dst / "_COMPLETE").read_text() == (src / "_COMPLETE").read_text():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    t0 = time.time()
    shutil.copytree(src, dst)
    print(f"[cache] copied {src.name} to local disk in {time.time() - t0:.0f}s")
    return dst


class CacheArrays:
    """Lazy, fork-safe access to a cache (memmap opened per process)."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.meta = pd.read_csv(self.dir / "meta.csv", keep_default_na=False)
        self._imgs = self._msks = None
        self._gt = None
        self.has_masks = "mask_path" not in self.meta.columns or bool((self.meta["mask_path"].astype(str) != "").all())

    def __getstate__(self):
        # Windows DataLoader workers are spawned: never pickle open memmaps (that would copy the arrays)
        st = self.__dict__.copy()
        st["_imgs"] = st["_msks"] = st["_gt"] = None
        return st

    def _open(self):
        if self._imgs is None:
            self._imgs = np.load(self.dir / "images.npy", mmap_mode="r")
            self._msks = np.load(self.dir / "masks.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.meta)

    def image(self, i: int) -> np.ndarray:
        self._open()
        return np.array(self._imgs[i])        # writable copy (memmap slices are read-only)

    def mask(self, i: int) -> np.ndarray:
        self._open()
        return np.array(self._msks[i])

    def original_gt(self, image_id: str) -> np.ndarray:
        if self._gt is None:
            self._gt = np.load(self.dir / "gt_original.npz")
        shape = tuple(self._gt["__shape__" + image_id])
        return np.unpackbits(self._gt[image_id], count=shape[0] * shape[1]).reshape(shape).astype(bool)
