"""Dataset manifests: one row per image (dataset, image_id, image_path, mask_path, lesion_id, dx), with paths
relative to ``datasets_root``. Missing or unreadable files and non-binary masks raise an error.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..paths import ProjectPaths

ISIC_ID = re.compile(r"(ISIC_\d{7})")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()      # portable manifests (Windows <-> Linux)


def build_ham_manifest(paths: ProjectPaths) -> pd.DataFrame:
    meta = pd.read_csv(paths.ham_metadata)
    required = {"lesion_id", "image_id", "dx"}
    if not required.issubset(meta.columns):
        raise ValueError(f"HAM10000 metadata lacks columns {required - set(meta.columns)}")
    if meta["image_id"].duplicated().any():
        raise ValueError("Duplicated image_id in HAM10000 metadata")
    images = {ISIC_ID.search(p.name).group(1): p for p in paths.ham_images.iterdir()
              if p.suffix.lower() in IMG_EXT and ISIC_ID.search(p.name)}
    masks = {ISIC_ID.search(p.name).group(1): p for p in paths.ham_masks.iterdir()
             if p.suffix.lower() in IMG_EXT and ISIC_ID.search(p.name)}
    rows, missing = [], []
    for r in meta.itertuples(index=False):
        img, msk = images.get(r.image_id), masks.get(r.image_id)
        if img is None or msk is None:
            missing.append((r.image_id, img is None, msk is None))
            continue
        rows.append(dict(dataset="ham10000", image_id=r.image_id,
                         image_path=_rel(img, paths.datasets_root), mask_path=_rel(msk, paths.datasets_root),
                         lesion_id=r.lesion_id, dx=r.dx))
    if missing:
        raise FileNotFoundError(f"{len(missing)} HAM10000 metadata rows without image or mask, e.g. {missing[:5]}")
    extra = set(images) - set(meta["image_id"])
    if extra:
        raise ValueError(f"{len(extra)} HAM10000 images not in metadata, e.g. {sorted(extra)[:5]}")
    return pd.DataFrame(rows)


def build_isic2018_test_manifest(paths: ProjectPaths) -> pd.DataFrame:
    img_dir, msk_dir = paths.isic_test_images(), paths.isic_test_masks()
    images = {ISIC_ID.search(p.name).group(1): p for p in img_dir.iterdir()
              if p.suffix.lower() in IMG_EXT and ISIC_ID.search(p.name)}
    masks = {ISIC_ID.search(p.name).group(1): p for p in msk_dir.iterdir()
             if p.suffix.lower() in IMG_EXT and ISIC_ID.search(p.name)}
    if set(images) != set(masks):
        raise ValueError(f"ISIC 2018 test: {len(set(images) ^ set(masks))} ids without a pair")
    rows = [dict(dataset="isic2018_test", image_id=i, image_path=_rel(images[i], paths.datasets_root),
                 mask_path=_rel(masks[i], paths.datasets_root), lesion_id="", dx="") for i in sorted(images)]
    return pd.DataFrame(rows)


def _strict_ids(folder, exts: set, pattern: str) -> dict:
    """{image_id: path}; files with a valid extension but an unexpected name raise (no silent overwrite)."""
    rx, out, odd = re.compile(pattern, re.IGNORECASE), {}, []
    for p in folder.iterdir():
        if p.suffix.lower() not in exts:
            continue
        m = rx.match(p.name)
        if not m:
            odd.append(p.name); continue
        if m.group(1) in out:
            raise ValueError(f"two files for {m.group(1)} in {folder}")
        out[m.group(1)] = p
    if odd:
        raise ValueError(f"{len(odd)} unexpected files in {folder}, e.g. {odd[:3]}")
    return out


def build_isic2018_pool_manifest(paths: ProjectPaths) -> pd.DataFrame:
    """ISIC 2018 Task 1 training and validation images (the ISIC training pool). ``dx`` holds the official
    split; the partition used here is made in ``domains`` after removing near-duplicates of the evaluation sets."""
    rows = []
    for split, (img_dir, msk_dir) in zip(("train", "val"), paths.isic_trainval_dirs()):
        images = _strict_ids(img_dir, {".jpg", ".jpeg"}, r"^(ISIC_\d{7})\.jpe?g$")
        masks = _strict_ids(msk_dir, {".png"}, r"^(ISIC_\d{7})_segmentation\.png$")
        if set(images) != set(masks):
            raise ValueError(f"ISIC 2018 Task 1 {split}: {len(set(images) ^ set(masks))} ids without a pair")
        rows += [dict(dataset="isic2018_pool", image_id=i, image_path=_rel(images[i], paths.datasets_root),
                      mask_path=_rel(masks[i], paths.datasets_root), lesion_id="", dx=split) for i in sorted(images)]
    df = pd.DataFrame(rows)
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicated image ids between ISIC 2018 Task 1 training and validation")
    return df


def build_ph2_manifest(paths: ProjectPaths) -> pd.DataFrame:
    rows = []
    for case in sorted(p for p in paths.ph2_cases.iterdir() if p.is_dir() and p.name.startswith("IMD")):
        cid = case.name
        img = case / f"{cid}_Dermoscopic_Image" / f"{cid}.bmp"
        msk = case / f"{cid}_lesion" / f"{cid}_lesion.bmp"
        if not img.exists() or not msk.exists():
            raise FileNotFoundError(f"PH2 case {cid}: image or lesion mask missing")
        rows.append(dict(dataset="ph2", image_id=cid, image_path=_rel(img, paths.datasets_root),
                         mask_path=_rel(msk, paths.datasets_root), lesion_id="", dx=""))
    return pd.DataFrame(rows)


def imread(path: Path, flags: int = cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-ASCII paths on Windows (cv2.imread returns None there)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def read_rgb(path: Path) -> np.ndarray:
    img = imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Unreadable image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    m = imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Unreadable mask {path}")
    return m


def read_mask(path: Path, threshold: int = 127) -> np.ndarray:
    return (read_gray(path) > threshold).astype(np.uint8)


def verify_manifest(df: pd.DataFrame, datasets_root: Path, threshold: int = 127) -> pd.DataFrame:
    """Open every pair; return per-image QA (size, lesion fraction, raw mask values). Raises on hard errors."""
    qa = []
    for r in df.itertuples(index=False):
        img = read_rgb(datasets_root / r.image_path)
        raw = read_gray(datasets_root / r.mask_path)
        if raw.shape != img.shape[:2]:
            raise ValueError(f"{r.image_id}: image {img.shape[:2]} vs mask {raw.shape}")
        vals = np.unique(raw)
        m = raw > threshold
        qa.append(dict(image_id=r.image_id, height=img.shape[0], width=img.shape[1],
                       lesion_fraction=float(m.mean()), n_unique_mask_values=int(len(vals)),
                       touches_border=bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())))
    qa = pd.DataFrame(qa)
    n_empty = int((qa["lesion_fraction"] == 0).sum())
    if n_empty:
        # Kept, but reported: the metric policy for empty GT is defined in metrics.py.
        print(f"[verify] WARNING: {n_empty} empty ground-truth masks in {df['dataset'].iloc[0]}")
    return qa


BCN_DIAG_COLS = ("diagnosis_3", "diagnosis_2", "diagnosis_1", "diagnosis")


def build_bcn_manifest(paths: ProjectPaths, max_images: int | None = None, seed: int = 2026) -> pd.DataFrame:
    """BCN20000 (ISIC Archive collection 249), used without masks.

    Expects ``BCN20000/images/*.jpg`` (or ``BCN20000/*.jpg``) and ``BCN20000/metadata.csv`` from the ISIC Archive.
    Dermoscopic images with a recorded diagnosis are kept (17,790 of 18,946 in the 2026 export); diagnoses and
    anatomical sites are only used as strata. ``dx`` is the most specific diagnosis."""
    d = paths.bcn_dir
    meta_f = d / "metadata.csv"
    if not meta_f.exists():
        raise FileNotFoundError(f"{meta_f} missing: download BCN20000 first (see README)")
    meta = pd.read_csv(meta_f, keep_default_na=False, low_memory=False)
    idcol = "isic_id" if "isic_id" in meta.columns else "image_id"
    img_dir = d / "images" if (d / "images").exists() else d
    images = {ISIC_ID.search(p.name).group(1): p for p in img_dir.iterdir()
              if p.suffix.lower() in IMG_EXT and ISIC_ID.search(p.name)}
    if "image_type" in meta.columns:
        meta = meta[meta["image_type"].isin(["dermoscopic", ""])]
    diag = next((c for c in BCN_DIAG_COLS if c in meta.columns), None)
    if diag is None:
        raise ValueError(f"BCN metadata has no diagnosis column ({BCN_DIAG_COLS})")
    lvl1 = "diagnosis_1" if "diagnosis_1" in meta.columns else diag
    meta = meta[meta[lvl1].astype(str).str.len() > 0]
    rows, missing = [], 0
    for r in meta.itertuples(index=False):
        rr = r._asdict()
        iid = rr[idcol]
        if iid not in images:
            missing += 1
            continue
        dx = next((rr[c] for c in BCN_DIAG_COLS if c in rr and str(rr[c])), "")
        # recent Archive exports use anatom_site_1 / anatom_site_2 instead of anatom_site_general
        site = next((str(rr[c]) for c in ("anatom_site_general", "anatom_site_1") if c in rr and str(rr[c])), "")
        rows.append(dict(dataset="bcn20000", image_id=iid, image_path=_rel(images[iid], paths.datasets_root),
                         mask_path="", lesion_id=rr.get("lesion_id", ""), dx=dx,
                         diagnosis_1=rr.get(lvl1, ""), anatom_site=site, anatom_site_detail=rr.get("anatom_site_2", ""),
                         age_approx=rr.get("age_approx", ""), sex=rr.get("sex", "")))
    if missing:
        print(f"[bcn] WARNING: {missing} metadata rows without an image file (skipped)")
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if max_images and len(df) > max_images:
        df = df.sample(max_images, random_state=seed).sort_values("image_id").reset_index(drop=True)
    return df


def build_all(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    paths.manifests.mkdir(parents=True, exist_ok=True)
    out = {"ham10000": build_ham_manifest(paths),
           "isic2018_test": build_isic2018_test_manifest(paths),
           "ph2": build_ph2_manifest(paths)}
    for name, df in out.items():
        df.to_csv(paths.manifests / f"{name}.csv", index=False)
    return out
