"""S5: ISIC 2018 Task 1 photographs that appear in both the training pool and the test set.

Each such photograph has a training mask and a test mask, so comparing the two measures the difference in
annotation convention with no model and no population difference. Pairs are the candidates of
``verify-duplicates`` marked ``same`` in ``isic2018_train_test_review.csv`` (or, without a review, those with
NCC >= ``ncc_min``). The test image is aligned to the training image (flip, then an affine ECC registration)
and both masks are compared in the training frame at cache resolution.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.stats as st

from .data.cache import CacheArrays

REVIEW_FILE = "isic2018_train_test_review.csv"
FLIPS = {"none": lambda a: a, "h": lambda a: a[:, ::-1], "v": lambda a: a[::-1]}


def _gray(img):
    g = img.astype(np.float32).mean(-1)
    g -= g.mean()
    return g


def best_flip(a: np.ndarray, b: np.ndarray) -> tuple[str, float]:
    """Flip of ``b`` that maximises NCC with ``a``."""
    ga, out = _gray(a), {}
    na = np.linalg.norm(ga)
    for k, f in FLIPS.items():
        gb = _gray(f(b))
        den = na * np.linalg.norm(gb)
        out[k] = float((ga * gb).sum() / den) if den > 0 else 0.0
    k = max(out, key=out.get)
    return k, out[k]


def register(img_ref: np.ndarray, img_mov: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Affine warp mapping ``img_mov`` onto ``img_ref`` (same photograph, possibly re-exported with a slight zoom,
    crop or brightness/contrast change). ECC maximisation is invariant to affine photometric changes. Returns
    (2x3 warp or None, NCC after warping)."""
    a = cv2.GaussianBlur(cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(img_mov, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        return None, float("nan")
    h, w = a.shape
    bw = cv2.warpAffine(b, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones_like(b), warp, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
    x, y = a[valid] - a[valid].mean(), bw[valid] - bw[valid].mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return warp, (float(x @ y / den) if den > 0 else 0.0)


def warp_mask(m: np.ndarray, warp: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Test mask in the training frame + the region of that frame covered by the test image."""
    h, w = shape
    mw = cv2.warpAffine(m.astype(np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
    valid = cv2.warpAffine(np.ones(m.shape, np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP) > 0
    return mw, valid


def _scale(warp: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(warp[:, :2])))) if warp is not None else float("nan")


def _lab_L(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * (100 / 255)


def edge_darkness_out(img: np.ndarray, m: np.ndarray) -> float:
    """Same tightness proxy as S3: L*(ring 8-16 px outside) - L*(band 1-3 px outside)."""
    m = m.astype(np.uint8)
    if not m.any() or m.all():
        return np.nan
    k = lambda n: np.ones((2 * n + 1, 2 * n + 1), np.uint8)
    near = (cv2.dilate(m, k(3)) > 0) & ~(m > 0)
    far = (cv2.dilate(m, k(16)) > 0) & ~(cv2.dilate(m, k(8)) > 0)
    if not near.any() or not far.any():
        return np.nan
    L = _lab_L(img)
    return float(L[far].mean() - L[near].mean())


def signed_offset(ref: np.ndarray, other: np.ndarray) -> float:
    """Signed boundary offset (px) of ``other`` relative to ``ref``: > 0 when ``other`` lies outside ``ref``
    (larger), < 0 when inside (tighter). Symmetrised, 0.5 * (d(ref->other) - d(other->ref)), which cancels the
    ~1 px bias of measuring at boundary pixels (identical masks give exactly 0)."""
    a, b = _raw_offset(ref, other), _raw_offset(other, ref)
    return 0.5 * (a - b) if np.isfinite(a) and np.isfinite(b) else np.nan


def _raw_offset(ref: np.ndarray, other: np.ndarray) -> float:
    ref, other = ref.astype(np.uint8), other.astype(np.uint8)
    if not ref.any() or not other.any():
        return np.nan
    d_out = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 5)        # distance outside ref
    d_in = cv2.distanceTransform(ref, cv2.DIST_L2, 5)             # distance inside ref
    sd = d_out - d_in
    cs, _ = cv2.findContours(other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c[:, 0, :] for c in cs]) if cs else np.empty((0, 2), int)
    return float(sd[pts[:, 1], pts[:, 0]].mean()) if len(pts) else np.nan


def read_review(path: Path) -> pd.DataFrame:
    """Review sheet as saved by a spreadsheet program: comma or semicolon separated, optional BOM."""
    r = pd.read_csv(path, sep=None, engine="python", keep_default_na=False, encoding="utf-8-sig")
    r.columns = [str(c).strip().lower() for c in r.columns]
    for c in ("query_id", "ref_id"):
        if c in r:
            r[c] = r[c].astype(str).str.strip()
    return r


def select_pairs(manifests: Path, ncc_min: float = 0.98) -> tuple[pd.DataFrame, str]:
    pairs = pd.read_csv(Path(manifests) / "near_duplicates_isic2018_pool_vs_isic2018_test_ncc.csv")
    rev = Path(manifests) / REVIEW_FILE
    rule = f"ncc >= {ncc_min}"
    if rev.exists():
        r = read_review(rev)
        if "decision" in r and (r["decision"].astype(str).str.strip() != "").any():
            same = r[r["decision"].astype(str).str.strip().str.lower().isin(["same", "yes", "1"])]
            pairs = pairs.merge(same[["query_id", "ref_id"]], on=["query_id", "ref_id"])
            rule = "visual review: same"
            return _one_to_one(pairs), rule
    return _one_to_one(pairs[pairs["ncc"] >= ncc_min]), rule


def _one_to_one(p: pd.DataFrame) -> pd.DataFrame:
    """Keep each training and each test image once (highest NCC first)."""
    p = p.sort_values("ncc", ascending=False)
    p = p.drop_duplicates("ref_id").drop_duplicates("query_id")
    return p.reset_index(drop=True)


def compare_pairs(pairs: pd.DataFrame, cache_train: CacheArrays, cache_test: CacheArrays,
                  min_ncc_registered: float = 0.9) -> pd.DataFrame:
    """Flip, then affine registration of the test image onto the training image; both masks are compared in the
    training frame, restricted to the region covered by both images. Pairs whose registration fails (NCC after
    warping < ``min_ncc_registered``) are kept in the table with ``registered = False`` and left out of the summary."""
    it = {i: k for k, i in enumerate(cache_train.meta["image_id"])}
    ie = {i: k for k, i in enumerate(cache_test.meta["image_id"])}
    rows = []
    for r in pairs.itertuples(index=False):
        if r.query_id not in it or r.ref_id not in ie:
            continue
        img_tr, m_tr = cache_train.image(it[r.query_id]), cache_train.mask(it[r.query_id]) > 0
        img_te, m_te = cache_test.image(ie[r.ref_id]), cache_test.mask(ie[r.ref_id]) > 0
        flip, ncc = best_flip(img_tr, img_te)
        img_te = np.ascontiguousarray(FLIPS[flip](img_te))
        m_te = np.ascontiguousarray(FLIPS[flip](m_te))
        warp, ncc_reg = register(img_tr, img_te)
        ok = warp is not None and ncc_reg >= min_ncc_registered
        if warp is None:
            warp = np.eye(2, 3, dtype=np.float32)
        m_te_w, valid = warp_mask(m_te, warp, m_tr.shape)
        m_tr_v = m_tr & valid
        a_tr, a_te = int(m_tr_v.sum()), int(m_te_w.sum())
        inter = int((m_tr_v & m_te_w).sum())
        h, w = m_tr.shape
        off = signed_offset(m_tr_v, m_te_w)
        rows.append({"train_id": r.query_id, "test_id": r.ref_id, "ncc": ncc, "flip": flip,
                     "ncc_registered": ncc_reg, "scale_test_to_train": _scale(warp),
                     "shift_px": float(np.hypot(warp[0, 2], warp[1, 2])), "overlap_frac": float(valid.mean()),
                     "registered": bool(ok), "area_train": a_tr, "area_test": a_te,
                     "log_area_ratio_test_vs_train": float(np.log((a_te + 1) / (a_tr + 1))),
                     "dice_train_test": 2 * inter / (a_tr + a_te) if a_tr + a_te else np.nan,
                     "edge_darkness_train_mask": edge_darkness_out(img_tr, m_tr_v),
                     "edge_darkness_test_mask": edge_darkness_out(img_tr, m_te_w),
                     "offset_test_vs_train_px": off,
                     "offset_test_vs_train_pct_diag": off / np.hypot(h, w) * 100})
    d = pd.DataFrame(rows)
    if len(d):
        d["edge_darkness_diff"] = d["edge_darkness_test_mask"] - d["edge_darkness_train_mask"]
    return d


def _median_ci(x, n_boot: int = 10_000, seed: int = 0):
    """Bootstrap 95% CI of the median (small n: the median is the robust summary here)."""
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """One-sample Wilcoxon against 0 of each paired difference (test mask minus train mask)."""
    rows = []
    if "registered" in d:
        d = d[d["registered"]]
    for col, lab in [("log_area_ratio_test_vs_train", "log area ratio (test/train)"),
                     ("edge_darkness_diff", "edge darkness, test - train mask"),
                     ("offset_test_vs_train_pct_diag", "signed boundary offset, % diagonal"),
                     ("dice_train_test", "Dice between the two masks")]:
        v = d[col].dropna().to_numpy(float) if col in d else np.array([])
        if len(v) < 3:
            continue
        test = col != "dice_train_test"
        p = (1.0 if np.allclose(v, 0) else float(st.wilcoxon(v).pvalue)) if test else np.nan
        lo, hi = _median_ci(v)
        rows.append({"quantity": lab, "n": len(v), "median": float(np.median(v)), "mean": float(v.mean()),
                     "median_ci95_low": lo, "median_ci95_high": hi, "share_negative": float((v < 0).mean()), "p_wilcoxon": p})
    out = pd.DataFrame(rows)
    if len(out):
        q = out["quantity"] == "log area ratio (test/train)"
        out.loc[q, "area_ratio_median"] = np.exp(out.loc[q, "median"])
    return out


def pairs_panel(d: pd.DataFrame, cache_train: CacheArrays, cache_test: CacheArrays, out_dir: Path,
                per_page: int = 16, upscale: int = 2) -> list[Path]:
    """Training image with the training mask (violet) and the aligned test mask (white)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .figures import DOUBLE, INK2, _save

    it = {i: k for k, i in enumerate(cache_train.meta["image_id"])}
    ie = {i: k for k, i in enumerate(cache_test.meta["image_id"])}
    violet = (98, 80, 214)
    files = []
    d = d.sort_values("log_area_ratio_test_vs_train").reset_index(drop=True)
    for s in range(0, len(d), per_page):
        chunk = d.iloc[s:s + per_page].reset_index(drop=True)
        ncol = 4
        nrow = int(np.ceil(len(chunk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, DOUBLE * 0.75 / ncol * nrow * 1.05))
        axes = np.atleast_2d(axes)
        for k, r in chunk.iterrows():
            ax = axes[k // ncol, k % ncol]
            img = cache_train.image(it[r.train_id])
            m_tr = cache_train.mask(it[r.train_id]) > 0
            img_te = np.ascontiguousarray(FLIPS[r.flip](cache_test.image(ie[r.test_id])))
            m_te = np.ascontiguousarray(FLIPS[r.flip](cache_test.mask(ie[r.test_id]) > 0))
            warp, _ = register(img, img_te)
            if warp is not None:
                m_te, _ = warp_mask(m_te, warp, m_tr.shape)
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
            size = (img.shape[1], img.shape[0])
            im = img.copy()
            for m, col in ((m_tr, violet), (m_te, (255, 255, 255))):
                mm = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                im = cv2.drawContours(im, cs, -1, col, 2)
            ax.imshow(im)
            flag = "" if r.registered else "  [registration failed]"
            ax.set_title(f"{r.train_id} | {r.test_id}  (scale {r.scale_test_to_train:.2f})\n"
                         f"test/train area {np.expm1(r.log_area_ratio_test_vs_train) * 100:+.0f}%{flag}",
                         fontsize=5.5, color=INK2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        for k in range(len(chunk), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        h = [Line2D([], [], color="#6250d6", lw=2, label="ISIC 2018 train mask"),
             Line2D([], [], color="#bbbbbb", lw=2, label="ISIC 2018 test mask (white), same photograph")]
        fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout()
        files += _save(fig, Path(out_dir) / f"duplicate_masks_{s // per_page:02d}")
    return files


def fig_pairs(d: pd.DataFrame, out: Path) -> list[Path]:
    """Paper figure (horizontal, one column wide): per-pair area ratio (test/train) - 1, sorted, with the median."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import GRID, INK, INK2, SINGLE, _save

    if "registered" in d:
        d = d[d["registered"]]
    v = np.sort(np.expm1(d["log_area_ratio_test_vs_train"].to_numpy(float)) * 100)
    x = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(SINGLE, 1.55))
    ax.vlines(x, 0, v, color="#6250d6", lw=0.6, alpha=0.45, zorder=2)
    ax.scatter(x, v, s=9, color="#6250d6", edgecolor="white", lw=0.4, zorder=3)
    med = float(np.median(v))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.axhline(med, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)
    neg = int((v < 0).sum())
    ax.text(1, med - 2, f"median {med:+.1f}%".replace("-", "\u2212"), fontsize=6.3, color=INK, va="top", ha="left")
    ax.text(len(v), v.min(), f"test mask smaller: {neg}/{len(v)} pairs", fontsize=6.3, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0, len(v) + 1)
    ax.set_xticks([1, len(v) // 2, len(v)])
    ax.set_xlabel("Duplicate pairs, sorted", fontsize=7)
    ax.set_ylabel("Test / train mask\narea − 1 (%)", fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    return _save(fig, out)


def run_duplicate_annotation(cfg: dict, ncc_min: float = 0.98) -> dict:
    from .paths import from_config

    P = from_config(cfg)
    pairs, rule = select_pairs(P.manifests, ncc_min)
    ctr, cte = CacheArrays(P.caches_local / "isic2018_pool"), CacheArrays(P.caches_local / "isic2018_test")
    d = compare_pairs(pairs, ctr, cte)
    out = P.analysis_dir
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "S5_duplicate_pairs.csv", index=False)
    s = summarize(d).assign(selection=rule)
    s.to_csv(out / "S5_summary.csv", index=False)
    files = []
    if len(d):
        files += pairs_panel(d, ctr, cte, P.figures_dir / "duplicate_masks")
        files += fig_pairs(d, P.figures_dir / "duplicate_pairs")
    n_reg = int(d["registered"].sum()) if len(d) else 0
    return {"pairs": len(d), "registered": n_reg, "selection": rule, "summary": s, "files": [str(f) for f in files],
            "scale_range": (float(d["scale_test_to_train"].min()), float(d["scale_test_to_train"].max())) if len(d) else None}
