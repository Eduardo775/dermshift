"""Per-image metrics on binary masks at the original resolution of the reference mask.

Region: Dice, IoU, sensitivity, specificity, accuracy.
Boundary (on contours, contour(M) = M and not erode(M), 4-connectivity): HD, HD95 and ASSD in pixels and in
percent of the image diagonal; normalised surface Dice (NSD) at tolerances in pixels and in percent of the
diagonal; log area ratio log((|P| + 1) / (|G| + 1)).
Empty masks: an empty prediction with a non-empty reference gives Dice = 0 and distances equal to the image
diagonal; both empty gives Dice = 1 and distances 0. These cases are flagged in the per-image table.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_CROSS = ndimage.generate_binary_structure(2, 1)


def contour(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=_CROSS, border_value=0)


def _directed(a_c: np.ndarray, b_c: np.ndarray) -> np.ndarray:
    dt = ndimage.distance_transform_edt(~b_c)
    return dt[a_c]


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> dict:
    diag = math.hypot(*gt.shape)
    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return {"hd": 0.0, "hd95": 0.0, "assd": 0.0}
    if pe or ge:
        return {"hd": diag, "hd95": diag, "assd": diag}
    pc, gc = contour(pred), contour(gt)
    d_pg, d_gp = _directed(pc, gc), _directed(gc, pc)
    both = np.hstack([d_pg, d_gp])
    return {"hd": float(both.max()), "hd95": float(np.percentile(both, 95)),
            "assd": float(both.mean())}


def surface_metrics(pred: np.ndarray, gt: np.ndarray, tol_px=(), tol_pct=()) -> dict:
    """HD/HD95/ASSD (same policy as :func:`surface_distances`) plus the normalised surface Dice
    (NSD, Nikolov et al. 2021) at each tolerance: the fraction of contour points of P and G that lie
    within tau of the other contour. ``tol_px`` in pixels, ``tol_pct`` in % of the image diagonal
    (comparable across datasets whose resolutions differ by an order of magnitude).
    NSD policy: both empty -> 1; exactly one empty -> 0. ``diag`` may be passed through ``gt`` crops
    by the caller (see evaluate._metrics_job) -- here it is the diagonal of ``gt``."""
    return _surface(pred, gt, tol_px, tol_pct, math.hypot(*gt.shape))


def _surface(pred, gt, tol_px, tol_pct, diag) -> dict:
    taus = {f"nsd_{t:g}px": float(t) for t in tol_px}
    taus.update({f"nsd_{t:g}pct": t / 100 * diag for t in tol_pct})
    pe, ge = not pred.any(), not gt.any()
    if pe and ge:
        return {"hd": 0.0, "hd95": 0.0, "assd": 0.0, **{k: 1.0 for k in taus}}
    if pe or ge:
        return {"hd": diag, "hd95": diag, "assd": diag, **{k: 0.0 for k in taus}}
    pc, gc = contour(pred), contour(gt)
    d_pg, d_gp = _directed(pc, gc), _directed(gc, pc)
    both = np.hstack([d_pg, d_gp])
    out = {"hd": float(both.max()), "hd95": float(np.percentile(both, 95)), "assd": float(both.mean())}
    for k, t in taus.items():
        out[k] = float((np.count_nonzero(d_pg <= t) + np.count_nonzero(d_gp <= t)) / len(both))
    return out


def log_area_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """log((|P| + 1) / (|G| + 1)): > 0 over-segmentation, < 0 under-segmentation (+1 avoids log 0)."""
    return float(np.log((np.count_nonzero(pred) + 1.0) / (np.count_nonzero(gt) + 1.0)))


def unlabelled_stats(prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Label-free descriptors of one probability map (network resolution), for domains without masks."""
    m = prob >= threshold
    eps = 1e-6
    p = np.clip(prob.astype(np.float64), eps, 1 - eps)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    band = ndimage.binary_dilation(contour(m), iterations=3) if m.any() else np.zeros_like(m)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    n_comp = int(ndimage.label(m)[1]) if m.any() else 0
    return {"pred_area_frac": float(m.mean()), "pred_empty": bool(not m.any()),
            "pred_border_frac": float(border.mean()), "n_components": n_comp,
            "entropy_mean": float(ent.mean()), "entropy_band": float(ent[band].mean()) if band.any() else 0.0}


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def image_metrics(pred: np.ndarray, gt: np.ndarray, boundary: bool = True) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    tp, fp, fn, tn = confusion(pred, gt)
    pe, ge = tp + fp == 0, tp + fn == 0
    if pe and ge:
        dice = iou = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    out = {
        "dice": dice, "iou": iou,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "accuracy": (tp + tn) / pred.size,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_empty": bool(pe), "gt_empty": bool(ge),
    }
    if boundary:
        out.update(surface_distances(pred, gt))
    return out


def pooled_metrics(rows) -> dict:
    """Dataset-level Dice and IoU from summed TP/FP/FN."""
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    return {"dice_pooled": 2 * tp / (2 * tp + fp + fn), "iou_pooled": tp / (tp + fp + fn)}


def batch_dice(logits, target, threshold: float = 0.5) -> list[float]:
    """Fast per-image Dice on the 192x256 tensors (validation / early stopping only)."""
    import torch

    with torch.no_grad():
        p = (torch.sigmoid(logits.float()) >= threshold).flatten(1)
        g = (target >= 0.5).flatten(1)
        tp = (p & g).sum(1).float(); s = p.sum(1).float() + g.sum(1).float()
        d = torch.where(s > 0, 2 * tp / s.clamp(min=1), torch.ones_like(s))
    return d.cpu().tolist()
