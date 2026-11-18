"""Evaluation at the original resolution of the reference masks.

The model runs on the cached 192x256 input; probabilities are upsampled bilinearly to the size of the reference
mask and thresholded at 0.5. Outputs in ``<run_dir>/eval/<set>/``::

    per_image.csv       metrics per image
    summary.json        mean, std, median and bootstrap CI per metric
    probs_192x256.npz   uint8 probability maps (used by the disagreement indicators and the figures)

Boundary metrics are computed on a crop around the union of both masks, which is exact and keeps
full-resolution distances tractable.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.cache import CacheArrays
from .data.dataset import SegDataset
from .metrics import image_metrics, pooled_metrics
from .stats import bootstrap_ci
from .utils.io import write_json

METRICS = ["dice", "iou", "sensitivity", "specificity", "accuracy", "hd", "hd95", "assd",
           "hd_pct", "hd95_pct", "assd_pct", "log_area_ratio"]
DEFAULT_TOL_PX, DEFAULT_TOL_PCT = (2, 5, 10), (0.5, 1, 2)


def _crop_union(pred: np.ndarray, gt: np.ndarray):
    u = pred | gt
    if not u.any():
        return pred, gt
    ys, xs = np.nonzero(u.any(1))[0], np.nonzero(u.any(0))[0]
    y0, y1, x0, x1 = max(ys[0] - 1, 0), min(ys[-1] + 2, u.shape[0]), max(xs[0] - 1, 0), min(xs[-1] + 2, u.shape[1])
    return pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]


def pack_mask(m: np.ndarray) -> tuple:
    """Bit-pack a boolean mask (8x smaller) for inter-process transfer."""
    return ("packed", np.packbits(m.ravel()), m.shape)


def _unpack(m):
    if isinstance(m, tuple) and len(m) == 3 and m[0] == "packed":
        n = int(np.prod(m[2]))
        return np.unpackbits(m[1], count=n).reshape(m[2]).astype(bool)
    return m


def _metrics_job(args):
    image_id, pred, gt = args[0], _unpack(args[1]), _unpack(args[2])
    tol_px, tol_pct = (args[3], args[4]) if len(args) > 3 else (DEFAULT_TOL_PX, DEFAULT_TOL_PCT)
    from .metrics import _surface, log_area_ratio

    region = image_metrics(pred, gt, boundary=False)
    diag = math.hypot(*gt.shape)
    pc, gc = _crop_union(pred, gt)
    d = _surface(pc, gc, tol_px, tol_pct, diag)           # crop is exact; tolerances use the FULL diagonal
    if region["pred_empty"] != region["gt_empty"]:          # policy: distance = full image diagonal
        d.update({"hd": diag, "hd95": diag, "assd": diag})
    d.update({f"{k}_pct": 100 * d[k] / diag for k in ("hd", "hd95", "assd")})
    return {"image_id": image_id, "height": gt.shape[0], "width": gt.shape[1], **region, **d,
            "log_area_ratio": log_area_ratio(pred, gt)}


@torch.no_grad()
def predict_probs(model, cache_dir: Path, cfg: dict, device=None, batch_size: int = 16,
                  image_fn=None) -> tuple[list, np.ndarray]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    ds = SegDataset(cache_dir, cfg, train=False, image_fn=image_fn)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=min(4, int(cfg["data"]["num_workers"])))
    ids, probs = [], []
    for b in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda" and cfg["optim"]["amp"]):
            out = model(b["image"].to(device))
        probs.append(torch.sigmoid(out["logits"].float()).cpu())
        ids += list(b["image_id"])
    return ids, torch.cat(probs)[:, 0].numpy()


def evaluate_probs(ids, probs: np.ndarray, cache_dir: Path, threshold: float = 0.5,
                   workers: int | None = None, device=None, tol_px=DEFAULT_TOL_PX,
                   tol_pct=DEFAULT_TOL_PCT) -> pd.DataFrame:
    """probs: (N, h, w) float in [0,1] at network resolution."""
    arr = CacheArrays(Path(cache_dir))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    rows, chunk = [], 24          # chunks bound the RAM used by original-size masks (ISIC test goes up to ~30 MP)
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        items = list(zip(ids, probs))
        for s0 in range(0, len(items), chunk):
            jobs = []
            for image_id, p in items[s0:s0 + chunk]:
                gt = arr.original_gt(image_id)
                up = F.interpolate(torch.from_numpy(p)[None, None].to(dev), size=gt.shape, mode="bilinear",
                                   align_corners=False)[0, 0]
                pred = (up >= threshold).cpu().numpy()
                if ex is None:
                    jobs.append((image_id, pred, gt, tuple(tol_px), tuple(tol_pct)))
                else:                     # bit-packed: large masks through Windows pipes caused WinError 1450
                    jobs.append((image_id, pack_mask(pred), pack_mask(gt.astype(bool)), tuple(tol_px), tuple(tol_pct)))
            rows += [_metrics_job(j) for j in jobs] if ex is None else list(ex.map(_metrics_job, jobs, chunksize=1))
    finally:
        if ex is not None:
            ex.shutdown()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    out = {"n_images": int(len(df)), "n_pred_empty": int(df["pred_empty"].sum()),
           "n_gt_empty": int(df["gt_empty"].sum())}
    for m in METRICS + [c for c in df.columns if c.startswith("nsd_")]:
        if m not in df.columns:
            continue
        v = df[m].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
        out[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "median": float(np.median(v)),
                  "ci95_low": lo, "ci95_high": hi, "n": int(len(v))}
    out.update(pooled_metrics(df.to_dict("records")))
    return out


def eval_is_current(out_dir: Path, cfg: dict, kind: str = "summary.json") -> bool:
    f = Path(out_dir) / kind
    if not f.exists():
        return False
    import json

    return json.loads(f.read_text()).get("metrics_version", 1) >= int(cfg["eval"].get("metrics_version", 1))


def load_best(model, run_dir: Path) -> dict:
    ck = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return ck


def evaluate_run(model, run_dir: Path, test_caches: dict, cfg: dict, overwrite: bool = False,
                 transform: str | None = None, image_fn=None) -> dict:
    """Load ``best.pt`` into ``model`` and evaluate it on every cache in ``test_caches`` ({name: dir}).
    With ``transform`` and ``image_fn`` the inputs are transformed first and results go to
    ``eval/<name>__<transform>/``. Evaluations with an older ``eval.metrics_version`` are recomputed."""
    import json

    run_dir = Path(run_dir)
    ck = load_best(model, run_dir)
    e = cfg["eval"]
    tol_px, tol_pct = tuple(e.get("nsd_tolerances", DEFAULT_TOL_PX)), tuple(e.get("nsd_tolerances_pct", DEFAULT_TOL_PCT))
    results = {}
    for name, cdir in test_caches.items():
        out_dir = run_dir / "eval" / (name if transform is None else f"{name}__{transform}")
        if eval_is_current(out_dir, cfg) and not overwrite:
            results[name] = json.loads((out_dir / "summary.json").read_text())
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        ids, probs = predict_probs(model, Path(cdir), cfg, image_fn=image_fn)
        np.savez_compressed(out_dir / "probs_192x256.npz", ids=np.array(ids),
                            probs=np.round(probs * 255).astype(np.uint8))
        df = evaluate_probs(ids, probs, Path(cdir), e["threshold"], tol_px=tol_px, tol_pct=tol_pct)
        df.to_csv(out_dir / "per_image.csv", index=False)
        s = summarize(df)
        s.update({"best_epoch": int(ck["epoch"]), "metrics_version": int(e.get("metrics_version", 1)),
                  "transform": transform})
        write_json(s, out_dir / "summary.json")
        results[name] = s
    return results


def evaluate_unlabelled(model, run_dir: Path, name: str, cache_dir: Path, cfg: dict, overwrite: bool = False) -> dict:
    """Domains without masks (BCN20000): probabilities + label-free per-image descriptors."""
    import json

    from .metrics import unlabelled_stats

    out_dir = Path(run_dir) / "eval" / name
    if eval_is_current(out_dir, cfg) and not overwrite:
        return json.loads((out_dir / "summary.json").read_text())
    ck = load_best(model, run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, probs = predict_probs(model, Path(cache_dir), cfg)
    np.savez_compressed(out_dir / "probs_192x256.npz", ids=np.array(ids), probs=np.round(probs * 255).astype(np.uint8))
    df = pd.DataFrame([{"image_id": i, **unlabelled_stats(p, cfg["eval"]["threshold"])} for i, p in zip(ids, probs)])
    df.to_csv(out_dir / "unlabelled.csv", index=False)
    s = {"n_images": int(len(df)), "n_pred_empty": int(df["pred_empty"].sum()),
         **{c: float(df[c].mean()) for c in ("pred_area_frac", "pred_border_frac", "entropy_mean", "entropy_band")},
         "best_epoch": int(ck["epoch"]), "metrics_version": int(cfg["eval"].get("metrics_version", 1)), "labelled": False}
    write_json(s, out_dir / "summary.json")
    return s
