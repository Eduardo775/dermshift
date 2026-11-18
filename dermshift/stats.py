"""Paired statistics: Wilcoxon signed-rank test, Holm correction, rank-biserial correlation and
percentile bootstrap confidence intervals of the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = st.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def seed_average(per_image: dict[str, list[pd.DataFrame]], metric: str) -> pd.DataFrame:
    """{model: [per_image df of each seed]} -> DataFrame(index=image_id, columns=model) of seed-averaged values."""
    cols = {}
    for model, dfs in per_image.items():
        stack = pd.concat([d.set_index("image_id")[metric].rename(i) for i, d in enumerate(dfs)], axis=1)
        cols[model] = stack.mean(1)
    out = pd.DataFrame(cols)
    if out.isna().any().any():
        raise ValueError("Models were evaluated on different image sets")
    return out


def paired_tests(table: pd.DataFrame, reference: str, higher_is_better: bool = True) -> pd.DataFrame:
    """Wilcoxon of ``reference`` vs every other column of ``table`` (rows = images)."""
    rows = []
    for other in table.columns:
        if other == reference:
            continue
        d = (table[reference] - table[other]).to_numpy()
        if np.allclose(d, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = st.wilcoxon(table[reference], table[other], zero_method="wilcox", alternative="two-sided")
        lo, hi = bootstrap_ci(d)
        rows.append({"comparison": f"{reference} vs {other}", "baseline": other,
                     "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                     "diff_ci95_low": lo, "diff_ci95_high": hi,
                     "W": float(stat), "p": float(p), "r_rb": rank_biserial(d),
                     "n": int(len(d)), "reference_better": bool((d.mean() > 0) == higher_is_better)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def seed_summary(summaries: list[dict], metric: str) -> dict:
    """summaries: list of summary.json dicts (one per seed)."""
    vals = np.array([s[metric]["mean"] for s in summaries], dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "n_seeds": int(len(vals)), "values": vals.tolist()}
