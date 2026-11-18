"""Analyses of the study, computed from run folders and caches (no training).

:func:`run_all` writes ``results/analysis/*.csv``. File prefixes follow the labels of Table 2 of the paper:
G1-G6 primary hypotheses, S1-S3 secondary analyses, S4 post hoc robustness (S5 is in ``annotation_pairs``).

* Per-image values are averaged over seeds before any test.
* ``HAM->ISIC``: size-matched HAM10000 models vs ISIC-trained models on ISIC 2018 test;
  ``ISIC->HAM``: ISIC-trained vs size-matched HAM10000 models on the HAM10000 validation fold.
* Paired gap = in-domain minus out-of-domain model on the same images, equal training size.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as st

from .data.cache import CacheArrays
from .stats import bootstrap_ci, holm, rank_biserial

DIRECTIONS = {  # name: (in-domain train_domain, out-of-domain train_domain, evaluation set)
    "HAM->ISIC": ("isic2018", "ham_matched", "isic2018_test"),
    "ISIC->HAM": ("ham_matched", "isic2018", "ham10000_val"),
}
GAP_METRICS = ["dice", "iou", "hd95_pct", "nsd_0.5pct", "nsd_1pct", "nsd_2pct", "nsd_2px", "nsd_5px", "nsd_10px"]


# ----------------------------------------------------------------------------- collection
def load_exclusions(manifests: Path) -> tuple[set, set]:
    """(evaluation exclusions {(set, image_id)}, BCN exclusions {image_id}) written by verify-duplicates.
    Images whose duplicate was reviewed as 'different' in duplicates_review.csv are kept."""
    m = Path(manifests)
    ev, bcn = set(), set()
    keep = set()
    rev = m / "duplicates_review.csv"
    if rev.exists():
        r = pd.read_csv(rev)
        d = r[r["decision"].astype(str).str.lower().isin(["different", "distinta", "no"])]
        keep = set(zip(d["set_a"], d["id_a"])) | set(zip(d["set_b"], d["id_b"]))
    if (m / "eval_excluded.csv").exists():
        e = pd.read_csv(m / "eval_excluded.csv")
        ev = {(a, b) for a, b in zip(e["set"], e["image_id"]) if (a, b) not in keep}
    if (m / "bcn20000_excluded.csv").exists():
        e = pd.read_csv(m / "bcn20000_excluded.csv")
        bcn = {i for i in e["image_id"] if ("bcn20000", i) not in keep}
    return ev, bcn


def collect(runs: Path, domains, experiments, seeds, sets, transform: str | None = None, exclude: set | None = None) -> pd.DataFrame:
    rows = []
    for d, e, s, ds in itertools.product(domains, experiments, seeds, sets):
        f = Path(runs) / d / e / f"seed{s}" / "eval" / (ds if transform is None else f"{ds}__{transform}") / "per_image.csv"
        if f.exists():
            df = pd.read_csv(f)
            if exclude:
                df = df[[(ds, i) not in exclude for i in df["image_id"]]]
            rows.append(df.assign(domain=d, experiment=e, seed=s, set=ds))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def seed_avg(long: pd.DataFrame, metrics=None) -> pd.DataFrame:
    if not len(long):
        return pd.DataFrame(columns=["domain", "experiment", "set", "image_id", "n_seeds"])
    metrics = [m for m in (metrics or GAP_METRICS + ["log_area_ratio"]) if m in long.columns]
    g = long.groupby(["domain", "experiment", "set", "image_id"])
    out = g[metrics].mean()
    out["n_seeds"] = g.size()
    return out.reset_index()


# ----------------------------------------------------------------------------- G1: paired gap
def gap_tests(avg: pd.DataFrame, metrics=None) -> pd.DataFrame:
    rows = []
    metrics = [m for m in (metrics or GAP_METRICS) if m in avg.columns]
    for direction, (din, dout, ds) in DIRECTIONS.items():
        for e in sorted(avg["experiment"].unique()):
            a = avg[(avg.domain == din) & (avg.experiment == e) & (avg.set == ds)].set_index("image_id")
            b = avg[(avg.domain == dout) & (avg.experiment == e) & (avg.set == ds)].set_index("image_id")
            ids = a.index.intersection(b.index)
            if len(ids) < 5:
                continue
            for m in metrics:
                x, y = a.loc[ids, m].to_numpy(float), b.loc[ids, m].to_numpy(float)
                d = x - y
                p = 1.0 if np.allclose(d, 0) else float(st.wilcoxon(x, y, zero_method="wilcox").pvalue)
                lo, hi = bootstrap_ci(d)
                rows.append({"direction": direction, "set": ds, "experiment": e, "metric": m, "n": len(ids),
                             "in_domain_mean": float(x.mean()), "out_domain_mean": float(y.mean()),
                             "gap_mean": float(d.mean()), "gap_ci95_low": lo, "gap_ci95_high": hi,
                             "p": p, "r_rb": rank_biserial(d)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("metric")["p"].transform(lambda s: pd.Series(holm(s.values), index=s.index))
    return out


def gap_curve(gaps: pd.DataFrame) -> pd.DataFrame:
    """G2a: gap per metric relative to the Dice gap. A gap that shrinks as the
    boundary tolerance grows means the error is mostly WHERE the border is placed."""
    if not len(gaps):
        return gaps
    g = gaps[gaps.metric.str.startswith("nsd_") | (gaps.metric == "dice")]
    piv = g.pivot_table(index=["direction", "experiment"], columns="metric", values="gap_mean")
    for c in piv.columns:
        if c != "dice":
            piv[f"{c}_over_dice"] = piv[c] / piv["dice"]
    return piv.reset_index()


# ----------------------------------------------------------------------------- G2: area bias
def area_bias(avg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in sorted(avg["experiment"].unique()):
        for (dom, ds), sub in avg[avg.experiment == e].groupby(["domain", "set"]):
            v = sub["log_area_ratio"].dropna().to_numpy(float)
            if len(v) < 5:
                continue
            p = 1.0 if np.allclose(v, 0) else float(st.wilcoxon(v).pvalue)
            lo, hi = bootstrap_ci(v)
            rows.append({"experiment": e, "domain": dom, "set": ds, "n": len(v), "median": float(np.median(v)),
                         "mean": float(v.mean()), "ci95_low": lo, "ci95_high": hi, "p": p,
                         "area_ratio_median": float(np.exp(np.median(v)))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].values)
    return out


def sign_reversal(bias: pd.DataFrame) -> pd.DataFrame:
    """Per architecture: sign of the median log area ratio in each out-of-domain direction."""
    rows = []
    for e in sorted(bias["experiment"].unique()) if len(bias) else []:
        b = bias[bias.experiment == e].set_index(["domain", "set"])
        try:
            h2i, i2h = b.loc[("ham_matched", "isic2018_test")], b.loc[("isic2018", "ham10000_val")]
        except KeyError:
            continue
        rows.append({"experiment": e, "HAM->ISIC_median": h2i["median"], "HAM->ISIC_p_holm": h2i["p_holm"],
                     "ISIC->HAM_median": i2h["median"], "ISIC->HAM_p_holm": i2h["p_holm"],
                     "reversal": bool(np.sign(h2i["median"]) == -np.sign(i2h["median"]) != 0
                                      and h2i["p_holm"] < 0.05 and i2h["p_holm"] < 0.05)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- descriptors
def hair_score(img_rgb: np.ndarray, k: int = 17, thr: int = 10) -> float:
    """Fraction of pixels flagged by a morphological black-hat hair detector (17x17 cross, grey level > 10)."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k)))
    return float((bh > thr).mean())


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def gt_shape_descriptors(cache_dir: Path) -> pd.DataFrame:
    """Shape descriptors of the reference masks at original resolution."""
    arr = CacheArrays(Path(cache_dir))
    rows = []
    for iid in arr.meta["image_id"]:
        m = arr.original_gt(iid).astype(np.uint8)
        H, W = m.shape
        r = {"image_id": iid, "gt_area_frac": float(m.mean())}
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        r["gt_n_components"] = len(cnts)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a, per = cv2.contourArea(c), cv2.arcLength(c, True)
            hull = cv2.convexHull(c)
            ha, hp = cv2.contourArea(hull), cv2.arcLength(hull, True)
            x, y, w, h = cv2.boundingRect(c)
            r.update({"gt_compactness": 4 * math.pi * a / per ** 2 if per > 0 else np.nan,
                      "gt_roughness": per / hp if hp > 0 else np.nan,             # 1 = convex-smooth
                      "gt_solidity": a / ha if ha > 0 else np.nan,
                      "gt_margin_pct": 100 * min(x, y, W - x - w, H - y - h) / math.hypot(H, W),
                      "gt_touches_border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())})
        rows.append(r)
    return pd.DataFrame(rows)


def image_descriptors(cache_dir: Path, ref_means: dict[str, list]) -> pd.DataFrame:
    """Per-image acquisition and population descriptors (192x256 inputs, original-size GT).
    ``ref_means``: {train_domain: CIELAB mean of its training images} -> ``color_dist_<domain>``."""
    arr = CacheArrays(Path(cache_dir))
    rows = []
    for i in range(len(arr)):
        iid = arr.meta.at[i, "image_id"]
        img = arr.image(i)
        lab = _lab(img)
        mean = lab.reshape(-1, 3).mean(0)
        r = {"image_id": iid, "brightness_L": float(mean[0]), "hair_score": hair_score(img),
             **{f"color_dist_{d}": float(np.linalg.norm(mean - np.asarray(v))) for d, v in ref_means.items()}}
        if arr.has_masks:
            m = arr.mask(i).astype(bool)
            gt = arr.original_gt(iid) if (arr.dir / "gt_original.npz").exists() else m
            ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
            r.update({"area_frac": float(gt.mean()),
                      "touches_border": bool(gt[0].any() or gt[-1].any() or gt[:, 0].any() or gt[:, -1].any()),
                      "contrast_dE": float(np.linalg.norm(lab[m].mean(0) - lab[ring].mean(0))) if m.any() and ring.any() else np.nan})
        if "dx" in arr.meta.columns:
            r["dx"] = arr.meta.at[i, "dx"]
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- G3/G4: mixed model + dominance
GROUPS = {"acquisition": ["color_dist_src", "brightness_L", "hair_score"],
          "population": ["log_area", "touches_border", "contrast_dE"],
          "architecture": ["C(experiment)"],
          "dataset": ["C(direction)"]}


def out_of_domain_table(avg: pd.DataFrame, desc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows = (architecture, image) for both out-of-domain directions and PH2, with descriptors."""
    parts = []
    combos = [("HAM->ISIC", "ham_matched", "isic2018_test"), ("ISIC->HAM", "isic2018", "ham10000_val"),
              ("HAM->PH2", "ham_matched", "ph2"), ("ISIC->PH2", "isic2018", "ph2")]
    for name, dom, ds in combos:
        a = avg[(avg.domain == dom) & (avg.set == ds)]
        if not len(a) or ds not in desc:
            continue
        dd = desc[ds].copy()
        dd["color_dist_src"] = dd[f"color_dist_{dom}"]
        parts.append(a.merge(dd, on="image_id").assign(direction=name))
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(t):
        t["logit_dice"] = np.log(t["dice"].clip(0.01, 0.99) / (1 - t["dice"].clip(0.01, 0.99)))
        t["log_area"] = np.log(t["area_frac"].clip(1e-4, 1))
        t["touches_border"] = t["touches_border"].astype(float)
        for c in GROUPS["acquisition"] + ["log_area", "contrast_dE"]:
            t[c] = (t[c] - t[c].mean()) / (t[c].std() or 1.0)
        t = t.dropna(subset=["contrast_dE"])
    return t


def _fit(model):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # convergence chatter; the fit is checked below
        for method in ("lbfgs", "powell", "nm"):
            try:
                return model.fit(reml=False, method=method)
            except (np.linalg.LinAlgError, ValueError):
                continue
    raise RuntimeError("mixed model did not converge with lbfgs/powell/nm")


def _r2_marginal(t: pd.DataFrame, terms: list[str], response: str = "logit_dice") -> float:
    import statsmodels.formula.api as smf

    if not terms:
        return 0.0
    f = f"{response} ~ " + " + ".join(terms)
    res = _fit(smf.mixedlm(f, t, groups=t["image_id"]))
    var_f = float(np.var(res.model.exog @ res.fe_params.values))     # variance of the fixed-effect predictor
    var_re = float(res.cov_re.values[0, 0])
    return var_f / (var_f + var_re + float(res.scale))


def _active(t: pd.DataFrame, term: str) -> bool:
    if term.startswith("C("):
        return t[term[2:-1]].nunique() > 1
    return term in t.columns and t[term].std() > 1e-12          # constant columns make X singular


def _groups(t: pd.DataFrame, groups: dict | None = None) -> dict:
    g = {k: [x for x in v if _active(t, x)] for k, v in (groups or GROUPS).items()}
    return {k: v for k, v in g.items() if v}


def mixed_model(t: pd.DataFrame, response: str = "logit_dice", groups: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full model coefficients + general dominance of each factor group on the marginal R2
    (Nakagawa & Schielzeth); random intercept per image."""
    import statsmodels.formula.api as smf

    G = _groups(t, groups)
    terms = [x for g in G.values() for x in g]
    res = _fit(smf.mixedlm(f"{response} ~ " + " + ".join(terms), t, groups=t["image_id"]))
    coef = pd.DataFrame({"coef": res.fe_params, "se": res.bse_fe, "p": res.pvalues[res.fe_params.index]}).reset_index()
    names = list(G)
    r2 = {}
    for k in range(len(names) + 1):
        for sub in itertools.combinations(names, k):
            r2[frozenset(sub)] = _r2_marginal(t, [x for g in sub for x in G[g]], response)
    dom = []
    for g in names:
        others = [n for n in names if n != g]
        incs = []
        for k in range(len(others) + 1):
            incs.append(np.mean([r2[frozenset(s) | {g}] - r2[frozenset(s)] for s in itertools.combinations(others, k)]))
        dom.append({"group": g, "general_dominance": float(np.mean(incs)), "terms": " + ".join(G[g])})
    dom = pd.DataFrame(dom)
    dom["share_of_R2m"] = dom["general_dominance"] / r2[frozenset(names)]
    dom["R2m_full"] = r2[frozenset(names)]
    return coef, dom


# ----------------------------------------------------------------------------- G4: reweighting
POP = ["area_frac", "touches_border", "contrast_dE"]


def reweighting(avg: pd.DataFrame, desc: dict[str, pd.DataFrame], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Gap explained by the lesion population: source in-domain Dice re-weighted (logistic density
    ratio on POP descriptors) to the target population, vs the observed out-of-domain Dice."""
    from sklearn.linear_model import LogisticRegression

    rows = []
    for direction, (din, dout, ds_t) in DIRECTIONS.items():
        ds_s = "ham10000_val" if dout.startswith("ham") else "isic2018_test"   # source in-domain test set
        if ds_s not in desc or ds_t not in desc:
            continue
        xs, xt = desc[ds_s].dropna(subset=POP), desc[ds_t].dropna(subset=POP)
        X = pd.concat([xs[POP], xt[POP]]).astype(float).to_numpy()
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        y = np.r_[np.zeros(len(xs)), np.ones(len(xt))]
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        p = clf.predict_proba(X[: len(xs)])[:, 1]
        w = p / (1 - p) * len(xs) / len(xt)
        w = np.minimum(w, np.quantile(w, 0.99))
        wmap = pd.Series(w, index=xs["image_id"].values)
        for e in sorted(avg["experiment"].unique()):
            src = avg[(avg.domain == dout) & (avg.experiment == e) & (avg.set == ds_s)].set_index("image_id")["dice"]
            tgt = avg[(avg.domain == dout) & (avg.experiment == e) & (avg.set == ds_t)].set_index("image_id")["dice"]
            ids = src.index.intersection(wmap.index)
            if len(ids) < 5 or not len(tgt):
                continue
            s, ww = src.loc[ids].to_numpy(float), wmap.loc[ids].to_numpy(float)
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(s), len(s)); j = rng.integers(0, len(tgt), len(tgt))
                boots.append(np.average(s[i], weights=ww[i]) - tgt.to_numpy()[j].mean())
            rows.append({"direction": direction, "experiment": e, "source_in_domain": float(s.mean()),
                         "source_reweighted": float(np.average(s, weights=ww)), "target_out_domain": float(tgt.mean()),
                         "gap_total": float(s.mean() - tgt.mean()),
                         "gap_explained_population": float(s.mean() - np.average(s, weights=ww)),
                         "gap_remaining": float(np.average(s, weights=ww) - tgt.mean()),
                         "gap_remaining_ci95_low": float(np.quantile(boots, 0.025)),
                         "gap_remaining_ci95_high": float(np.quantile(boots, 0.975)),
                         "ess": float(ww.sum() ** 2 / (ww ** 2).sum()), "n_source": len(ids)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- G3: interventions
def intervention_tests(runs: Path, domains, experiments, seeds, sets, transforms, exclude: set | None = None) -> pd.DataFrame:
    base = seed_avg(collect(runs, domains, experiments, seeds, sets, exclude=exclude), ["dice", "hd95_pct"])
    rows = []
    if not len(base):
        return pd.DataFrame()
    for t in transforms:
        alt = seed_avg(collect(runs, domains, experiments, seeds, sets, transform=t, exclude=exclude), ["dice", "hd95_pct"])
        if not len(alt):
            continue
        m = base.merge(alt, on=["domain", "experiment", "set", "image_id"], suffixes=("", "_tf"))
        for (d, e, ds), g in m.groupby(["domain", "experiment", "set"]):
            x, y = g["dice_tf"].to_numpy(float), g["dice"].to_numpy(float)
            dd = x - y
            p = 1.0 if np.allclose(dd, 0) else float(st.wilcoxon(x, y, zero_method="wilcox").pvalue)
            rows.append({"transform": t, "domain": d, "experiment": e, "set": ds, "n": len(g),
                         "dice": float(y.mean()), "dice_tf": float(x.mean()), "diff": float(dd.mean()),
                         "p": p, "r_rb": rank_biserial(dd)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("transform")["p"].transform(lambda s: pd.Series(holm(s.values), index=s.index))
    return out


# ----------------------------------------------------------------------------- G6: label-free failure prediction
def _load_probs(f: Path) -> dict:
    z = np.load(f)
    return {str(i): p for i, p in zip(z["ids"], z["probs"])}


def _dice_bin(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def indicators(runs: Path, domain: str, experiments, seeds, ds: str) -> pd.DataFrame:
    """Per image: seed disagreement (mean over architectures of the mean pairwise 1-Dice between
    seeds), architecture disagreement (pairwise 1-Dice between seed-averaged masks), and the mean
    binary entropy of the ensemble-mean probability. All at 192x256."""
    probs = {}
    for e in experiments:
        for s in seeds:
            f = Path(runs) / domain / e / f"seed{s}" / "eval" / ds / "probs_192x256.npz"
            if f.exists():
                probs[(e, s)] = _load_probs(f)
    if not probs:
        return pd.DataFrame()
    ids = sorted(set.intersection(*[set(v) for v in probs.values()]))
    rows = []
    for iid in ids:
        per_arch, seed_dis = {}, []
        for e in experiments:
            ps = [probs[(e, s)][iid].astype(np.float32) / 255 for s in seeds if (e, s) in probs]
            if not ps:
                continue
            per_arch[e] = np.mean(ps, 0)
            if len(ps) > 1:
                seed_dis.append(np.mean([1 - _dice_bin(a >= .5, b >= .5) for a, b in itertools.combinations(ps, 2)]))
        archs = list(per_arch.values())
        ens = np.clip(np.mean(archs, 0), 1e-6, 1 - 1e-6)
        rows.append({"image_id": iid,
                     "seed_disagreement": float(np.mean(seed_dis)) if seed_dis else np.nan,
                     "arch_disagreement": float(np.mean([1 - _dice_bin(a >= .5, b >= .5) for a, b in itertools.combinations(archs, 2)]))
                     if len(archs) > 1 else np.nan,
                     "ensemble_entropy": float(np.mean(-(ens * np.log2(ens) + (1 - ens) * np.log2(1 - ens))))})
    return pd.DataFrame(rows)


def aurc(risk: np.ndarray, confidence: np.ndarray) -> float:
    o = np.argsort(-confidence)
    r = risk[o]
    return float(np.mean(np.cumsum(r) / np.arange(1, len(r) + 1)))


def failure_prediction(runs: Path, avg: pd.DataFrame, experiments, seeds, pairs) -> tuple[pd.DataFrame, dict]:
    """``pairs`` = [(train_domain, labelled set), ...]. Returns per-indicator Spearman/AURC and the
    fitted isotonic maps (indicator -> expected Dice) pooled over the labelled out-of-domain pairs."""
    from sklearn.isotonic import IsotonicRegression

    rows, pooled = [], []
    for dom, ds in pairs:
        ind = indicators(runs, dom, experiments, seeds, ds)
        if not len(ind):
            continue
        y = avg[(avg.domain == dom) & (avg.set == ds)].groupby("image_id")["dice"].mean()   # mean over architectures
        t = ind.merge(y.rename("dice"), left_on="image_id", right_index=True)
        pooled.append(t)
        for c in ("seed_disagreement", "arch_disagreement", "ensemble_entropy"):
            v = t[[c, "dice"]].dropna()
            if len(v) < 5:
                continue
            rho, p = st.spearmanr(v[c], v["dice"])
            rows.append({"train_domain": dom, "set": ds, "indicator": c, "n": len(v), "spearman_rho": float(rho), "p": float(p),
                         "aurc": aurc(1 - v["dice"].to_numpy(), -v[c].to_numpy()),
                         "aurc_oracle": aurc(1 - v["dice"].to_numpy(), v["dice"].to_numpy()),
                         "aurc_random": float((1 - v["dice"]).mean())})
    maps = {}
    if pooled:
        P = pd.concat(pooled)
        for c in ("seed_disagreement", "arch_disagreement", "ensemble_entropy"):
            v = P[[c, "dice"]].dropna()
            if len(v) >= 5:
                maps[c] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(v[c], v["dice"])
    return pd.DataFrame(rows), maps


def bcn_risk(runs: Path, manifests: Path, domain: str, experiments, seeds, maps: dict, indicator: str = "arch_disagreement",
             exclude: set | None = None) -> pd.DataFrame:
    """Expected Dice on BCN20000 (no masks) from the isotonic map of one indicator, by stratum."""
    ind = indicators(runs, domain, experiments, seeds, "bcn20000")
    if exclude and len(ind):
        ind = ind[~ind["image_id"].isin(exclude)]
    if not len(ind) or indicator not in maps:
        return pd.DataFrame()
    man = pd.read_csv(Path(manifests) / "bcn20000.csv", keep_default_na=False)
    t = ind.merge(man[["image_id", "diagnosis_1", "anatom_site", "dx"]], on="image_id")
    t["expected_dice"] = maps[indicator].predict(t[indicator].fillna(t[indicator].median()))
    out = []
    for col in ("anatom_site", "diagnosis_1", "dx"):
        g = t.groupby(col)["expected_dice"].agg(["mean", "median", "count"]).reset_index().rename(columns={col: "stratum"})
        out.append(g.assign(variable=col, train_domain=domain, indicator=indicator))
    allrow = pd.DataFrame([{"stratum": "ALL", "mean": t["expected_dice"].mean(), "median": t["expected_dice"].median(),
                            "count": len(t), "variable": "all", "train_domain": domain, "indicator": indicator}])
    return pd.concat([allrow] + out, ignore_index=True)


# ----------------------------------------------------------------------------- domain map
def imagenet_features(cache_dir: Path, n_max: int = 1000, seed: int = 0, device=None, exclude: set | None = None) -> np.ndarray:
    """2048-d global-average-pooled ImageNet ResNet-50 features (model-agnostic domain map)."""
    import torch
    import torchvision

    from .data.transforms import IMAGENET_MEAN, IMAGENET_STD

    arr = CacheArrays(Path(cache_dir))
    idx = np.arange(len(arr))
    if exclude:
        idx = np.array([i for i in idx if arr.meta.at[i, "image_id"] not in exclude], dtype=int)
    if len(idx) > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_max, replace=False))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception:                                   # offline (tests): random weights, still a valid embedding
        net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Identity()
    net = net.to(dev).eval()
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            x = torch.from_numpy(np.stack([arr.image(int(i)) for i in idx[s:s + 64]])).permute(0, 3, 1, 2).float() / 255
            feats.append(net(((x - mean) / std).to(dev)).float().cpu().numpy())
    return np.concatenate(feats)


def fid(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg

    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = linalg.sqrtm(ca @ cb)
    if isinstance(cs, tuple):                       # older SciPy returned (sqrtm, errest) with disp=False
        cs = cs[0]
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * np.real(cs)))


def mmd2(a: np.ndarray, b: np.ndarray, n_max: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = a[rng.permutation(len(a))[:n_max]]; b = b[rng.permutation(len(b))[:n_max]]
    z = np.vstack([a, b])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1) if len(z) <= 600 else _sqdist(z)
    sigma2 = np.median(d2[d2 > 0])
    k = np.exp(-d2 / sigma2)
    n = len(a)
    kaa, kbb, kab = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1)) - 2 * kab.mean())


def _sqdist(z: np.ndarray) -> np.ndarray:
    sq = (z ** 2).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)


def domain_map(caches: Path, sets, n_max: int = 1000, n_pca: int = 64, exclude: dict | None = None) -> pd.DataFrame:
    """FID and MMD between every pair of sets on ImageNet ResNet-50 features projected to ``n_pca``
    principal components (fitted on all sets pooled): with 200-1000 images per set the 2048-d
    covariances are rank-deficient, so FID on raw features would be unstable (PH2 has 200 images)."""
    from sklearn.decomposition import PCA

    exclude = exclude or {}
    feats = {s: imagenet_features(Path(caches) / s, n_max, exclude=exclude.get(s)) for s in sets
             if (Path(caches) / s / "_COMPLETE").exists()}
    if not feats:
        return pd.DataFrame()
    allf = np.vstack(list(feats.values()))
    k = int(min(n_pca, allf.shape[0] - 1, allf.shape[1], min(len(v) for v in feats.values()) - 1))
    pca = PCA(n_components=max(k, 1), random_state=0).fit(allf)
    feats = {s: pca.transform(v) for s, v in feats.items()}
    rows = []
    for a, b in itertools.combinations(feats, 2):
        rows.append({"set_a": a, "set_b": b, "fid_pca": fid(feats[a], feats[b]), "mmd2_rbf": mmd2(feats[a], feats[b]),
                     "n_a": len(feats[a]), "n_b": len(feats[b]), "pca_dims": feats[a].shape[1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- secondary analyses S1-S3
GAP_GROUPS = {"acquisition": ["color_shift", "brightness_L", "hair_score"],
              "population": ["log_area", "touches_border", "contrast_dE"],
              "architecture": ["C(experiment)"],
              "dataset": ["C(direction)"]}


def gap_table(avg: pd.DataFrame, desc: dict[str, pd.DataFrame], metric: str = "dice") -> pd.DataFrame:
    """Rows = (architecture, image, direction) with the PAIRED gap (in-domain minus out-of-domain model,
    same image, seed-averaged) and the image descriptors. ``color_shift`` = colour distance to the
    out-of-domain model's training mean minus distance to the in-domain model's training mean."""
    parts = []
    for direction, (din, dout, ds) in DIRECTIONS.items():
        if ds not in desc:
            continue
        a = avg[(avg.domain == din) & (avg.set == ds)][["experiment", "image_id", metric]].rename(columns={metric: "m_in"})
        b = avg[(avg.domain == dout) & (avg.set == ds)][["experiment", "image_id", metric]].rename(columns={metric: "m_out"})
        m = a.merge(b, on=["experiment", "image_id"])
        if not len(m):
            continue
        d = desc[ds].copy()
        d["color_shift"] = d[f"color_dist_{dout}"] - d[f"color_dist_{din}"]
        parts.append(m.merge(d, on="image_id").assign(direction=direction))
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(t):
        t["gap"] = t["m_in"] - t["m_out"]
        t["log_area"] = np.log(t["area_frac"].clip(1e-4, 1))
        t["touches_border"] = t["touches_border"].astype(float)
        for c in ("color_shift", "brightness_L", "hair_score", "log_area", "contrast_dE"):
            t[c] = (t[c] - t[c].mean()) / (t[c].std() or 1.0)
        t = t.dropna(subset=["contrast_dE"])
    return t


def secondary_gap_isic_val(avg: pd.DataFrame, metrics=("dice", "nsd_1pct", "log_area_ratio")) -> pd.DataFrame:
    """Paired gap on the isic2018 VALIDATION partition (annotation source = ISIC train): in-domain
    isic2018 models vs ham_matched models. Note: that partition was used for early stopping of the
    isic2018 models (same small optimistic bias as the HAM validation fold)."""
    rows = []
    for e in sorted(avg["experiment"].unique()):
        a = avg[(avg.domain == "isic2018") & (avg.experiment == e) & (avg.set == "isic2018_val")].set_index("image_id")
        b = avg[(avg.domain == "ham_matched") & (avg.experiment == e) & (avg.set == "isic2018_val")].set_index("image_id")
        ids = a.index.intersection(b.index)
        if len(ids) < 5:
            continue
        for m in metrics:
            if m not in a.columns:
                continue
            x, y = a.loc[ids, m].to_numpy(float), b.loc[ids, m].to_numpy(float)
            d = x - y
            p = 1.0 if np.allclose(d, 0) else float(st.wilcoxon(x, y, zero_method="wilcox").pvalue)
            lo, hi = bootstrap_ci(d)
            rows.append({"direction": "HAM->ISICval", "experiment": e, "metric": m, "n": len(ids),
                         "in_domain_mean": float(x.mean()), "out_domain_mean": float(y.mean()), "gap_mean": float(d.mean()),
                         "gap_ci95_low": lo, "gap_ci95_high": hi, "p": p, "r_rb": rank_biserial(d)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("metric")["p"].transform(lambda s_: pd.Series(holm(s_.values), index=s_.index))
    return out


def gt_annotation_descriptors(cache_dir: Path, ids=None, long_side: int = 600) -> pd.DataFrame:
    """S3: annotation descriptors computed on the reference masks resized
    (nearest) so that its longer side is ``long_side`` px -- removes the resolution dependence of contour
    measures -- plus a model-independent TIGHTNESS proxy on the 192x256 image:
    ``edge_darkness_out`` = L*(far skin ring, 8-16 px outside the mask) - L*(near band, 1-3 px outside).
    A tight mask leaves pigmented lesion just outside it (near band darker -> larger value)."""
    arr = CacheArrays(Path(cache_dir))
    wanted = None if ids is None else set(ids)
    rows = []
    for i in range(len(arr)):
        iid = arr.meta.at[i, "image_id"]
        if wanted is not None and iid not in wanted:
            continue
        g = arr.original_gt(iid).astype(np.uint8)
        H, W = g.shape
        s_ = long_side / max(H, W)
        g = cv2.resize(g, (max(1, round(W * s_)), max(1, round(H * s_))), interpolation=cv2.INTER_NEAREST)
        r = {"image_id": iid, "gt_area_frac": float(g.mean())}
        cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        r["gt_n_components"] = len(cnts)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a_, per = cv2.contourArea(c), cv2.arcLength(c, True)
            hull = cv2.convexHull(c)
            ha, hp = cv2.contourArea(hull), cv2.arcLength(hull, True)
            r.update({"gt_compactness": 4 * math.pi * a_ / per ** 2 if per > 0 else np.nan,
                      "gt_roughness": per / hp if hp > 0 else np.nan,
                      "gt_solidity": a_ / ha if ha > 0 else np.nan})
        m = arr.mask(i).astype(np.uint8)
        if m.any() and not m.all():
            L = _lab(arr.image(i))[..., 0]
            k = lambda n: np.ones((2 * n + 1, 2 * n + 1), np.uint8)
            near = (cv2.dilate(m, k(3)) > 0) & ~(cv2.dilate(m, k(0)) > 0)
            far = (cv2.dilate(m, k(16)) > 0) & ~(cv2.dilate(m, k(8)) > 0)
            inner = (m > 0) & ~(cv2.erode(m, k(3)) > 0)
            core = cv2.erode(m, k(8)) > 0
            if near.any() and far.any():
                r["edge_darkness_out"] = float(L[far].mean() - L[near].mean())
            if inner.any() and core.any():
                r["edge_contrast_in"] = float(L[inner].mean() - L[core].mean())
        rows.append(r)
    return pd.DataFrame(rows)


GT_DESC = ["gt_area_frac", "gt_compactness", "gt_roughness", "gt_solidity", "edge_darkness_out", "edge_contrast_in"]


def gt_annotation_comparison(desc: pd.DataFrame, ref: str = "isic2018_test") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-set medians/IQR of the annotation descriptors and Mann-Whitney tests of every set against
    ``ref`` (Holm within descriptor)."""
    summ = []
    for (ds), g in desc.groupby("set"):
        for c in GT_DESC:
            if c in g:
                v = g[c].dropna()
                summ.append({"set": ds, "descriptor": c, "n": len(v), "median": float(v.median()) if len(v) else np.nan,
                             "q25": float(v.quantile(.25)) if len(v) else np.nan, "q75": float(v.quantile(.75)) if len(v) else np.nan})
    tests = []
    r = desc[desc["set"] == ref]
    for ds in sorted(set(desc["set"]) - {ref}):
        o = desc[desc["set"] == ds]
        for c in GT_DESC:
            if c not in desc:
                continue
            x, y = o[c].dropna(), r[c].dropna()
            if len(x) < 5 or len(y) < 5:
                continue
            u = st.mannwhitneyu(x, y, alternative="two-sided")
            tests.append({"set": ds, "vs": ref, "descriptor": c, "median_set": float(x.median()), "median_ref": float(y.median()),
                          "p": float(u.pvalue), "auc": float(u.statistic / (len(x) * len(y)))})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = tests.groupby("descriptor")["p"].transform(lambda s_: pd.Series(holm(s_.values), index=s_.index))
    return pd.DataFrame(summ), tests


def run_secondary(P, avg: pd.DataFrame, desc: dict, out: Path) -> dict:
    res = {}
    prim = avg[avg["set"] != "isic2018_val"]
    for metric in ("dice", "nsd_1pct"):
        if metric not in prim.columns:
            continue
        t = gap_table(prim, desc, metric)
        if len(t) and t["direction"].nunique() >= 1:
            coef, dom = mixed_model(t, "gap", GAP_GROUPS)
            coef.to_csv(out / f"S1_gap_model_coefficients_{metric}.csv", index=False)
            dom.to_csv(out / f"S1_gap_dominance_{metric}.csv", index=False)
            res[f"gap_dominance_{metric}"] = dom.set_index("group")["share_of_R2m"].round(3).to_dict()
    if "isic2018_val" in set(avg["set"]):
        secondary_gap_isic_val(avg).to_csv(out / "S2_gap_isic_val.csv", index=False)
        area_bias(avg[avg["set"] == "isic2018_val"]).to_csv(out / "S2_area_bias_isic_val.csv", index=False)
    parts = []
    for ds in ("ham10000_val", "isic2018_test", "ph2", "isic2018_val"):
        if (P.caches_local / ds / "_COMPLETE").exists():
            parts.append(gt_annotation_descriptors(P.caches_local / ds).assign(set=ds))
    sp = P.manifests / "isic2018_split.csv"
    if sp.exists() and (P.caches_local / "isic2018_pool" / "_COMPLETE").exists():
        tr = pd.read_csv(sp).query("split == 'train'")["image_id"]
        parts.append(gt_annotation_descriptors(P.caches_local / "isic2018_pool", tr).assign(set="isic2018_train"))
    if parts:
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(out / "S3_gt_annotation_descriptors.csv", index=False)
        summ, tests = gt_annotation_comparison(d)
        summ.to_csv(out / "S3_gt_annotation_summary.csv", index=False)
        tests.to_csv(out / "S3_gt_annotation_tests.csv", index=False)
    return res


# ----------------------------------------------------------------------------- S4: robustness to gross failures
def area_bias_no_failures(avg: pd.DataFrame, thresholds=(0.7, 0.8, 0.9)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Area bias recomputed after removing gross failures. For each (architecture, set) an image is kept only if
    every training domain reaches seed-averaged Dice >= threshold on it, so the three domains are compared on the
    same images. Post hoc analysis, reported separately; Holm correction within each threshold.
    Returns (bias table, failure-rate table)."""
    rows = []
    for thr in thresholds:
        keep = []
        for (e, ds), sub in avg.groupby(["experiment", "set"]):
            piv = sub.pivot_table(index="image_id", columns="domain", values="dice")
            ok = piv.index[(piv >= thr).all(axis=1) & piv.notna().all(axis=1)]
            keep.append(sub[sub["image_id"].isin(ok)])
        k = pd.concat(keep) if keep else avg.iloc[:0]
        b = area_bias(k)
        if len(b):
            n_all = avg.groupby(["experiment", "domain", "set"])["image_id"].nunique().rename("n_all")
            b = b.join(n_all, on=["experiment", "domain", "set"]).assign(threshold=thr)
            b["share_kept"] = b["n"] / b["n_all"]
            rows.append(b)
    bias = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fr = (avg.assign(**{f"fail_{int(t * 100)}": avg["dice"] < t for t in thresholds})
          .groupby(["experiment", "domain", "set"])[[f"fail_{int(t * 100)}" for t in thresholds]].mean().reset_index())
    return bias, fr


def run_robustness(P, out: Path | None = None) -> dict:
    """S4 from the saved seed-averaged table (``analyze`` need not be re-run)."""
    out = Path(out or P.analysis_dir)
    avg = pd.read_csv(out / "per_image_seed_avg.csv")
    bias, fr = area_bias_no_failures(avg)
    bias.to_csv(out / "S4_area_bias_no_failures.csv", index=False)
    fr.to_csv(out / "S4_failure_rate.csv", index=False)
    if not len(bias):
        return {}
    summ = (bias.groupby(["threshold", "domain", "set"])
            .agg(median_pct=("area_ratio_median", lambda v: (np.median(v) - 1) * 100), share_kept=("share_kept", "median"))
            .round(3).reset_index())
    summ.to_csv(out / "S4_summary.csv", index=False)
    return {"S4_rows": len(bias)}


# ----------------------------------------------------------------------------- orchestration
def run_all(cfg: dict, experiments=None, seeds=(42, 43, 44), out: Path | None = None, transforms=("sog", "reinhard", "hairless"),
            with_domain_map: bool = True) -> dict:
    from .domains import LABELLED_SETS
    from .interventions import domain_color_stats
    from .paths import from_config
    from .models.registry import MODELS

    P = from_config(cfg)
    experiments = list(experiments or MODELS)
    out = Path(out or P.analysis_dir)
    out.mkdir(parents=True, exist_ok=True)
    doms = ["ham", "ham_matched", "isic2018"]
    ev_ex, bcn_ex = load_exclusions(P.manifests)
    sets_all = list(LABELLED_SETS) + ["isic2018_val"]          # isic2018_val: secondary analysis S2
    long = collect(P.runs, doms, experiments, seeds, sets_all, exclude=ev_ex)
    if not len(long):
        raise FileNotFoundError("no evaluated runs found")
    long.to_csv(out / "per_image_long.csv", index=False)
    avg_all = seed_avg(long)
    avg_all.to_csv(out / "per_image_seed_avg.csv", index=False)
    avg = avg_all[avg_all["set"].isin(LABELLED_SETS)]           # primary analyses
    res = {"excluded_eval_images": len(ev_ex), "excluded_bcn_images": len(bcn_ex)}
    gaps = gap_tests(avg); gaps.to_csv(out / "G1_gap_tests.csv", index=False); res["G1"] = len(gaps)
    gap_curve(gaps).to_csv(out / "G2_gap_vs_tolerance.csv", index=False)
    bias = area_bias(avg); bias.to_csv(out / "G2_area_bias.csv", index=False)
    rev = sign_reversal(bias); rev.to_csv(out / "G2_sign_reversal.csv", index=False)
    refs = {d: domain_color_stats(P.manifests, P.caches_local, d)["mean"] for d in doms}
    desc = {}
    for ds in list(LABELLED_SETS) + (["bcn20000"] if (P.caches_local / "bcn20000" / "_COMPLETE").exists() else []):
        f = out / f"descriptors_{ds}.csv"
        desc[ds] = pd.read_csv(f) if f.exists() else image_descriptors(P.caches_local / ds, refs)
        desc[ds].to_csv(f, index=False)
        drop = bcn_ex if ds == "bcn20000" else {i for (s_, i) in ev_ex if s_ == ds}
        desc[ds] = desc[ds][~desc[ds]["image_id"].isin(drop)]
    pd.concat([gt_shape_descriptors(P.caches_local / ds).assign(set=ds) for ds in LABELLED_SETS]).to_csv(
        out / "gt_shape_descriptors.csv", index=False)
    t = out_of_domain_table(avg, desc)
    if len(t) and t["experiment"].nunique() >= 1 and t["direction"].nunique() >= 2:
        coef, dom = mixed_model(t)
        coef.to_csv(out / "G3G4_mixed_model_coefficients.csv", index=False)
        dom.to_csv(out / "G3G4_dominance.csv", index=False)
        res["dominance"] = dom.set_index("group")["share_of_R2m"].round(3).to_dict()
    reweighting(avg, desc).to_csv(out / "G4_reweighting.csv", index=False)
    intervention_tests(P.runs, doms, experiments, seeds, LABELLED_SETS, transforms, exclude=ev_ex).to_csv(out / "G3_interventions.csv", index=False)
    fp, maps = failure_prediction(P.runs, avg, experiments, seeds,
                                  [("ham_matched", "isic2018_test"), ("ham_matched", "ph2"), ("isic2018", "ham10000_val"),
                                   ("isic2018", "ph2"), ("ham", "isic2018_test"), ("ham", "ph2")])
    fp.to_csv(out / "G6_failure_prediction.csv", index=False)
    if "bcn20000" in desc:
        pd.concat([bcn_risk(P.runs, P.manifests, d, experiments, seeds, maps, exclude=bcn_ex) for d in doms]).to_csv(out / "G6_bcn_expected_dice.csv", index=False)
    if with_domain_map:
        ex_by_set = {"bcn20000": bcn_ex, **{s_: {i for (t, i) in ev_ex if t == s_} for s_ in LABELLED_SETS}}
        domain_map(P.caches_local, ["ham10000", "isic2018_pool", "isic2018_test", "ph2", "bcn20000"],
                   exclude=ex_by_set).to_csv(out / "domain_map.csv", index=False)
    res.update(run_secondary(P, avg_all, desc, out))
    res.update(run_robustness(P, out))
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res
