"""Figures and tables of the paper, generated from the CSV files of ``analysis.run_all``.

``make_figures(results_dir, out_dir)`` writes PDF and PNG (300 dpi) figures and CSV and LaTeX tables.
``annotation_panel(cfg)`` also needs the caches and the saved probability maps.

Colour encodes the training set (fixed order); marker shape repeats it, so the figures remain readable in
greyscale and for colour-blind readers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
DOMAINS = ["ham", "ham_matched", "isic2018"]
DOM_COLOR = {"ham": "#2a78d6", "ham_matched": "#eb6834", "isic2018": "#1baf7a"}
DOM_MARK = {"ham": "o", "ham_matched": "s", "isic2018": "^"}
DOM_LABEL = {"ham": "HAM10000 (full)", "ham_matched": "HAM10000 (size-matched)", "isic2018": "ISIC 2018 train"}
SET_LABEL = {"ham10000_val": "HAM10000 val", "isic2018_val": "ISIC 2018 val", "isic2018_train": "ISIC 2018 train",
             "isic2018_test": "ISIC 2018 test", "ph2": "PH2", "bcn20000": "BCN20000",
             "ham10000": "HAM10000", "isic2018_pool": "ISIC 2018 train"}
MODEL_LABEL = {"unet": "U-Net", "resnet50_unet": "ResNet-50 U-Net", "deeplabv3plus": "DeepLabV3+",
               "attention_unet": "Attention U-Net", "transunet": "TransUNet", "swin_unet": "Swin-Unet",
               "cascade": "Two-stage cascade"}
MODEL_ORDER = list(MODEL_LABEL)
IN_DOMAIN = {("ham", "ham10000_val"), ("ham_matched", "ham10000_val"), ("isic2018", "isic2018_val"),
             ("isic2018", "isic2018_test")}
SINGLE, DOUBLE = 3.5, 7.2          # journal column widths (in)

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig, out: Path) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out.with_suffix(".pdf"), out.with_suffix(".png")]


def _xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _domain_handles(domains=DOMAINS, in_domain=False):
    h = [Line2D([], [], ls="", marker=DOM_MARK[d], ms=4.5, mfc=DOM_COLOR[d], mec="white", mew=0.6, label=DOM_LABEL[d])
         for d in domains]
    if in_domain:
        h.append(Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec=INK, mew=0.9, label="in-domain evaluation"))
    return h


def _read(results: Path, name: str) -> pd.DataFrame | None:
    f = Path(results) / name
    if not f.exists():
        return None
    try:
        d = pd.read_csv(f)
    except pd.errors.EmptyDataError:          # e.g. no intervention evaluations yet
        return None
    return d if len(d) else None


# ----------------------------------------------------------------------------- study design + domain map
def _fid_matrix(dm: pd.DataFrame, order: list[str]) -> np.ndarray:
    M = pd.DataFrame(np.nan, index=order, columns=order)
    for r in dm.itertuples():
        if r.set_a in M.index and r.set_b in M.index:
            M.loc[r.set_a, r.set_b] = M.loc[r.set_b, r.set_a] = r.fid_pca
    return M.to_numpy(float)


def fig_design(results: Path, out: Path):
    """(a) Training x evaluation matrix with the three comparisons of the design; (b) Frechet distances."""
    from matplotlib.patches import FancyBboxPatch

    dm = _read(results, "domain_map.csv")
    s5 = _read(results, "S5_summary.csv")
    n_dup = int(s5["n"].iloc[0]) if s5 is not None else None
    rows = [("ham", "HAM10000 train\n(full)"), ("ham_matched", "HAM10000 train\n(size-matched)"), ("isic2018", "ISIC 2018 train")]
    cols = [("ham10000_val", "HAM10000\nval"), ("isic2018_val", "ISIC 2018\nval"), ("isic2018_test", "ISIC 2018\ntest"),
            ("ph2", "PH2"), ("bcn20000", "BCN20000\n(no masks)")]
    fig = plt.figure(figsize=(DOUBLE, 2.75))
    ax = fig.add_axes([0.0, 0.0, 0.60 if dm is not None else 1.0, 0.86])
    ax.set_xlim(-2.75, len(cols) - 0.35)
    ax.set_ylim(len(rows) + 1.25, -1.35)
    ax.axis("off")
    for j, (_, lab) in enumerate(cols):
        ax.text(j, -0.78, lab, ha="center", va="center", fontsize=6.3)
    for i, (d, lab) in enumerate(rows):
        ax.text(-0.55, i, lab, ha="right", va="center", fontsize=6.3)
        for j, (c, _) in enumerate(cols):
            ax.add_patch(FancyBboxPatch((j - 0.42, i - 0.36), 0.84, 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="#f4f3ef", ec=GRID, lw=0.6))
            if c == "bcn20000":
                ax.text(j, i, "risk\n(label-free)", ha="center", va="center", fontsize=5.4, color=INK2)
                continue
            ax.plot([j], [i], marker=DOM_MARK[d], ms=6.5, color=DOM_COLOR[d], mec="white", mew=0.7)
            if (d, c) in IN_DOMAIN:
                ax.plot([j], [i], marker="o", ms=12, mfc="none", mec=INK, mew=0.9)
    for x in (0.0, 2.0):
        ax.annotate("", xy=(x, 1.22), xytext=(x, 1.78), arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK))
        ax.text(x + 0.1, 1.5, "i", fontsize=6.5, style="italic", va="center")
    ax.annotate("", xy=(1.18, 2.0), xytext=(1.8, 2.0), arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK))
    ax.text(1.49, 2.12, "ii", fontsize=6.5, style="italic", ha="center", va="top")
    notes = ["i   paired per-image gap: same images, in- vs out-of-domain model, equal training size",
             "ii  same model, two ISIC 2018 annotation sources (val: train source; test)",
             "iii " + (f"{n_dup} photographs" if n_dup else "photographs") + " in both ISIC train and test: two masks per image (S5)"]
    for k, t in enumerate(notes):
        ax.text(-2.7, 2.72 + 0.36 * k, t, fontsize=5.8, ha="left", va="center")
    ax.text(-2.7, -1.25, "(a) Design", fontsize=7.5, ha="left")
    ax.plot([], [], ls="", marker="o", ms=8, mfc="none", mec=INK, label="in-domain evaluation")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6, handletextpad=0.3)
    if dm is not None:
        order = [o for o in ["ham10000", "isic2018_test", "isic2018_pool", "ph2", "bcn20000"] if o in set(dm.set_a) | set(dm.set_b)]
        lab = {"ham10000": "HAM10000", "isic2018_test": "ISIC test", "isic2018_pool": "ISIC train", "ph2": "PH2", "bcn20000": "BCN20000"}
        v = _fid_matrix(dm, order)
        bx = fig.add_axes([0.70, 0.14, 0.29, 0.62])
        ramp = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
        vmax = np.nanmax(v)
        bx.imshow(np.where(np.triu(np.ones_like(v, bool)), np.nan, v), cmap=ramp, vmin=0, vmax=vmax)
        for i in range(len(order)):
            for j in range(i):
                bx.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=6.3, color="white" if v[i, j] > 0.55 * vmax else INK)
        bx.set_xticks(range(len(order)), [lab[o] for o in order], rotation=35, ha="right", fontsize=6)
        bx.set_yticks(range(len(order)), [lab[o] for o in order], fontsize=6)
        bx.set_xlim(-0.5, len(order) - 1.5)
        bx.set_ylim(len(order) - 0.5, 0.5)
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.tick_params(length=0)
        bx.set_title("(b) Fréchet distance (ImageNet\nResNet-50 features, 64 PCs)", loc="left", fontsize=7, x=-0.55)
    return _save(fig, out)


# ----------------------------------------------------------------------------- gap vs tolerance
TOL_METRICS = [("dice", "Dice"), ("nsd_0.5pct", "0.5%"), ("nsd_1pct", "1%"), ("nsd_2pct", "2%")]


def fig_gap_vs_tolerance(results: Path, out: Path):
    g = _read(results, "G2_gap_vs_tolerance.csv")
    if g is None:
        return []
    mets = [(m, l) for m, l in TOL_METRICS if m in g.columns]
    x = np.arange(len(mets))
    titles = {"HAM->ISIC": "HAM10000 (size-matched) → ISIC 2018 test", "ISIC->HAM": "ISIC 2018 train → HAM10000 val"}
    dirs = [d for d in titles if d in set(g.direction)]
    fig, axes = plt.subplots(1, len(dirs), figsize=(DOUBLE * 0.72, 2.2), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dirs):
        s = g[g.direction == d]
        Y = s[[m for m, _ in mets]].to_numpy(float) * 100
        nsd = [k for k, (m, _) in enumerate(mets) if m != "dice"]      # Dice is not on the tolerance axis: no line to it
        for y in Y:
            ax.plot(x[nsd], y[nsd], color=MUTED, lw=0.8, alpha=0.7, zorder=2)
            if mets[0][0] == "dice":
                ax.scatter([x[0]], [y[0]], s=6, color=MUTED, zorder=2)
        med = np.median(Y, axis=0)
        ax.plot(x[nsd], med[nsd], color=INK, lw=2, marker="o", ms=4, mfc=INK, mec="white", mew=0.8, zorder=3)
        if mets[0][0] == "dice":
            ax.plot([x[0]], [med[0]], marker="o", ms=4.5, color=INK, mec="white", mew=0.8, zorder=3)
            ax.axvline(0.5, color=GRID, lw=0.8)
        ax.annotate("median of 7 models", (x[-1], med[-1]), xytext=(4, -2), textcoords="offset points",
                    fontsize=6.5, color=INK, va="top")
        ax.set_xticks(x, [l for _, l in mets])
        ax.set_xlabel("                    NSD, tolerance τ")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(titles[d], loc="left", fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlim(-0.3, len(mets) - 0.2)
    np.atleast_1d(axes)[0].set_ylabel("In-domain − out-of-domain\n(points)")
    fig.text(0.99, -0.03, "τ: boundary tolerance as % of the image diagonal", ha="right", fontsize=6, color=INK2)
    fig.tight_layout(w_pad=1.5)
    return _save(fig, out)


# ----------------------------------------------------------------------------- area bias
def fig_area_bias(results: Path, out: Path, bias: pd.DataFrame | None = None, note: str | None = None):
    if bias is None:
        b = _read(results, "G2_area_bias.csv")
        if b is None:
            return []
        s2 = _read(results, "S2_area_bias_isic_val.csv")
        if s2 is not None:
            b = pd.concat([b, s2], ignore_index=True)
    else:
        b = bias.copy()
    b["pct"] = (b["area_ratio_median"] - 1) * 100
    sets = [s for s in ["ham10000_val", "isic2018_val", "isic2018_test", "ph2"] if s in set(b["set"])]
    off = {"ham": -0.24, "ham_matched": 0.0, "isic2018": 0.24}
    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 2.6))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(sets):
        for d in DOMAINS:
            s = b[(b["set"] == ds) & (b["domain"] == d)]
            if not len(s):
                continue
            y = i + off[d]
            ax.scatter(s["pct"], y + rng.uniform(-0.05, 0.05, len(s)), s=7, marker=DOM_MARK[d],
                       color=DOM_COLOR[d], alpha=0.45, lw=0, zorder=3)
            m = float(s["pct"].median())
            ax.plot([s["pct"].min(), s["pct"].max()], [y, y], color=DOM_COLOR[d], lw=1.2, alpha=0.6, zorder=2)
            ax.scatter([m], [y], s=34, marker=DOM_MARK[d], color=DOM_COLOR[d], edgecolor="white", lw=0.8, zorder=4)
            if (d, ds) in IN_DOMAIN:
                ax.scatter([m], [y], s=95, marker="o", facecolor="none", edgecolor=INK, lw=0.9, zorder=5)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(sets)), [SET_LABEL[s] for s in sets])
    ax.set_ylim(len(sets) - 0.5, -0.5)
    ax.set_xlabel("Median area ratio − 1 (%)   (predicted / reference;  > 0 = over-segmentation)")
    _xgrid(ax)
    ax.legend(handles=_domain_handles(in_domain=True), loc="upper left", bbox_to_anchor=(1.0, 1.0),
              title="Training data", title_fontsize=6.8, alignment="left")
    ax.text(1.02, 0.02, "small marks: 7 architectures\nlarge mark: their median" + (f"\n{note}" if note else ""),
            transform=ax.transAxes, fontsize=6.3, color=INK2, va="bottom")
    return _save(fig, out)


def fig_area_bias_no_failures(results: Path, out: Path, threshold: float = 0.8):
    s4 = _read(results, "S4_area_bias_no_failures.csv")
    if s4 is None:
        return []
    b = s4[np.isclose(s4["threshold"], threshold)]
    return fig_area_bias(results, out, bias=b, note=f"images with Dice ≥ {threshold:g}\nfor all three training sets")


def table_robustness(results: Path, out: Path):
    s4 = _read(results, "S4_area_bias_no_failures.csv")
    if s4 is None:
        return []
    base = _read(results, "G2_area_bias.csv")
    s2 = _read(results, "S2_area_bias_isic_val.csv")
    parts = [x.assign(threshold=0.0) for x in (base, s2) if x is not None] + [s4]
    b = pd.concat(parts, ignore_index=True)
    sets = [x for x in ["ham10000_val", "isic2018_val", "isic2018_test", "ph2"] if x in set(b["set"])]
    rows = []
    for d in DOMAINS:
        for thr in sorted(b.threshold.unique()):
            r = {"Training data": DOM_LABEL[d], "Dice filter": "none" if thr == 0 else f"≥ {thr:g}"}
            for ds in sets:
                q = b[(b.domain == d) & (b["set"] == ds) & np.isclose(b.threshold, thr)]
                if len(q):
                    med = (np.median(q.area_ratio_median) - 1) * 100
                    kept = f" ({np.median(q.share_kept) * 100:.0f}%)" if "share_kept" in q and q.share_kept.notna().any() else ""
                    r[SET_LABEL[ds]] = f"{med:+.1f}{kept}"
            rows.append(r)
    cap = ("Post hoc robustness: median over the 7 architectures of the median area ratio minus one (\\%), after "
           "removing images on which any training set gives Dice below the threshold (share of images kept in "
           "parentheses). Declared after the visual panel review.")
    return _write_table(pd.DataFrame(rows), out, cap, "tab:robustness", wide=True)


# ----------------------------------------------------------------------------- GT annotation descriptors
def fig_annotation(results: Path, out: Path):
    s = _read(results, "S3_gt_annotation_summary.csv")
    if s is None:
        return []
    t = _read(results, "S3_gt_annotation_tests.csv")
    panels = [("edge_darkness_out", "Boundary tightness\n(ΔL* skin − 1–3 px outside mask)"),
              ("edge_contrast_in", "Lesion contrast at boundary\n(ΔL*, control)"),
              ("gt_area_frac", "Reference area\n(fraction of image)")]
    panels = [p for p in panels if p[0] in set(s.descriptor)]
    sets = [x for x in ["ham10000_val", "isic2018_train", "isic2018_val", "isic2018_test", "ph2"] if x in set(s["set"])]
    hi = {"isic2018_test", "ph2"}
    col = {x: ("#2a78d6" if x in hi else INK2) for x in sets}
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 1.9), sharey=True)
    for ax, (dsc, lab) in zip(np.atleast_1d(axes), panels):
        for i, x in enumerate(sets):
            r = s[(s["set"] == x) & (s.descriptor == dsc)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([r.q25, r.q75], [i, i], color=col[x], lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([r["median"]], [i], s=26, color=col[x], edgecolor="white", lw=0.7, zorder=3)
            if t is not None and x != "isic2018_test":
                q = t[(t["set"] == x) & (t["descriptor"] == dsc)]
                if len(q) and q.iloc[0].p_holm >= 0.05:
                    ax.annotate("n.s.", (r.q75, i), xytext=(3, 0), textcoords="offset points",
                                va="center", fontsize=6, color=INK2)
        ax.set_xlabel(lab)
        _xgrid(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(sets)), [SET_LABEL[x] for x in sets])
    ax0.set_ylim(len(sets) - 0.5, -0.5)
    handles = [Line2D([], [], color="#2a78d6", marker="o", ms=4, lw=1.6, label="sets over-segmented by every model"),
               Line2D([], [], color=INK2, marker="o", ms=4, lw=1.6, label="other sets")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, -0.02, "median and interquartile range; n.s. = no Holm-significant difference vs ISIC 2018 test",
             ha="right", va="top", fontsize=6, color=INK2)
    fig.tight_layout(w_pad=1.2)
    return _save(fig, out)


# ----------------------------------------------------------------------------- label-free failure indicators
IND = {"seed_disagreement": ("Seed disagreement", "#2a78d6", "o"),
       "arch_disagreement": ("Architecture disagreement", "#eb6834", "s"),
       "ensemble_entropy": ("Ensemble entropy", "#1baf7a", "^")}


def fig_failure(results: Path, out: Path):
    fp = _read(results, "G6_failure_prediction.csv")
    bcn = _read(results, "G6_bcn_expected_dice.csv")
    if fp is None:
        return []
    combos = fp[["train_domain", "set"]].drop_duplicates().reset_index(drop=True)
    has_bcn = bcn is not None and (bcn.variable == "anatom_site").any()
    fig, axes = plt.subplots(2 if has_bcn else 1, 1, figsize=(DOUBLE * 0.8, 4.4 if has_bcn else 2.2),
                             gridspec_kw={"height_ratios": [1.2, 1]} if has_bcn else None)
    ax = np.atleast_1d(axes)[0]
    for i, r in combos.iterrows():
        s = fp[(fp.train_domain == r.train_domain) & (fp["set"] == r["set"])]
        o, rnd = s.aurc_oracle.iloc[0], s.aurc_random.iloc[0]
        ax.plot([o, rnd], [i, i], color=GRID, lw=3, solid_capstyle="butt", zorder=1)
        ax.plot([o], [i], marker="|", ms=9, mew=1.4, color=INK, zorder=2)
        ax.plot([rnd], [i], marker="|", ms=9, mew=1.4, color=MUTED, zorder=2)
        for j, (k, (lab, c, mk)) in enumerate(IND.items()):
            v = s[s.indicator == k]
            if len(v):
                ax.scatter(v.aurc, [i + (j - 1) * 0.2], marker=mk, s=24, color=c, edgecolor="white", lw=0.6, zorder=3)
    ax.set_yticks(range(len(combos)), [f"{DOM_LABEL[r.train_domain]} → {SET_LABEL[r['set']]}" for _, r in combos.iterrows()])
    ax.set_ylim(len(combos) - 0.5, -0.5)
    ax.set_xlabel("Area under the risk–coverage curve (lower is better)")
    _xgrid(ax)
    h = [Line2D([], [], ls="", marker=mk, ms=4.5, mfc=c, mec="white", label=lab) for lab, c, mk in IND.values()]
    h += [Line2D([], [], ls="", marker="|", ms=8, mew=1.4, color=INK, label="oracle"),
          Line2D([], [], ls="", marker="|", ms=8, mew=1.4, color=MUTED, label="random")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("(a) Label-free failure indicators", loc="left", fontsize=7.5)
    if has_bcn:
        ax = axes[1]
        b = bcn[bcn.variable == "anatom_site"].copy()
        b["stratum"] = b["stratum"].fillna("Not recorded")
        order = (b[b.train_domain == "ham"].sort_values("mean")["stratum"].tolist()
                 if (b.train_domain == "ham").any() else sorted(b.stratum.unique()))
        off = {"ham": -0.2, "ham_matched": 0.0, "isic2018": 0.2}
        for d in DOMAINS:
            s = b[b.train_domain == d].set_index("stratum")
            for i, st in enumerate(order):
                if st in s.index:
                    ax.scatter([s.loc[st, "mean"]], [i + off[d]], marker=DOM_MARK[d], s=24, color=DOM_COLOR[d],
                               edgecolor="white", lw=0.6, zorder=3)
            allv = bcn[(bcn.variable == "all") & (bcn.train_domain == d)]
            if len(allv):
                ax.axvline(allv["mean"].iloc[0], color=DOM_COLOR[d], lw=0.8, ls=(0, (3, 2)), zorder=1)
        n = b.drop_duplicates("stratum").set_index("stratum")["count"]
        ax.set_yticks(range(len(order)), [f"{st} (n={int(n[st]):,})" for st in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel("Expected Dice on BCN20000 (from architecture disagreement)")
        _xgrid(ax)
        ax.legend(handles=_domain_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  title="Training data\n(dashed line: all images)", title_fontsize=6.5, alignment="left")
        ax.set_title("(b) Expected Dice on BCN20000 by anatomical site", loc="left", fontsize=7.5)
    fig.tight_layout(h_pad=1.5)
    return _save(fig, out)


# ----------------------------------------------------------------------------- test-time interventions
TF_LABEL = {"sog": "Shades of Gray (p = 6)", "reinhard": "Reinhard colour transfer", "hairless": "Hair removal"}


def fig_interventions(results: Path, out: Path):
    iv = _read(results, "G3_interventions.csv")
    if iv is None or not len(iv):
        return []
    tfs = [t for t in TF_LABEL if t in set(iv["transform"])]
    sets = [s for s in ["ham10000_val", "isic2018_test", "ph2"] if s in set(iv["set"])]
    off = {"ham": -0.22, "ham_matched": 0.0, "isic2018": 0.22}
    fig, axes = plt.subplots(1, len(tfs), figsize=(DOUBLE, 1.8), sharey=True, sharex=True)
    rng = np.random.default_rng(1)
    for ax, tf in zip(np.atleast_1d(axes), tfs):
        for i, ds in enumerate(sets):
            for d in DOMAINS:
                s = iv[(iv["transform"] == tf) & (iv["set"] == ds) & (iv.domain == d)]
                if not len(s):
                    continue
                sig = s.p_holm < 0.05
                y = i + off[d] + rng.uniform(-0.04, 0.04, len(s))
                ax.scatter(s["diff"][sig] * 100, y[sig.to_numpy()], marker=DOM_MARK[d], s=12, color=DOM_COLOR[d], lw=0, zorder=3)
                ax.scatter(s["diff"][~sig] * 100, y[~sig.to_numpy()], marker=DOM_MARK[d], s=12, facecolor="none",
                           edgecolor=DOM_COLOR[d], lw=0.7, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(TF_LABEL[tf], loc="left", fontsize=7)
        ax.set_xlabel("Δ Dice with − without (points)")
        _xgrid(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(sets)), [SET_LABEL[s] for s in sets])
    ax0.set_ylim(len(sets) - 0.5, -0.5)
    h = _domain_handles() + [Line2D([], [], ls="", marker="o", ms=4, mfc="none", mec=INK2, label="not significant (Holm)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1)
    return _save(fig, out)


# ----------------------------------------------------------------------------- tables
def _fmt(v, d=1):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


def _latex(df: pd.DataFrame, caption: str, label: str, align: str | None = None, wide: bool = False) -> str:
    align = align or "l" + "r" * (df.shape[1] - 1)
    esc = lambda s: str(s).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("→", "$\\rightarrow$")
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\small", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(esc(c) for c in df.columns) + " \\\\",
             "\\midrule"]
    lines += [" & ".join(esc(v) for v in row) + " \\\\" for row in df.astype(str).itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def _write_table(df: pd.DataFrame, out: Path, caption: str, label: str, wide: bool = False) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".tex").write_text(_latex(df, caption, label, wide=wide), encoding="utf-8")
    return [out.with_suffix(".csv"), out.with_suffix(".tex")]


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def table_gaps(results: Path, out: Path):
    g = _read(results, "G1_gap_tests.csv")
    if g is None:
        return []
    s2 = _read(results, "S2_gap_isic_val.csv")
    blocks = [("HAM->ISIC", "isic2018_test", g, "HAM→ISIC test"), ("ISIC->HAM", "ham10000_val", g, "ISIC→HAM val")]
    if s2 is not None:
        blocks.append((s2.direction.iloc[0], None, s2, "HAM→ISIC val"))
    rows = []
    for e in MODEL_ORDER:
        r = {"Model": MODEL_LABEL[e]}
        for dirn, ds, src, lab in blocks:
            for m, ml in [("dice", "Dice"), ("nsd_1pct", "NSD 1%")]:
                q = src[(src.direction == dirn) & (src.experiment == e) & (src.metric == m)]
                if ds is not None:
                    q = q[q["set"] == ds]
                if len(q):
                    q = q.iloc[0]
                    r[f"{lab} {ml}"] = f"{q.gap_mean * 100:.1f} [{q.gap_ci95_low * 100:.1f}, {q.gap_ci95_high * 100:.1f}]{_stars(q.p_holm)}"
        rows.append(r)
    df = pd.DataFrame(rows)
    cap = ("Paired per-image gap (in-domain minus out-of-domain, points; mean and 95\\% bootstrap CI) "
           "after averaging three seeds per image. Out-of-domain models are trained on the size-matched HAM10000 "
           "subset. Holm-adjusted Wilcoxon: * $p<0.05$, ** $p<0.01$, *** $p<0.001$.")
    return _write_table(df, out, cap, "tab:gaps", wide=True)


def table_area_bias(results: Path, out: Path):
    b = _read(results, "G2_area_bias.csv")
    if b is None:
        return []
    s2 = _read(results, "S2_area_bias_isic_val.csv")
    if s2 is not None:
        b = pd.concat([b, s2], ignore_index=True)
    sets = [s for s in ["ham10000_val", "isic2018_val", "isic2018_test", "ph2"] if s in set(b["set"])]
    rows = []
    for d in DOMAINS:
        for e in MODEL_ORDER:
            r = {"Training data": DOM_LABEL[d], "Model": MODEL_LABEL[e]}
            for ds in sets:
                q = b[(b.domain == d) & (b.experiment == e) & (b["set"] == ds)]
                if len(q):
                    q = q.iloc[0]
                    r[SET_LABEL[ds]] = f"{(q.area_ratio_median - 1) * 100:+.1f}{_stars(q.p_holm)}"
            rows.append(r)
    cap = ("Median area ratio minus one (\\%; predicted/reference area, seed-averaged per image). Positive values "
           "mean over-segmentation. Holm-adjusted one-sample Wilcoxon of the log area ratio against 0.")
    return _write_table(pd.DataFrame(rows), out, cap, "tab:area_bias", wide=True)


def table_dominance(results: Path, out: Path):
    src = [("G3G4_dominance.csv", "Dice (out-of-domain)"), ("S1_gap_dominance_dice.csv", "Paired gap, Dice"),
           ("S1_gap_dominance_nsd_1pct.csv", "Paired gap, NSD 1%")]
    cols, r2 = {}, {}
    for f, lab in src:
        d = _read(results, f)
        if d is None:
            continue
        cols[lab] = d.set_index("group")["share_of_R2m"]
        r2[lab] = d["R2m_full"].iloc[0]
    if not cols:
        return []
    groups = ["acquisition", "population", "architecture", "dataset"]
    df = pd.DataFrame(cols)
    df = df.reindex([g for g in groups if g in df.index] + [g for g in df.index if g not in groups])
    out_df = pd.DataFrame({"Factor group": [g.capitalize() for g in df.index]})
    for lab in df.columns:
        out_df[lab] = [_fmt(v * 100) for v in df[lab]]
    out_df.loc[len(out_df)] = ["Marginal $R^2$"] + [f"{r2[lab]:.3f}" for lab in df.columns]
    cap = ("General dominance analysis of the mixed-effects models (random intercept per image): share of the "
           "marginal $R^2$ (\\%) attributed to each factor group.")
    return _write_table(out_df, out, cap, "tab:dominance")


def table_reweighting(results: Path, out: Path):
    r = _read(results, "G4_reweighting.csv")
    if r is None:
        return []
    df = pd.DataFrame({"Direction": r.direction.str.replace("->", "→"), "Model": r.experiment.map(MODEL_LABEL),
                       "Gap": (r.gap_total * 100).map(_fmt), "Explained by population": (r.gap_explained_population * 100).map(_fmt),
                       "Remaining": (r.gap_remaining * 100).map(_fmt)})
    cap = ("Density-ratio reweighting of the source set to the target population (Dice points). "
           "A population effect would make the explained part a large share of the gap.")
    return _write_table(df, out, cap, "tab:reweighting")


# ----------------------------------------------------------------------------- entry points
FIGURES = [("design", fig_design), ("gap_vs_tolerance", fig_gap_vs_tolerance), ("area_bias", fig_area_bias),
           ("annotation_descriptors", fig_annotation), ("failure_indicators", fig_failure),
           ("interventions", fig_interventions), ("area_bias_no_failures", fig_area_bias_no_failures)]
TABLES = [("table_gaps", table_gaps), ("table_area_bias", table_area_bias), ("table_dominance", table_dominance),
          ("table_reweighting", table_reweighting), ("table_robustness", table_robustness)]


def make_figures(results: Path, out: Path) -> dict[str, list[str]]:
    results, out = Path(results), Path(out)
    done = {}
    for name, fn in FIGURES + TABLES:
        files = fn(results, out / name)
        done[name] = [p.name for p in files]
    return done


# ----------------------------------------------------------------------------- annotation panel (workstation)
PANEL_SETS = ("isic2018_val", "isic2018_test", "ph2", "ham10000_val")
PANEL_REF = {"isic2018_test": "isic2018", "isic2018_val": "isic2018", "ph2": "isic2018", "ham10000_val": "ham_matched"}


def select_panel_cases(avg: pd.DataFrame, sets=PANEL_SETS, n_extreme: int = 6, n_random: int = 6, seed: int = 2026,
                       mode: str = "review", experiment: str = "swin_unet", n_typical: int = 2,
                       min_dice: float = 0.8, allowed: dict | None = None) -> pd.DataFrame:
    """``review``: per set, the images most over-segmented (median log area ratio over the 7 architectures of the
    reference training set: ISIC-trained, HAM-trained for HAM val) plus a seeded random sample.
    ``typical``: per set, images where the shown architecture reaches Dice >= ``min_dice`` for both training sets
    (no gross failure) and whose area ratio (reference training set) is closest to that set's median."""
    rows = []
    rng = np.random.default_rng(seed)
    for ds in sets:
        a = avg[(avg["set"] == ds) & (avg.domain == PANEL_REF[ds])]
        if not len(a):
            continue
        if mode == "typical":
            e = avg[(avg["set"] == ds) & (avg.experiment == experiment)]
            piv = e.pivot_table(index="image_id", columns="domain", values="dice")
            cols = [c for c in ("ham_matched", "isic2018") if c in piv.columns]
            ok = piv.index[(piv[cols] >= min_dice).all(axis=1)] if cols else piv.index
            if allowed is not None and ds in allowed:
                ok = ok[ok.isin(allowed[ds])]
            lar = e[(e.domain == PANEL_REF[ds]) & e["image_id"].isin(ok)].set_index("image_id")["log_area_ratio"]
            if not len(lar):
                continue
            target = float(lar.median())
            pick = (lar - target).abs().sort_values().index[:n_typical]
            rows += [{"set": ds, "image_id": i, "kind": "typical", "median_log_area_ratio": lar[i],
                      "set_median_log_area_ratio": target} for i in pick]
            continue
        m = a.groupby("image_id")["log_area_ratio"].median().sort_values(ascending=False)
        ext = list(m.index[:n_extreme])
        rest = [i for i in m.index if i not in ext]
        rnd = list(rng.choice(rest, size=min(n_random, len(rest)), replace=False))
        rows += [{"set": ds, "image_id": i, "kind": "extreme", "median_log_area_ratio": m[i]} for i in ext]
        rows += [{"set": ds, "image_id": i, "kind": "random", "median_log_area_ratio": m[i]} for i in rnd]
    return pd.DataFrame(rows)


def _contour(img: np.ndarray, mask: np.ndarray, color, thickness: int = 1) -> np.ndarray:
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(img, cs, -1, color, thickness)


def _seed_avg_probs(run_dirs, ds: str) -> dict[str, np.ndarray]:
    acc, n = {}, 0
    for rd in run_dirs:
        f = Path(rd) / "eval" / ds / "probs_192x256.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for i, p in zip(z["ids"], z["probs"]):
            acc[str(i)] = acc.get(str(i), 0) + p.astype(np.float32)
        n += 1
    return {k: v / n for k, v in acc.items()} if n else {}


def _case_image(cache, idx, image_id, probs, upscale, crop_pad=None):
    """Input image with the two predictions (training-set colours) and the reference (white) as contours.
    ``crop_pad``: crop around the union of masks with that relative margin (typical-case figure)."""
    import cv2

    rgb = {d: tuple(int(DOM_COLOR[d][k:k + 2], 16) for k in (1, 3, 5)) for d in DOMAINS}
    img = cache.image(idx[image_id])
    gt0 = cache.mask(idx[image_id]) > 0
    preds0 = {d: probs[d][image_id] >= 127.5 for d in ("ham_matched", "isic2018") if image_id in probs[d]}
    if crop_pad is not None:
        u = gt0.copy()
        for pm in preds0.values():
            u |= pm
        ys, xs = np.nonzero(u)
        if len(ys):
            h, w = u.shape
            py, px = int((ys.max() - ys.min() + 1) * crop_pad) + 4, int((xs.max() - xs.min() + 1) * crop_pad) + 4
            y0, y1 = max(0, ys.min() - py), min(h, ys.max() + py + 1)
            x0, x1 = max(0, xs.min() - px), min(w, xs.max() + px + 1)
            img, gt0 = img[y0:y1, x0:x1], gt0[y0:y1, x0:x1]
            preds0 = {d: pm[y0:y1, x0:x1] for d, pm in preds0.items()}
    img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    size = (img.shape[1], img.shape[0])
    im = img.copy()
    for d, pm in preds0.items():
        pm = cv2.resize(pm.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) >= 0.5
        im = _contour(im, pm, rgb[d], 2)
    gt = cv2.resize(gt0.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return _contour(im, gt, (255, 255, 255), 2)


def _panel_legend(fig, experiment):
    name = MODEL_LABEL.get(experiment, experiment)
    h = [Line2D([], [], color="#bbbbbb", lw=2, label="reference mask (white)"),
         Line2D([], [], color=DOM_COLOR["ham_matched"], lw=2, label=f"{name} · HAM10000 (size-matched)"),
         Line2D([], [], color=DOM_COLOR["isic2018"], lw=2, label=f"{name} · ISIC 2018 train")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))


def _area_text(per, image_id):
    ar = []
    for d, tag in (("ham_matched", "HAM"), ("isic2018", "ISIC")):
        if (d, image_id) in per.index:
            ar.append(f"{tag} {np.expm1(per.loc[(d, image_id), 'log_area_ratio']) * 100:+.0f}%")
    return ", ".join(ar)


def annotation_panel(cfg: dict, experiment: str = "swin_unet", seeds=(42, 43, 44), out: Path | None = None,
                     n_extreme: int = 6, n_random: int = 6, upscale: int = 2, mode: str = "review",
                     n_typical: int = 2, min_dice: float = 0.8, min_area: float = 0.05) -> dict:
    """Visual check of the boundary convention: reference contour (white) and the seed-averaged prediction of one
    architecture trained on HAM10000 size-matched (orange) and on ISIC 2018 train (aqua).

    * ``mode="review"``: one page per test set (extreme + random cases) and ``panel_cases.csv`` with an empty
      ``review`` column, in ``results/paper/annotation_panel``.
    * ``mode="typical"``: ``results/figures/typical_cases``: one column per set, ``n_typical``
      rows of non-failure cases whose area ratio is closest to the set median, cropped around the lesion; lesions
      covering less than ``min_area`` of the image are skipped (they pixelate when cropped)."""
    from .data.cache import CacheArrays
    from .paths import from_config

    P = from_config(cfg)
    avg = pd.read_csv(P.analysis_dir / "per_image_seed_avg.csv")
    allowed = None
    if mode == "typical" and min_area > 0:          # small lesions turn into pixelated blobs once cropped
        allowed = {}
        for ds in PANEL_SETS:
            if (P.caches_local / ds / "meta.csv").exists():
                c = CacheArrays(P.caches_local / ds)
                allowed[ds] = {str(i) for k, i in enumerate(c.meta.image_id) if (c.mask(k) > 0).mean() >= min_area}
    cases = select_panel_cases(avg, n_extreme=n_extreme, n_random=n_random, mode=mode, experiment=experiment,
                               n_typical=n_typical, min_dice=min_dice, allowed=allowed)
    if not len(cases):
        return {"cases": 0, "files": [], "out": ""}
    data = {}
    for ds in cases["set"].unique():
        cache = CacheArrays(P.caches_local / ds)
        data[ds] = (cache, {str(i): k for k, i in enumerate(cache.meta.image_id)},
                    {d: _seed_avg_probs([P.run_dir(d, experiment, s) for s in seeds], ds) for d in ("ham_matched", "isic2018")},
                    avg[(avg["set"] == ds) & (avg.experiment == experiment)].set_index(["domain", "image_id"]))

    if mode == "typical":
        out = Path(out or P.figures_dir)
        sets = list(dict.fromkeys(cases["set"]))
        fig, axes = plt.subplots(n_typical, len(sets), figsize=(DOUBLE, DOUBLE / len(sets) * 0.8 * n_typical + 0.3))
        axes = np.atleast_2d(axes).reshape(n_typical, len(sets))
        for c, ds in enumerate(sets):
            cache, idx, probs, per = data[ds]
            cs = cases[cases["set"] == ds].reset_index(drop=True)
            for r in range(n_typical):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r >= len(cs) or cs.image_id[r] not in idx:
                    ax.axis("off"); continue
                iid = cs.image_id[r]
                ax.imshow(_case_image(cache, idx, iid, probs, upscale + 1, crop_pad=0.15))
                ax.set_xlabel(f"{iid}\narea: {_area_text(per, iid)}", fontsize=5.5, color=INK2)
                if r == 0:
                    ax.set_title(SET_LABEL[ds], fontsize=7.5)
        _panel_legend(fig, experiment)
        fig.tight_layout()
        files = [str(p) for p in _save(fig, out / "typical_cases")]
        cases.to_csv(out / "typical_cases.csv", index=False)
        return {"cases": len(cases), "files": files, "out": str(out)}

    out = Path(out or P.figures_dir / "annotation_panel")
    pages = []
    for ds, cs in cases.groupby("set", sort=False):
        cache, idx, probs, per = data[ds]
        cs = cs.reset_index(drop=True)
        ncol = 4
        nrow = int(np.ceil(len(cs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, DOUBLE * 0.75 / ncol * nrow * 1.05))
        axes = np.atleast_2d(axes)
        for k, r in cs.iterrows():
            ax = axes[k // ncol, k % ncol]
            if r.image_id not in idx:
                ax.axis("off"); continue
            ax.imshow(_case_image(cache, idx, r.image_id, probs, upscale))
            ax.set_title(f"{r.image_id} · {r.kind}\narea: {_area_text(per, r.image_id)}", fontsize=5.5, color=INK2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        for k in range(len(cs), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        _panel_legend(fig, experiment)
        fig.suptitle(SET_LABEL[ds], x=0.01, ha="left", y=1.04, fontsize=8)
        fig.tight_layout()
        pages += [str(p) for p in _save(fig, out / f"panel_{ds}")]
    cases.assign(review="").to_csv(out / "panel_cases.csv", index=False)
    return {"cases": len(cases), "files": pages, "out": str(out)}
