"""High-level steps used by ``scripts/run.py`` and the tests.

prepare_data        manifests, caches, perceptual-hash screening, partitions, size-matched subset
verify_duplicates   NCC for every cross-set candidate pair and the resulting exclusion lists
run_experiment      train one model / seed / training set (resumable) and evaluate it
evaluate_existing   inference-only evaluation of a finished run (extra sets, test-time transforms)
predict_unlabelled  BCN20000 predictions
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import domains as D
from .config import load_config
from .data import cache as C
from .data.dataset import SegDataset
from .data.manifest import (build_bcn_manifest, build_ham_manifest, build_isic2018_pool_manifest,
                            build_isic2018_test_manifest, build_ph2_manifest)
from .data.split import make_lesion_split, split_summary
from .engine import fit
from .evaluate import eval_is_current, evaluate_run, evaluate_unlabelled
from .models.registry import MODELS, build_model
from .paths import from_config
from .utils.io import write_json
from .utils.repro import seed_everything

LABELLED = D.LABELLED_SETS
TEST_CACHES = ("isic2018_test", "ph2", "ham10000_val", D.POOL)      # caches that keep the original-size masks
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "configs" / "models"
PUBLISHED = ROOT / "data"                       # partitions and reviewed duplicate lists released with the paper
PUBLISHED_FILES = ("ham10000_split.csv", "isic2018_split.csv", "ham10000_matched.csv",
                   "eval_excluded.csv", "bcn20000_excluded.csv", "isic2018_train_test_review.csv")


def install_published(P, source: Path | None = PUBLISHED) -> list[str]:
    """Copy the released partitions and duplicate reviews into ``manifests/`` (existing files are kept)."""
    P.manifests.mkdir(parents=True, exist_ok=True)
    done = []
    for f in PUBLISHED_FILES if source is not None else ():
        src, dst = Path(source) / f, P.manifests / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            done.append(f)
    return done


# ----------------------------------------------------------------------------- data preparation
def _hashes(P, name: str) -> np.ndarray:
    from .data.dedup import phash_array

    f = P.manifests / f"phash_{name}.npy"
    if f.exists():
        return np.load(f)
    a = C.CacheArrays(P.caches_local / name); a._open()
    h = phash_array(a._imgs)
    np.save(f, h)
    return h


def _ids(P, name: str) -> list[str]:
    return pd.read_csv(P.caches_local / name / "meta.csv", keep_default_na=False)["image_id"].tolist()


def prepare_data(cfg: dict, dedup: bool = True, overwrite: bool = False, threads: int = 8,
                 bcn: str = "auto", bcn_max: int | None = None, max_dist: int = 8,
                 published: Path | None = PUBLISHED) -> dict:
    """Build everything the experiments need. ``bcn``: 'auto' (use BCN20000 if present), 'yes' or 'no'.

    Released partitions in ``data/`` are installed first and kept; regenerated partitions are only compared
    with them, so the published study is reproduced exactly."""
    from .data.dedup import near_duplicates, self_pairs

    P = from_config(cfg)
    d = cfg["data"]
    installed = install_published(P, published)
    print("[prepare] listing files and building manifests...", flush=True)
    man = {"ham10000": build_ham_manifest(P), "isic2018_test": build_isic2018_test_manifest(P),
           "ph2": build_ph2_manifest(P), D.POOL: build_isic2018_pool_manifest(P)}
    use_bcn = bcn == "yes" or (bcn == "auto" and (P.bcn_dir / "metadata.csv").exists())
    if use_bcn:
        man["bcn20000"] = build_bcn_manifest(P, bcn_max)
    report = {k: len(v) for k, v in man.items()}
    report["installed_from_data"] = installed
    print("[prepare] manifests:", report, flush=True)

    # HAM10000 lesion-level split
    regenerated = make_lesion_split(man["ham10000"], d["n_folds"], d["val_fold"], d["split_seed"])
    f_split = P.manifests / "ham10000_split.csv"
    if f_split.exists():
        split = pd.read_csv(f_split)
        same = (split.set_index("image_id")["split"].sort_index()
                .equals(regenerated.set_index("image_id")["split"].sort_index()))
        report["ham_split"] = {"source": "existing file", "identical_to_regenerated": bool(same)}
        if set(split["image_id"]) != set(man["ham10000"]["image_id"]):
            raise ValueError("ham10000_split.csv does not match the HAM10000 images on disk")
    else:
        split = regenerated
        split.to_csv(f_split, index=False)
        report["ham_split"] = {"source": "regenerated"}
    split_summary(split).to_csv(P.manifests / "ham10000_split_summary.csv")
    report["split"] = split["split"].value_counts().to_dict()
    man["ham10000_val"] = man["ham10000"][man["ham10000"]["image_id"].isin(split.loc[split.split == "val", "image_id"])]
    man["ham10000_val"] = man["ham10000_val"].assign(dataset="ham10000_val").reset_index(drop=True)
    for name, df in man.items():
        df.to_csv(P.manifests / f"{name}.csv", index=False)

    for name, df in man.items():
        C.build_cache(df, P.datasets_root, P.caches / name, d["height"], d["width"], d["mask_threshold"],
                      keep_original_gt=name in TEST_CACHES, overwrite=overwrite, work_dir=P.caches_local,
                      threads=threads)
        qa = pd.read_csv(P.caches / name / "qa.csv")
        qa.to_csv(P.manifests / f"{name}_qa.csv", index=False)
        if name != "bcn20000":
            report[f"{name}_empty_gt"] = int((qa["lesion_fraction"] == 0).sum())
            report[f"{name}_non_binary_masks"] = int((qa["n_unique_mask_values"] > 2).sum())
        report[f"{name}_sizes"] = (qa["height"].astype(str) + "x" + qa["width"].astype(str)).value_counts().head(5).to_dict()

    if dedup:
        print("[prepare] perceptual-hash screening between all sets...", flush=True)
        names = list(man)
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if {a, b} == {"ham10000", "ham10000_val"}:
                    continue                                   # val is a subset of ham10000
                df = near_duplicates(_hashes(P, a), _ids(P, a), _hashes(P, b), _ids(P, b), max_dist)
                df.to_csv(P.manifests / f"near_duplicates_{a}_vs_{b}.csv", index=False)
                pairs[f"{a}_vs_{b}"] = {"pairs": int(len(df)), "query_images": int(df["query_id"].nunique()) if len(df) else 0}
                if b == D.POOL or a == D.POOL:                 # keep the pool as the query side for exclusions
                    q, r = (a, b) if a == D.POOL else (b, a)
                    if q != a:
                        df = df.rename(columns={"query_id": "ref_id", "ref_id": "query_id"})[["query_id", "ref_id", "hamming"]]
                    df.to_csv(P.manifests / f"near_duplicates_{q}_vs_{r}.csv", index=False)
        report["near_duplicates"] = pairs
        sp = self_pairs(_hashes(P, D.POOL), _ids(P, D.POOL), max_dist)
        sp.to_csv(P.manifests / f"near_duplicates_{D.POOL}_self.csv", index=False)
        ex = D.pool_exclusions(P.manifests)
        psplit = D.make_pool_split(_ids(P, D.POOL), set(ex["image_id"]), sp, 0.1, d["split_seed"])
        psplit = _keep_or_write(psplit, P.manifests / "isic2018_split.csv", report, "isic2018_split")
        n_train = int((psplit["split"] == "train").sum())
        matched = D.make_matched_subset(split, n_train, d["split_seed"])
        matched = _keep_or_write(matched, P.manifests / "ham10000_matched.csv", report, "ham10000_matched_source")
        report["isic2018_partition"] = {"n": len(man[D.POOL]), "excluded": int(len(ex)),
                                   "train": n_train, "val": int((psplit["split"] == "val").sum()),
                                   "self_duplicate_pairs": int(len(sp))}
        report["ham10000_matched"] = {"n": int(len(matched)), "lesions": int(matched["lesion_id"].nunique()),
                                      "dx": matched["dx"].value_counts().to_dict()}
    write_json(report, P.manifests / "data_report.json")
    return report


def _keep_or_write(new: pd.DataFrame, f: Path, report: dict, key: str) -> pd.DataFrame:
    if f.exists():
        old = pd.read_csv(f)
        report[key] = {"source": "existing file", "identical_to_regenerated": bool(set(old["image_id"]) == set(new["image_id"]))}
        return old
    new.to_csv(f, index=False)
    report[key] = {"source": "regenerated"}
    return new


DUP_SETS = ("ham10000", "ham10000_val", "isic2018_pool", "isic2018_test", "ph2", "bcn20000")
LABELLED_EVAL = ("isic2018_test", "ph2", "ham10000_val")


def _pair_files(P) -> list[tuple[str, str, Path]]:
    """One candidate file per unordered pair of sets (whichever orientation prepare wrote)."""
    import itertools

    out = []
    for a, b in itertools.combinations(DUP_SETS, 2):
        if {a, b} == {"ham10000", "ham10000_val"}:
            continue                                            # val is a subset of ham10000
        f1, f2 = P.manifests / f"near_duplicates_{a}_vs_{b}.csv", P.manifests / f"near_duplicates_{b}_vs_{a}.csv"
        if f1.exists():
            out.append((a, b, f1))
        elif f2.exists():
            out.append((b, a, f2))
    return out


def verify_duplicates(cfg: dict, threshold: float = 0.98, sheet_threshold: float = 0.85,
                      max_sheet_pairs: int = 300) -> pd.DataFrame:
    """NCC for every perceptual-hash candidate pair between sets.

    * ``manifests/near_duplicates_<a>_vs_<b>_ncc.csv``  candidates + NCC (max over flips and, when BCN20000
      is involved, also over central square crops: BCN images are 1:1 exports of 4:3 photographs).
    * ``results/duplicates/<a>_vs_<b>/sheet_*.png``   pairs with NCC > ``sheet_threshold`` for visual review.
    * ``manifests/duplicates_review.csv``              pairs with NCC > threshold, ``decision`` = auto (editable).
    * ``manifests/isic2018_train_test_review.csv``     every ISIC train-test candidate, ``decision`` to fill in
                                                       (same / different) for analysis S5.
    * Exclusion lists, applied by the analyses (the evaluation files keep every image):
        - ``bcn20000_excluded.csv``  BCN images with NCC > threshold against any labelled set;
        - ``eval_excluded.csv``      evaluation images (isic2018_test, ph2, ham10000_val) with NCC > threshold
                                     against a TRAINING partition (HAM train fold or ISIC pool train/val).
    ISIC pool images with a candidate in an evaluation set are already removed from training
    (``domains.pool_exclusions``)."""
    from .data.dedup import add_ncc, review_sheets

    P = from_config(cfg)
    split = pd.read_csv(P.manifests / "ham10000_split.csv")
    ham_train = set(split.loc[split.split == "train", "image_id"])
    psplit = pd.read_csv(P.manifests / "isic2018_split.csv")
    pool_train = set(psplit["image_id"])                       # train + val partitions (both used in training)
    rows, review, bcn_ex, ev_ex = [], [], [], []
    for a, b, f in _pair_files(P):
        if not ((P.caches_local / a / "_COMPLETE").exists() and (P.caches_local / b / "_COMPLETE").exists()):
            continue
        pairs = pd.read_csv(f)
        ca, cb = C.CacheArrays(P.caches_local / a), C.CacheArrays(P.caches_local / b)
        pairs = add_ncc(pairs, ca, cb, square="bcn20000" in (a, b))
        pairs.to_csv(P.manifests / f"near_duplicates_{a}_vs_{b}_ncc.csv", index=False)
        hi = pairs[pairs["ncc"] > threshold]
        train_test = (a, b) == (D.POOL, "isic2018_test")       # all candidates are reviewed for analysis S5
        shown = pairs if train_test else pairs[pairs["ncc"] > sheet_threshold]
        sheets = review_sheets(shown, ca, cb, P.results / "duplicates" / f"{a}_vs_{b}",
                               max_pairs=max(max_sheet_pairs, len(shown)) if train_test else max_sheet_pairs) if len(shown) else []
        if train_test and not (P.manifests / "isic2018_train_test_review.csv").exists():
            pairs.assign(decision="").to_csv(P.manifests / "isic2018_train_test_review.csv", index=False)
        rows.append({"set_a": a, "set_b": b, "candidates": len(pairs), f"ncc_gt_{threshold}": len(hi),
                     f"ncc_gt_{sheet_threshold}": int((pairs["ncc"] > sheet_threshold).sum()),
                     "images_a": int(hi["query_id"].nunique()), "images_b": int(hi["ref_id"].nunique()),
                     "sheets": len(sheets)})
        review += [{"set_a": a, "id_a": r.query_id, "set_b": b, "id_b": r.ref_id, "hamming": r.hamming,
                    "ncc": r.ncc, "decision": "auto"} for r in hi.itertuples(index=False)]
        for side, other, col, ocol in ((a, b, "query_id", "ref_id"), (b, a, "ref_id", "query_id")):
            if side == "bcn20000" and other != "bcn20000":
                bcn_ex += [{"image_id": i, "reason": f"NCC>{threshold} vs {other}"} for i in hi[col]]
            if side in LABELLED_EVAL:
                for i, o in zip(hi[col], hi[ocol]):
                    if (other == "ham10000" and o in ham_train) or (other == "isic2018_pool" and o in pool_train):
                        ev_ex.append({"set": side, "image_id": i, "reason": f"NCC>{threshold} vs {other}:{o} (training)"})
    summ = pd.DataFrame(rows)
    P.results.mkdir(parents=True, exist_ok=True)
    summ.to_csv(P.results / "duplicates_summary.csv", index=False)
    rev = P.manifests / "duplicates_review.csv"
    if rev.exists():                                           # keep decisions already entered by hand
        old = pd.read_csv(rev)
        keep = old[old["decision"] != "auto"].set_index(["set_a", "id_a", "set_b", "id_b"])["decision"]
        new = pd.DataFrame(review)
        if len(new) and len(keep):
            idx = new.set_index(["set_a", "id_a", "set_b", "id_b"]).index
            new["decision"] = [keep.get(k, "auto") for k in idx]
        pd.DataFrame(new).to_csv(rev, index=False)
    else:
        pd.DataFrame(review, columns=["set_a", "id_a", "set_b", "id_b", "hamming", "ncc", "decision"]).to_csv(rev, index=False)
    _agg = lambda rows, keys: (pd.DataFrame(rows, columns=keys + ["reason"]).groupby(keys)["reason"]
                               .agg(lambda x: "; ".join(sorted(set(x)))).reset_index())
    _agg(bcn_ex, ["image_id"]).to_csv(P.manifests / "bcn20000_excluded.csv", index=False)
    _agg(ev_ex, ["set", "image_id"]).to_csv(P.manifests / "eval_excluded.csv", index=False)
    return summ


def build_isic_val_cache(cfg: dict, threads: int = 8) -> Path:
    """Cache of the ISIC 2018 validation partition (annotation source of ISIC train) as evaluation set
    ``isic2018_val`` (analysis S2)."""
    P = from_config(cfg)
    d = cfg["data"]
    val = set(pd.read_csv(P.manifests / "isic2018_split.csv").query("split == 'val'")["image_id"])
    man = pd.read_csv(P.manifests / f"{D.POOL}.csv", keep_default_na=False)
    man = man[man["image_id"].isin(val)].assign(dataset="isic2018_val").reset_index(drop=True)
    man.to_csv(P.manifests / "isic2018_val.csv", index=False)
    out = P.caches / "isic2018_val"
    C.build_cache(man, P.datasets_root, out, d["height"], d["width"], d["mask_threshold"], keep_original_gt=True,
                  work_dir=P.caches_local, threads=threads)
    return out


def extra_evaluations(cfg: dict, sets=("isic2018_val",), seeds=(42, 43, 44), domains=("ham", "ham_matched", "isic2018"),
                      root_overrides: dict | None = None) -> pd.DataFrame:
    """Inference-only evaluation of every finished run on additional sets."""
    P = from_config(cfg)
    if "isic2018_val" in sets and not (P.caches_local / "isic2018_val" / "_COMPLETE").exists():
        build_isic_val_cache(cfg)
    rows = []
    for dom in domains:
        for e in MODELS:
            for s in seeds:
                rd = P.run_dir(dom, e, s)
                if not (rd / "DONE.json").exists():
                    continue
                r = evaluate_existing(MODEL_DIR / f"{e}.yaml", s, dom, list(sets), root_overrides=root_overrides)
                rows += [{"domain": dom, "experiment": e, "seed": s, "set": k, "dice": v["dice"]["mean"],
                          "log_area_ratio_median": v["log_area_ratio"]["median"]} for k, v in r.items()]
    return pd.DataFrame(rows)


def stage_local(cfg: dict) -> dict:
    """Copy finished caches to ``paths.local_root`` (only when it is set)."""
    P = from_config(cfg)
    names = [p.name for p in P.caches.iterdir() if (p / "_COMPLETE").exists()] if P.caches.exists() else []
    return {n: C.copy_to_local(P.caches / n, P.caches_local / n) for n in names}


# ----------------------------------------------------------------------------- experiments
def _cfg(exp_path, seed: int, domain: str | None, root_overrides: dict | None, **extra) -> dict:
    ov = {"run.seed": seed, **(root_overrides or {}), **extra}
    if domain is not None:
        ov["data.train_domain"] = domain
    return load_config(exp_path, ov)


def labelled_caches(P, sets) -> dict:
    out = {}
    for s in sets:
        c = P.caches_local / s
        if not (c / "_COMPLETE").exists():
            raise FileNotFoundError(f"cache {c} missing: run prepare first")
        out[s] = c
    return out


def run_experiment(exp_path: str | Path, seed: int, domain: str | None = None, max_minutes: float | None = None,
                   evaluate: bool = True, root_overrides: dict | None = None) -> dict:
    cfg = _cfg(exp_path, seed, domain, root_overrides)
    P = from_config(cfg)
    dom, name = cfg["data"]["train_domain"], cfg["experiment"]
    run_dir = P.run_dir(dom, name, seed)
    cache, tr, va = D.domain_indices(P.manifests, P.caches_local, dom)
    train_ds = SegDataset(P.caches_local / cache, cfg, tr, train=True, seed=seed)
    val_ds = SegDataset(P.caches_local / cache, cfg, va, train=False)
    seed_everything(seed, cfg["run"]["deterministic"])     # model initialisation depends on the seed
    model = build_model(cfg)
    summary = fit(cfg, model, train_ds, val_ds, run_dir, max_minutes=max_minutes)
    if summary["status"] == "paused" or not evaluate:
        return summary
    res = evaluate_run(model, run_dir, labelled_caches(P, cfg["eval"]["sets"]), cfg)
    return {**summary, "eval": {k: {m: v[m]["mean"] for m in ("dice", "iou", "hd95")} for k, v in res.items()}}


def _trained_model(cfg: dict, run_dir: Path):
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"{run_dir} has no best.pt")
    return build_model(cfg)


def evaluate_existing(exp_path: str | Path, seed: int, domain: str = "ham", sets=None, transform: str | None = None,
                      root_overrides: dict | None = None, overwrite: bool = False) -> dict:
    """Inference-only evaluation of a finished run (extra sets, new metrics or a test-time transform)."""
    cfg = _cfg(exp_path, seed, domain, root_overrides, **{"model.pretrained": False})
    P = from_config(cfg)
    run_dir = P.run_dir(domain, cfg["experiment"], seed)
    model = _trained_model(cfg, run_dir)
    sets = list(sets or cfg["eval"]["sets"])
    fn = None
    if transform is not None:
        from .interventions import domain_color_stats, get_transform

        stats = domain_color_stats(P.manifests, P.caches_local, domain) if transform == "reinhard" else None
        fn = get_transform(transform, stats)
    return evaluate_run(model, run_dir, labelled_caches(P, sets), cfg, overwrite=overwrite, transform=transform, image_fn=fn)


def predict_unlabelled(exp_path: str | Path, seed: int, domain: str = "ham", name: str = "bcn20000",
                       root_overrides: dict | None = None, overwrite: bool = False) -> dict:
    cfg = _cfg(exp_path, seed, domain, root_overrides, **{"model.pretrained": False})
    P = from_config(cfg)
    run_dir = P.run_dir(domain, cfg["experiment"], seed)
    model = _trained_model(cfg, run_dir)
    return evaluate_unlabelled(model, run_dir, name, P.caches_local / name, cfg, overwrite=overwrite)


def run_is_complete(P, cfg: dict, domain: str, exp: str, seed: int, sets=None) -> bool:
    rd = P.run_dir(domain, exp, seed)
    sets = sets or cfg["eval"]["sets"]
    return (rd / "DONE.json").exists() and all(eval_is_current(rd / "eval" / s, cfg) for s in sets)
