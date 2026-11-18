"""Command-line interface.

    python scripts/run.py env                  check PyTorch, CUDA and the configured paths
    python scripts/run.py prepare              manifests, caches, duplicate screening, partitions
    python scripts/run.py verify-duplicates    NCC of all cross-set candidates, exclusion lists, review sheets
    python scripts/run.py queue                train and evaluate everything (resumable)
    python scripts/run.py status               progress of the queue
    python scripts/run.py train --model unet --domain isic2018 --seed 42
    python scripts/run.py evaluate --model unet --domain isic2018 --seed 42 [--sets ...] [--transform sog]
    python scripts/run.py analyze              results/analysis/*.csv (G1-G6, S1-S4)
    python scripts/run.py duplicate-masks      S5: train vs test masks of the same photograph
    python scripts/run.py figures              figures and tables of the paper -> results/figures
    python scripts/run.py panel [--mode typical]

Any configuration key can be overridden with ``--set key=value`` (repeatable), e.g. ``--set data.num_workers=0``.
Paths are read from configs/local.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # used when run.deterministic is true
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")     # Windows without developer mode
os.environ.setdefault("PYTHONUTF8", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "configs" / "base.yaml"
MODEL_DIR = ROOT / "configs" / "models"


def _overrides(pairs) -> dict:
    import yaml

    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def _base_cfg(ov):
    from dermshift.config import load_config

    cfg = load_config(BASE, ov)
    if not cfg["paths"]["datasets_root"] or not cfg["paths"]["project_root"]:
        sys.exit("set paths.datasets_root and paths.project_root in configs/local.yaml "
                 "(copy configs/local.example.yaml)")
    return cfg


def _paths(ov):
    from dermshift.paths import from_config

    return from_config(_base_cfg(ov))


def cmd_env(a, ov):
    import torch

    print("python  ", sys.version.split()[0])
    print("torch   ", torch.__version__, "| CUDA build", torch.version.cuda)
    print("cuda    ", torch.cuda.is_available())
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print("gpu     ", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}",
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
        if cap[0] >= 12 and not any(x.startswith("sm_12") for x in torch.cuda.get_arch_list()):
            print("warning: this PyTorch build has no sm_120 kernels; install a CUDA 12.8 (or newer) build")
    for m in ("segmentation_models_pytorch", "timm", "albumentations", "cv2", "statsmodels", "imagehash"):
        try:
            print(f"{m:8s}", getattr(__import__(m), "__version__", "?"))
        except ImportError as e:
            print(f"{m:8s} missing ({e})")
    if (ROOT / "configs" / "local.yaml").exists():
        cfg = _base_cfg(ov)
        for k in ("datasets_root", "project_root"):
            p = Path(cfg["paths"][k])
            print(f"{k:14s} {p}  {'(exists)' if p.exists() else '(not found)'}")
    else:
        print("configs/local.yaml not found: copy configs/local.example.yaml and edit the paths")


def cmd_prepare(a, ov):
    from dermshift.pipeline import prepare_data

    rep = prepare_data(_base_cfg(ov), dedup=not a.no_dedup, overwrite=a.overwrite, threads=a.threads,
                       bcn=a.bcn, bcn_max=a.bcn_max)
    print(json.dumps(rep, indent=1, default=str))


def cmd_verify_duplicates(a, ov):
    from dermshift.pipeline import verify_duplicates

    print(verify_duplicates(_base_cfg(ov), a.threshold).to_string(index=False))


def cmd_queue(a, ov):
    from dermshift.queue import run_queue

    st = run_queue(ov, a.max_minutes, tuple(a.kinds.split(",")) if a.kinds else None)
    print(st.groupby(["kind", "state"]).size().unstack(fill_value=0).to_string())


def cmd_status(a, ov):
    from dermshift.queue import status

    st = status(ov)
    print(st.groupby(["kind", "state"]).size().unstack(fill_value=0).to_string())
    if a.full:
        print(st[st.state != "done"].to_string(index=False))


def cmd_train(a, ov):
    from dermshift.pipeline import run_experiment

    print(run_experiment(MODEL_DIR / f"{a.model}.yaml", a.seed, a.domain, a.max_minutes, root_overrides=ov))


def cmd_evaluate(a, ov):
    from dermshift.pipeline import evaluate_existing, predict_unlabelled

    if a.bcn:
        print(predict_unlabelled(MODEL_DIR / f"{a.model}.yaml", a.seed, a.domain, root_overrides=ov, overwrite=a.overwrite))
        return
    sets = a.sets.split(",") if a.sets else None
    r = evaluate_existing(MODEL_DIR / f"{a.model}.yaml", a.seed, a.domain, sets, a.transform,
                          root_overrides=ov, overwrite=a.overwrite)
    for k, v in r.items():
        print(f"{k:16s} Dice {v['dice']['mean']:.4f}  NSD@1% {v.get('nsd_1pct', {}).get('mean', float('nan')):.4f}")


def cmd_analyze(a, ov):
    from dermshift.analysis import run_all

    print(json.dumps(run_all(_base_cfg(ov), with_domain_map=not a.no_domain_map), indent=1, default=str))


def cmd_duplicate_masks(a, ov):
    from dermshift.annotation_pairs import run_duplicate_annotation

    r = run_duplicate_annotation(_base_cfg(ov), ncc_min=a.ncc_min)
    print(f"{r['pairs']} pairs ({r['selection']}), {r['registered']} registered")
    if len(r["summary"]):
        print(r["summary"].drop(columns=["selection"]).round(4).to_string(index=False))


def cmd_figures(a, ov):
    from dermshift.figures import make_figures

    P = _paths(ov)
    res = Path(a.results) if a.results else P.analysis_dir
    out = Path(a.out) if a.out else P.figures_dir
    for k, v in make_figures(res, out).items():
        print(f"  {k:30s} {', '.join(v) if v else 'skipped (input missing)'}")
    print(f"written to {out}")


def cmd_panel(a, ov):
    from dermshift.figures import annotation_panel

    r = annotation_panel(_base_cfg(ov), experiment=a.model, n_extreme=a.n_extreme, n_random=a.n_random,
                         mode=a.mode, n_typical=a.n_typical, min_dice=a.min_dice, min_area=a.min_area)
    print(f"{r['cases']} cases -> {r['out']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="DermShift: cross-dataset analysis of dermoscopic lesion segmentation")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="configuration override")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    p = sub.add_parser("prepare")
    p.add_argument("--bcn", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--bcn-max", type=int, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("verify-duplicates")
    p.add_argument("--threshold", type=float, default=0.98)
    p = sub.add_parser("queue")
    p.add_argument("--kinds", default="", help="comma-separated subset of train,extra,bcn,tf")
    p.add_argument("--max-minutes", type=float, default=None)
    p = sub.add_parser("status")
    p.add_argument("--full", action="store_true")
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--domain", default="ham", choices=["ham", "ham_matched", "isic2018"])
        p.add_argument("--seed", type=int, required=True)
        if name == "train":
            p.add_argument("--max-minutes", type=float, default=None)
        else:
            p.add_argument("--sets", default="")
            p.add_argument("--transform", default=None, choices=["sog", "reinhard", "hairless"])
            p.add_argument("--bcn", action="store_true", help="predict BCN20000 instead")
            p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("analyze")
    p.add_argument("--no-domain-map", action="store_true")
    p = sub.add_parser("duplicate-masks")
    p.add_argument("--ncc-min", type=float, default=0.98, help="used only when the review sheet is empty")
    p = sub.add_parser("figures")
    p.add_argument("--results", default="")
    p.add_argument("--out", default="")
    p = sub.add_parser("panel")
    p.add_argument("--model", default="swin_unet")
    p.add_argument("--mode", choices=["review", "typical"], default="review")
    p.add_argument("--n-extreme", type=int, default=6)
    p.add_argument("--n-random", type=int, default=6)
    p.add_argument("--n-typical", type=int, default=2)
    p.add_argument("--min-dice", type=float, default=0.8)
    p.add_argument("--min-area", type=float, default=0.05)
    a = ap.parse_args(argv)
    ov = _overrides(a.set)
    {"env": cmd_env, "prepare": cmd_prepare, "verify-duplicates": cmd_verify_duplicates, "queue": cmd_queue,
     "status": cmd_status, "train": cmd_train, "evaluate": cmd_evaluate, "analyze": cmd_analyze,
     "duplicate-masks": cmd_duplicate_masks, "figures": cmd_figures, "panel": cmd_panel}[a.cmd](a, ov)


if __name__ == "__main__":       # required on Windows (spawned DataLoader workers)
    main()
