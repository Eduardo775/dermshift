"""Resumable queue that runs the whole study: 3 training sets x 7 models x 3 seeds, then inference-only items.

Item kinds: ``train`` (train if needed, evaluate on ``eval.sets``), ``extra`` (ISIC 2018 validation partition),
``bcn`` (BCN20000 predictions) and ``tf:<name>`` (test-time transform). Finished items are skipped and an
interrupted run resumes from ``last.pt``, so the queue can be stopped and restarted at any time.
"""
from __future__ import annotations

import time

import pandas as pd

from .config import load_config
from .evaluate import eval_is_current
from .models.registry import MODELS
from .paths import from_config
from .pipeline import MODEL_DIR, evaluate_existing, predict_unlabelled, run_experiment, run_is_complete

DOMAINS = ("ham", "ham_matched", "isic2018")
SEEDS = (42, 43, 44)
TRANSFORMS = ("sog", "reinhard", "hairless")


def plan(seeds=SEEDS) -> list[tuple[str, str, str, int]]:
    """(kind, training set, model, seed) in execution order."""
    return ([("train", d, e, s) for s in seeds for d in DOMAINS for e in MODELS]
            + [("extra", d, e, s) for s in seeds for d in DOMAINS for e in MODELS]
            + [("bcn", d, e, s) for s in seeds for d in DOMAINS for e in MODELS]
            + [(f"tf:{t}", d, e, s) for t in TRANSFORMS for s in seeds for d in DOMAINS for e in MODELS])


def _cfg(overrides: dict, exp: str, seed: int, domain: str) -> dict:
    return load_config(MODEL_DIR / f"{exp}.yaml", {**overrides, "run.seed": seed, "data.train_domain": domain})


def _done(P, cfg, kind: str, domain: str, exp: str, seed: int) -> bool:
    rd = P.run_dir(domain, exp, seed)
    if kind == "train":
        return run_is_complete(P, cfg, domain, exp, seed)
    if kind == "extra":
        return eval_is_current(rd / "eval" / "isic2018_val", cfg)
    if kind == "bcn":
        return eval_is_current(rd / "eval" / "bcn20000", cfg)
    t = kind.split(":", 1)[1]
    return all(eval_is_current(rd / "eval" / f"{s}__{t}", cfg) for s in cfg["eval"]["sets"])


def status(overrides: dict, seeds=SEEDS) -> pd.DataFrame:
    rows = []
    for kind, domain, exp, seed in plan(seeds):
        cfg = _cfg(overrides, exp, seed, domain)
        P = from_config(cfg)
        started = kind == "train" and (P.run_dir(domain, exp, seed) / "last.pt").exists()
        state = "done" if _done(P, cfg, kind, domain, exp, seed) else "in progress" if started else "pending"
        rows.append({"kind": kind.split(":")[0], "item": kind, "domain": domain, "model": exp, "seed": seed, "state": state})
    return pd.DataFrame(rows)


def run_queue(overrides: dict, max_minutes: float | None = None, kinds: tuple[str, ...] | None = None,
              seeds=SEEDS) -> pd.DataFrame:
    """Run pending items in order until the end of the plan or the optional time budget."""
    from .pipeline import build_isic_val_cache

    t0 = time.time()
    left = (lambda: None) if max_minutes is None else (lambda: max_minutes - (time.time() - t0) / 60)
    for kind, domain, exp, seed in plan(seeds):
        if kinds and kind.split(":")[0] not in kinds:
            continue
        cfg = _cfg(overrides, exp, seed, domain)
        P = from_config(cfg)
        if _done(P, cfg, kind, domain, exp, seed):
            continue
        if left() is not None and left() <= 5:
            print("[queue] time budget used")
            break
        trained = (P.run_dir(domain, exp, seed) / "best.pt").exists()
        if kind != "train" and not trained:
            continue
        print(f"[queue] {kind} {domain}/{exp} seed {seed}", flush=True)
        ov = {**overrides, "data.train_domain": domain}
        if kind == "train":
            if run_experiment(MODEL_DIR / f"{exp}.yaml", seed, domain, left(), root_overrides=overrides)["status"] == "paused":
                break
        elif kind == "extra":
            if not (P.caches_local / "isic2018_val" / "_COMPLETE").exists():
                build_isic_val_cache(cfg)
            evaluate_existing(MODEL_DIR / f"{exp}.yaml", seed, domain, ["isic2018_val"], root_overrides=ov)
        elif kind == "bcn":
            if not (P.caches_local / "bcn20000" / "_COMPLETE").exists():
                print("[queue] BCN20000 cache not found: download BCN20000 and run prepare")
                continue
            predict_unlabelled(MODEL_DIR / f"{exp}.yaml", seed, domain, root_overrides=ov)
        else:
            evaluate_existing(MODEL_DIR / f"{exp}.yaml", seed, domain, transform=kind[3:], root_overrides=ov)
    return status(overrides, seeds)
