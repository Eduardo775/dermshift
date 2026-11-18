"""Training loop shared by every model.

Run folder (``<project_root>/runs/<train_domain>/<experiment>/seed<k>/``)::

    config.yaml     resolved configuration            env.json   library and GPU versions
    train_log.csv   one row per epoch                 best.pt    weights at the best validation epoch
    last.pt         resumable state, written every ``run.checkpoint_minutes``
    DONE.json       written when training ends (early stopping or max epochs); last.pt is then deleted

Calling :func:`fit` on an existing folder resumes from ``last.pt``; a finished run is skipped.
``max_minutes`` stops cleanly after the current epoch.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_hash, save_config
from .losses import build_loss
from .metrics import batch_dice
from .models.registry import param_groups
from .utils.io import CSVLogger, atomic_save, load_checkpoint, write_json
from .utils.repro import environment, rng_state, seed_everything, set_rng_state, worker_init_fn


class EpochSampler(torch.utils.data.Sampler):
    """Yields (position, epoch). The order is a function of (seed, epoch) only -> exact resume, and the
    epoch reaches the dataset with every index, so the DataLoader workers can be persistent (on Windows
    workers are spawned: re-creating them every epoch costs 10-20 s per loader)."""

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True):
        self.n, self.seed, self.shuffle, self.epoch = int(n), int(seed), bool(shuffle), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed * 10_007 + self.epoch)
            order = torch.randperm(self.n, generator=g).tolist()
        else:
            order = range(self.n)
        e = self.epoch
        return iter([(int(k), e) for k in order])


def make_loader(ds, cfg, train: bool, seed: int = 0, batch_size: int | None = None) -> tuple[DataLoader, EpochSampler]:
    nw = int(cfg["data"]["num_workers"])
    sampler = EpochSampler(len(ds), seed, shuffle=train)
    dl = DataLoader(ds, batch_size=batch_size or cfg["optim"]["batch_size"], sampler=sampler, drop_last=train,
                    num_workers=nw, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
                    persistent_workers=nw > 0)
    return dl, sampler


def close_loader(dl) -> None:
    """Shut down persistent DataLoader workers now (on Windows they otherwise linger until garbage
    collection and, run after run, exhaust system resources: WinError 1450)."""
    it = getattr(dl, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()
    try:
        dl._iterator = None
    except Exception:            # noqa: BLE001
        pass


def _loader(ds, cfg, train: bool, epoch: int = 0, seed: int = 0, batch_size: int | None = None) -> DataLoader:
    """One-shot loader for a given epoch (benchmark, tests)."""
    dl, sampler = make_loader(ds, cfg, train, seed, batch_size)
    sampler.set_epoch(epoch)
    return dl


@torch.no_grad()
def validate_seg(model, loader, device, loss_fn, amp: bool) -> dict:
    model.eval()
    dices, losses = [], []
    for b in loader:
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x)
        losses.append(float(loss_fn(out, y)["loss"]) * len(x))
        dices += batch_dice(out["logits"], y)
    return {"val_dice": float(np.mean(dices)), "val_loss": float(np.sum(losses) / len(dices))}


def fit(cfg: dict, model: torch.nn.Module, train_ds, val_ds, run_dir: str | Path,
        max_minutes: float | None = None, device: str | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE.json").exists():
        print(f"[fit] {run_dir} already finished; skipping")
        return {"status": "done", "run_dir": str(run_dir)}

    seed = int(cfg["run"]["seed"])
    seed_everything(seed, cfg["run"]["deterministic"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    cl = bool(cfg["run"].get("channels_last", False)) and device.type == "cuda"
    model.to(device)
    if cl:
        model.to(memory_format=torch.channels_last)
    loss_fn = build_loss(cfg)
    o = cfg["optim"]
    groups = param_groups(model, cfg)
    opt = torch.optim.Adam(groups, weight_decay=o["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["t_max"], eta_min=o["eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    key = "val_dice"

    start_epoch, best, best_epoch, bad_epochs, gpu_seconds = 0, -math.inf, -1, 0, 0.0
    h = config_hash(cfg)
    if (run_dir / "last.pt").exists():
        ck = load_checkpoint(run_dir / "last.pt")
        if ck["config_hash"] != h:
            raise RuntimeError(f"{run_dir}: config changed since the checkpoint was written "
                               f"({ck['config_hash']} != {h}). Use a new experiment name.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        start_epoch, best, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
        bad_epochs, gpu_seconds = ck["bad_epochs"], ck["train_seconds"]
        print(f"[fit] resumed {run_dir.name} at epoch {start_epoch} (best {key}={best:.4f} @ {best_epoch})")
    else:
        save_config(cfg, run_dir / "config.yaml")
        write_json({**environment(), "config_hash": h,
                    "param_groups": {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}},
                   run_dir / "env.json")
    log = CSVLogger(run_dir / "train_log.csv", resume_from_epoch=start_epoch)
    train_dl, train_sampler = make_loader(train_ds, cfg, True, seed)
    val_dl, _ = make_loader(val_ds, cfg, False)
    session_t0 = time.time()
    status = "running"
    epoch = start_epoch - 1
    ckpt_minutes = float(cfg["run"].get("checkpoint_minutes", 0))   # 0 -> every epoch
    last_save = time.time()

    for epoch in range(start_epoch, o["max_epochs"]):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        sums, n, gnorms, n_clipped, gate_m = {}, 0, [], 0, []
        for b in train_dl:
            x = b["image"].to(device, non_blocking=True)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            y = b["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            terms = loss_fn(out, y)                      # fp32, outside autocast
            if not torch.isfinite(terms["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(terms["loss"]).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], o["grad_clip"]))
            gnorms.append(gn); n_clipped += int(gn > o["grad_clip"])
            scaler.step(opt); scaler.update()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(x)
            n += len(x)
            if out.get("gate") is not None:
                gate_m.append(float(out["gate"].detach().float().mean()))
        sched.step()
        train_time = time.time() - t0
        gpu_seconds += train_time
        val = validate_seg(model, val_dl, device, loss_fn, amp)
        row = {"epoch": epoch, **{f"train_{k}": v / max(n, 1) for k, v in sums.items()}, **val,
               "lr_" + groups[0]["name"]: opt.param_groups[0]["lr"],
               "grad_norm_mean": float(np.mean(gnorms)) if gnorms else float("nan"),
               "grad_norm_max": float(np.max(gnorms)) if gnorms else float("nan"),
               "frac_clipped": n_clipped / max(len(gnorms), 1),
               "gate_mean": float(np.mean(gate_m)) if gate_m else "",
               "epoch_seconds": round(time.time() - t0, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else ""}
        improved = val[key] > best
        if improved:
            best, best_epoch, bad_epochs = val[key], epoch, 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, key: best, "config_hash": h}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        row["best_so_far"] = best
        log.log(row)
        print(f"[{run_dir.parent.name}/{run_dir.name}] ep {epoch:3d} loss {row.get('train_loss', float('nan')):.4f} "
              f"{key} {val[key]:.4f} best {best:.4f}@{best_epoch} ({row['epoch_seconds']}s)")
        stop = bad_epochs >= o["patience"]
        pause = max_minutes is not None and (time.time() - session_t0) / 60 > max_minutes
        if stop or pause or epoch == o["max_epochs"] - 1 or (time.time() - last_save) / 60 >= ckpt_minutes:
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                         "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch, "best": best,
                         "best_epoch": best_epoch, "bad_epochs": bad_epochs, "train_seconds": gpu_seconds,
                         "config_hash": h}, run_dir / "last.pt")
            last_save = time.time()
        if stop:
            status = "early_stopped"
            break
        if pause:
            print("[fit] time budget reached; state saved, re-run to resume")
            close_loader(train_dl); close_loader(val_dl)
            return {"status": "paused", "epoch": epoch, "run_dir": str(run_dir)}
    else:
        status = "max_epochs"
    summary = {"status": status, "best_epoch": best_epoch, key: best, "last_epoch": epoch,
               "train_hours": gpu_seconds / 3600, "config_hash": h}
    write_json(summary, run_dir / "DONE.json")
    (run_dir / "last.pt").unlink(missing_ok=True)
    close_loader(train_dl); close_loader(val_dl)
    return summary
