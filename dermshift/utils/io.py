"""Atomic checkpoint writing (temporary file + rename) and CSV logging.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import torch


def atomic_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class CSVLogger:
    """Append-only CSV; on resume, rows with epoch >= resume epoch are dropped first so
    the log never contains duplicated epochs."""

    def __init__(self, path: Path, resume_from_epoch: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = None
        if self.path.exists():
            with open(self.path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                self.fields = list(rows[0].keys())
            if resume_from_epoch is not None:
                rows = [r for r in rows if int(r["epoch"]) < resume_from_epoch]
                with open(self.path, "w", newline="") as f:
                    if self.fields:
                        w = csv.DictWriter(f, self.fields)
                        w.writeheader()
                        w.writerows(rows)

    def log(self, row: dict) -> None:
        new = not self.path.exists() or self.path.stat().st_size == 0
        if self.fields is None:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
