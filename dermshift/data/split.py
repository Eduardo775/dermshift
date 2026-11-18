"""Lesion-level HAM10000 split (stratified group k-fold), so images of one lesion never cross partitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_lesion_split(df: pd.DataFrame, n_folds: int = 10, val_fold: int = 0, seed: int = 2026) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column in {train, val}.

    StratifiedGroupKFold keeps every image of a ``lesion_id`` in the same subset and
    approximately preserves the ``dx`` distribution. With 10 folds the ratio is ~90/10.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(df)), y=df["dx"].values, groups=df["lesion_id"].values))
    _, val_idx = folds[val_fold]
    out = df.copy()
    out["split"] = "train"
    out.iloc[val_idx, out.columns.get_loc("split")] = "val"
    check_no_group_leak(out)
    return out


def check_no_group_leak(df: pd.DataFrame) -> None:
    tr = set(df.loc[df["split"] == "train", "lesion_id"])
    va = set(df.loc[df["split"] == "val", "lesion_id"])
    inter = tr & va
    if inter:
        raise AssertionError(f"{len(inter)} lesion_id shared by train and val")


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per split and dx (goes to the paper's data-split table)."""
    g = df.groupby(["split", "dx"]).size().unstack(fill_value=0)
    g["images"] = g.sum(axis=1)
    g["lesions"] = df.groupby("split")["lesion_id"].nunique()
    return g
