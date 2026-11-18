"""Training sets ("domains") and the partitions they use.

=============  ==========================================  ==============================  ====================
train_domain   training images                             early stopping                  in-domain test
=============  ==========================================  ==============================  ====================
ham            HAM10000 train fold (~9,000)                HAM10000 val fold               ham10000_val
ham_matched    random subset of the HAM10000 train fold     HAM10000 val fold               ham10000_val
               with the SAME size as the isic2018 train
               partition (stratified by dx, whole lesions)
isic2018       ISIC 2018 Task 1 train+val pool minus        isic2018 val partition (10 %)   isic2018_test
               near-duplicates of the evaluation sets,
               90 % partition
=============  ==========================================  ==============================  ====================

Evaluation sets (every model): ``ham10000_val``, ``isic2018_test``, ``ph2`` (+ unlabelled ``bcn20000``).

Exclusion rule: a pool image is removed if it has any perceptual-hash candidate (Hamming <= 8) in
isic2018_test, ph2 or ham10000_val. Candidates are not reviewed, so the rule is conservative.
Pool partition: near-duplicate clusters inside the pool (same pHash rule) are kept in the same
partition; clusters are shuffled with seed 2026 and assigned to val until 10 % of the images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ("ham", "ham_matched", "isic2018")
LABELLED_SETS = ("ham10000_val", "isic2018_test", "ph2")
UNLABELLED_SETS = ("bcn20000",)
IN_DOMAIN = {"ham": "ham10000_val", "ham_matched": "ham10000_val", "isic2018": "isic2018_test"}
POOL = "isic2018_pool"
EXCLUDE_VS = ("isic2018_test", "ph2", "ham10000_val")
TRAIN_CACHE = {"ham": "ham10000", "ham_matched": "ham10000", "isic2018": POOL}


# ----------------------------------------------------------------------------- pool exclusions / split
def pool_exclusions(manifests: Path, write: bool = True) -> pd.DataFrame:
    rows = []
    for ref in EXCLUDE_VS:
        f = manifests / f"near_duplicates_{POOL}_vs_{ref}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run prepare first")
        df = pd.read_csv(f)
        rows += [{"image_id": q, "reason": f"pHash candidate vs {ref}"} for q in sorted(set(df["query_id"]))]
    ex = (pd.DataFrame(rows, columns=["image_id", "reason"]).groupby("image_id")["reason"]
          .agg("; ".join).reset_index())
    if write:
        ex.to_csv(manifests / f"{POOL}_excluded.csv", index=False)
    return ex


def _clusters(ids: list[str], pairs: pd.DataFrame) -> dict[str, int]:
    """Union-find over near-duplicate pairs -> {image_id: cluster id}."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs.get("query_id", []), pairs.get("ref_id", [])):
        if a in parent and b in parent and a != b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in ids}))}
    return {i: roots[find(i)] for i in ids}


def make_pool_split(pool_ids: list[str], excluded: set, self_pairs: pd.DataFrame, val_frac: float = 0.1,
                    seed: int = 2026) -> pd.DataFrame:
    keep = [i for i in sorted(pool_ids) if i not in excluded]
    cl = _clusters(keep, self_pairs)
    groups = pd.Series(cl).rename("group")
    order = np.random.default_rng(seed).permutation(sorted(groups.unique()))
    sizes = groups.value_counts()
    target, n_val, val_groups = val_frac * len(keep), 0, set()
    for g in order:
        if n_val >= target:
            break
        val_groups.add(g); n_val += int(sizes[g])
    df = pd.DataFrame({"image_id": keep, "group": [cl[i] for i in keep]})
    df["split"] = np.where(df["group"].isin(val_groups), "val", "train")
    assert not set(df.loc[df.split == "train", "group"]) & set(df.loc[df.split == "val", "group"])
    return df


# ----------------------------------------------------------------------------- size-matched HAM subset
def make_matched_subset(ham_split: pd.DataFrame, n: int, seed: int = 2026) -> pd.DataFrame:
    """``n`` images of the HAM10000 TRAIN fold, dx proportions preserved (largest-remainder
    allocation), whole lesions drawn in random order; the last lesion of a class may be cut to hit
    the quota exactly (no leakage: validation is the untouched HAM val fold)."""
    tr = ham_split[ham_split["split"] == "train"]
    if n > len(tr):
        raise ValueError(f"matched size {n} > HAM train fold {len(tr)}")
    frac = tr["dx"].value_counts(normalize=True).sort_index()
    raw = frac * n
    quota = np.floor(raw).astype(int)
    for dx in (raw - quota).sort_values(ascending=False).index[: n - int(quota.sum())]:
        quota[dx] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for dx, q in quota.items():
        sub = tr[tr["dx"] == dx]
        lesions = rng.permutation(sorted(sub["lesion_id"].unique()))
        picked = []
        for les in lesions:
            if len(picked) >= q:
                break
            picked += sorted(sub.loc[sub["lesion_id"] == les, "image_id"])
        chosen += picked[:q]
    out = tr[tr["image_id"].isin(chosen)].copy()
    assert len(out) == n, (len(out), n)
    out["split"] = "train"
    return out.sort_values("image_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- indices per domain
def _positions(meta_ids, wanted) -> np.ndarray:
    pos = {iid: i for i, iid in enumerate(meta_ids)}
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise KeyError(f"{len(missing)} ids not in cache, e.g. {missing[:3]}")
    return np.sort(np.array([pos[w] for w in wanted], dtype=np.int64))


def domain_indices(manifests: Path, caches: Path, domain: str) -> tuple[str, np.ndarray, np.ndarray]:
    """-> (training cache name, train row indices, validation row indices) for ``domain``."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown train_domain '{domain}' (expected one of {DOMAINS})")
    cache = TRAIN_CACHE[domain]
    meta = pd.read_csv(caches / cache / "meta.csv", keep_default_na=False)["image_id"].tolist()
    if domain in ("ham", "ham_matched"):
        split = pd.read_csv(manifests / "ham10000_split.csv")
        va = split.loc[split["split"] == "val", "image_id"].tolist()
        if domain == "ham":
            tr = split.loc[split["split"] == "train", "image_id"].tolist()
        else:
            tr = pd.read_csv(manifests / "ham10000_matched.csv")["image_id"].tolist()
    else:
        split = pd.read_csv(manifests / "isic2018_split.csv")
        tr = split.loc[split["split"] == "train", "image_id"].tolist()
        va = split.loc[split["split"] == "val", "image_id"].tolist()
    return cache, _positions(meta, tr), _positions(meta, va)
