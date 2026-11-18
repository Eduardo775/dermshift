"""Near-duplicate screening: 64-bit perceptual hashes, candidate pairs within a Hamming distance, normalised
cross-correlation of candidates and contact sheets for visual review.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def phash_array(images: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """images: uint8 (N,H,W,3) -> uint64 (N,) perceptual hashes."""
    import imagehash

    out = np.empty(len(images), dtype=np.uint64)
    for i, im in enumerate(images):
        h = imagehash.phash(Image.fromarray(im), hash_size=hash_size)
        out[i] = np.uint64(int(str(h), 16))
    return out


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):                       # numpy >= 2.0: hardware popcount
        return np.bitwise_count(x.astype(np.uint64)).astype(np.uint8)
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.uint8)
        x >>= np.uint64(1)
    return c


def near_duplicates(h_query: np.ndarray, ids_query, h_ref: np.ndarray, ids_ref, max_dist: int = 8,
                    chunk: int = 256) -> pd.DataFrame:
    rows = []
    for s in range(0, len(h_query), chunk):
        q = h_query[s:s + chunk, None]
        d = _popcount64(np.bitwise_xor(q, h_ref[None, :]))
        qi, ri = np.nonzero(d <= max_dist)
        for a, b in zip(qi, ri):
            rows.append(dict(query_id=ids_query[s + a], ref_id=ids_ref[b], hamming=int(d[a, b])))
    return pd.DataFrame(rows, columns=["query_id", "ref_id", "hamming"]).sort_values("hamming")


def self_pairs(h: np.ndarray, ids, max_dist: int = 8) -> pd.DataFrame:
    """Near-duplicate pairs inside one set (each unordered pair once, no self-pairs)."""
    df = near_duplicates(h, ids, h, ids, max_dist)
    ids = list(ids)
    pos = {i: k for k, i in enumerate(ids)}
    return df[[pos[a] < pos[b] for a, b in zip(df["query_id"], df["ref_id"])]].reset_index(drop=True)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equally-sized RGB images (grey level)."""
    ga = a.astype(np.float32).mean(-1).ravel(); gb = b.astype(np.float32).mean(-1).ravel()
    ga -= ga.mean(); gb -= gb.mean()
    den = float(np.linalg.norm(ga) * np.linalg.norm(gb))
    return float(ga @ gb / den) if den > 0 else 0.0


def _square(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Central square crop resized to size x size: aligns a 1:1 export (BCN20000, 1024x1024) with the
    4:3 original of the same photograph, which the 192x256 caches would otherwise distort differently."""
    import cv2

    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def best_ncc(a: np.ndarray, b: np.ndarray, square: bool = False) -> float:
    """Max NCC over identity / horizontal / vertical flip; with ``square`` also on central square crops."""
    v = max(ncc(a, b), ncc(a, b[:, ::-1]), ncc(a, b[::-1]))
    if square:
        sa, sb = _square(a), _square(b)
        v = max(v, ncc(sa, sb), ncc(sa, sb[:, ::-1]), ncc(sa, sb[::-1]))
    return v


def add_ncc(pairs: pd.DataFrame, cache_q, cache_r, square: bool = False) -> pd.DataFrame:
    """Add an ``ncc`` column (max over flips; optionally also on central square crops) to a candidate
    list, using the 192x256 cached images of the query and reference sets."""
    if not len(pairs):
        return pairs.assign(ncc=pd.Series(dtype=float))
    pq = {i: k for k, i in enumerate(cache_q.meta["image_id"])}
    pr = {i: k for k, i in enumerate(cache_r.meta["image_id"])}
    vals = [best_ncc(cache_q.image(pq[a]), cache_r.image(pr[b]), square)
            for a, b in zip(pairs["query_id"], pairs["ref_id"])]
    return pairs.assign(ncc=vals).sort_values("ncc", ascending=False).reset_index(drop=True)


def review_sheets(pairs: pd.DataFrame, cache_q, cache_r, out_dir: Path, per_sheet: int = 12, max_pairs: int = 600) -> list:
    """PNG contact sheets (query | reference, with Hamming and NCC) for visual review."""
    import cv2

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pq = {i: k for k, i in enumerate(cache_q.meta["image_id"])}
    pr = {i: k for k, i in enumerate(cache_r.meta["image_id"])}
    files, rows = [], pairs.head(max_pairs).reset_index(drop=True)
    for s in range(0, len(rows), per_sheet):
        tiles = []
        for r in rows.iloc[s:s + per_sheet].itertuples(index=False):
            a, b = cache_q.image(pq[r.query_id]), cache_r.image(pr[r.ref_id])
            t = np.concatenate([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b], 1)
            bar = np.full((22, t.shape[1], 3), 255, np.uint8)
            txt = f"{r.query_id} | {r.ref_id}  ham={r.hamming}" + (f" ncc={r.ncc:.3f}" if hasattr(r, "ncc") else "")
            cv2.putText(bar, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([bar, t], 0))
        f = out_dir / f"sheet_{s // per_sheet:03d}.png"
        sheet = np.concatenate(tiles, 0)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        buf.tofile(str(f))                                  # unicode-safe on Windows
        files.append(f)
    return files
