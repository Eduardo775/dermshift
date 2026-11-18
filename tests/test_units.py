"""Unit tests: metrics vs medpy, losses vs reference implementations, model plumbing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _ellipses(rng, H, W):
    yy, xx = np.mgrid[:H, :W]
    a = lambda: ((yy - H / 2 - rng.normal(0, 8)) ** 2 / rng.uniform(5, H / 2) ** 2
                 + (xx - W / 2 - rng.normal(0, 8)) ** 2 / rng.uniform(5, W / 2) ** 2) < 1
    return a(), a()


def test_metrics_match_medpy():
    mb = pytest.importorskip("medpy.metric.binary")
    from dermshift.metrics import image_metrics

    rng = np.random.default_rng(1)
    for t in range(40):
        H, W = rng.integers(40, 300, 2)
        p, g = _ellipses(rng, H, W)
        if t % 3 == 0:
            p |= rng.random((H, W)) > 0.995
        if not p.any() or not g.any():
            continue
        m = image_metrics(p, g)
        ref = dict(hd=mb.hd(p, g), hd95=mb.hd95(p, g), assd=mb.assd(p, g), dice=mb.dc(p, g), iou=mb.jc(p, g),
                   sensitivity=mb.sensitivity(p, g), specificity=mb.specificity(p, g))
        for k, v in ref.items():
            assert m[k] == pytest.approx(v, abs=1e-9), k


def test_crop_is_exact():
    from dermshift.evaluate import _metrics_job
    from dermshift.metrics import image_metrics

    rng = np.random.default_rng(2)
    p, g = _ellipses(rng, 400, 600)
    a = _metrics_job(("x", p, g))
    b = image_metrics(p, g)
    for k in ("hd", "hd95", "assd", "dice"):
        assert a[k] == pytest.approx(b[k], abs=1e-9)


def test_empty_policy():
    from dermshift.evaluate import _metrics_job

    g = np.zeros((30, 40), bool); g[5:10, 5:10] = True
    e = np.zeros_like(g)
    r = _metrics_job(("x", e, g))
    assert r["dice"] == 0 and r["hd95"] == pytest.approx(50.0) and r["pred_empty"]
    r = _metrics_job(("x", e, e))
    assert r["dice"] == 1 and r["hd95"] == 0


def test_rank_loss_reference():
    """Vectorised per-image rank loss vs an explicit double loop (no subsampling)."""
    from dermshift.losses import topk_rank_loss

    torch.manual_seed(0)
    p = torch.rand(2, 200); y = (torch.rand(2, 200) > 0.7).float()
    ours = topk_rank_loss(p, y, fraction=0.05, margin=0.3, max_pairs=10_000)
    for b in range(2):
        pos = sorted(p[b][y[b] > 0.5].tolist())[:10]
        neg = sorted(p[b][y[b] <= 0.5].tolist(), reverse=True)[:10]
        ref = np.mean([max(0, n - q + 0.3) for q in pos for n in neg])
        assert float(ours[b]) == pytest.approx(ref, rel=1e-6)


def test_param_groups_cover_everything():
    from dermshift.config import load_config
    from dermshift.models.registry import build_model, param_groups

    cfg = load_config(ROOT / "configs" / "models" / "cascade.yaml", {"model.pretrained": False})
    cfg["model"]["pretrained"] = True        # grouping rule only; weights stay random
    m = build_model({**cfg, "model": {**cfg["model"], "pretrained": False}})
    groups = param_groups(m, cfg)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for _ in m.parameters())
    names = {g["name"]: len(g["params"]) for g in groups}
    assert names["fusion"] == 3   # W1, W2.weight, W2.bias


def _rank_loop_reference(p, y, fraction, margin):
    """Original per-image implementation (no subsampling) kept as the reference."""
    import torch.nn.functional as F
    B, N = p.shape
    k_base = max(1, int(fraction * N))
    out = p.new_zeros(B)
    for b in range(B):
        pos, neg = p[b][y[b] > 0.5], p[b][y[b] <= 0.5]
        k = min(k_base, pos.numel(), neg.numel())
        if k == 0:
            continue
        hp = torch.topk(pos, k, largest=False).values
        hn = torch.topk(neg, k, largest=True).values
        out[b] = F.relu(hn[None, :] - hp[:, None] + margin).mean()
    return out


def test_vectorised_rank_loss_equals_loop():
    from dermshift.losses import topk_rank_loss

    torch.manual_seed(1)
    p = torch.rand(6, 3000, requires_grad=True)
    y = (torch.rand(6, 3000) > 0.8).float()
    y[1] = 0; y[2] = 1; y[3, :5] = 1; y[3, 5:] = 0          # no positives, no negatives, tiny lesion
    a = topk_rank_loss(p, y, 0.1, 0.3, max_pairs=10_000)
    b = _rank_loop_reference(p, y, 0.1, 0.3)
    assert torch.allclose(a, b, atol=1e-6)
    ga, = torch.autograd.grad(a.sum(), p)
    gb, = torch.autograd.grad(b.sum(), p)
    assert torch.allclose(ga, gb, atol=1e-6)


def test_rank_loss_subsampling_is_unbiased():
    from dermshift.losses import topk_rank_loss

    torch.manual_seed(0)
    p = torch.rand(2, 4000); y = (torch.rand(2, 4000) > 0.5).float()
    full = _rank_loop_reference(p, y, 0.1, 0.3)                      # K = 400 per side
    est = torch.stack([topk_rank_loss(p, y, 0.1, 0.3, max_pairs=100) for _ in range(300)]).mean(0)
    assert torch.allclose(est, full, rtol=0.03)


# ----------------------------------------------------------------------------- study-specific metrics and analyses
def test_nsd_and_area_ratio():
    from dermshift.evaluate import _metrics_job
    from dermshift.metrics import log_area_ratio, surface_metrics

    g = np.zeros((200, 300), bool); g[50:150, 80:200] = True
    r = surface_metrics(g, g, (2,), (1,))
    assert r["nsd_2px"] == 1 and r["nsd_1pct"] == 1 and r["hd95"] == 0
    p = np.zeros_like(g); p[54:154, 84:204] = True          # shifted by 4 px
    vals = [surface_metrics(p, g, (t,), ())[f"nsd_{t}px"] for t in (1, 2, 4, 6)]
    assert vals[0] < vals[1] < vals[2] <= vals[3] and vals[3] == pytest.approx(1.0)
    big = np.zeros_like(g); big[40:160, 70:210] = True
    assert log_area_ratio(big, g) > 0 > log_area_ratio(g[:, ::-1] & big & (np.arange(300) < 150), g)
    e = np.zeros_like(g)
    j = _metrics_job(("x", e, g, (2,), (1,)))
    assert j["nsd_2px"] == 0 and j["nsd_1pct"] == 0 and j["dice"] == 0
    j = _metrics_job(("x", p, g, (2, 4), (0.5, 1, 2)))     # crop must not change the tolerances
    assert j["nsd_4px"] == pytest.approx(surface_metrics(p, g, (4,), ())["nsd_4px"])
    assert j["nsd_2pct"] == pytest.approx(surface_metrics(p, g, (), (2,))["nsd_2pct"])


def test_unlabelled_stats():
    from dermshift.metrics import unlabelled_stats

    pr = np.zeros((96, 128), np.float32); pr[30:60, 40:90] = 0.9
    s = unlabelled_stats(pr)
    assert s["n_components"] == 1 and not s["pred_empty"] and 0 < s["pred_area_frac"] < 1 and s["pred_border_frac"] == 0
    assert unlabelled_stats(np.zeros_like(pr))["pred_empty"]


def test_interventions():
    from dermshift.interventions import get_transform, hair_mask, remove_hair, shades_of_gray

    rng = np.random.default_rng(0)
    img = np.clip(rng.normal([200, 140, 120], 10, (96, 128, 3)), 0, 255).astype(np.uint8)   # reddish cast
    sg = shades_of_gray(img)
    assert sg.dtype == np.uint8 and sg.shape == img.shape
    assert np.ptp(sg.reshape(-1, 3).mean(0)) < np.ptp(img.reshape(-1, 3).mean(0)) / 3
    hairy = img.copy(); hairy[:, 60:62] = 20                                                   # a dark hair
    assert hair_mask(hairy).sum() > hair_mask(img).sum()
    clean = remove_hair(hairy)
    assert clean[:, 60:62].mean() > 100
    ref = {"mean": [50.0, 10.0, 10.0], "std": [10.0, 5.0, 5.0]}
    out = get_transform("reinhard", ref)(img)
    import cv2
    lab = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    assert abs(lab[:, 0].mean() - 50) < 3
    with pytest.raises(ValueError):
        get_transform("reinhard")
    import pickle
    pickle.dumps(get_transform("reinhard", ref)); pickle.dumps(get_transform("sog"))           # spawn-safe


def test_matched_subset_and_pool_split():
    import pandas as pd

    from dermshift.domains import make_matched_subset, make_pool_split

    rows = [{"image_id": f"I{i:03d}", "lesion_id": f"L{i // 3}", "dx": ["nv", "mel", "bkl"][i % 3 if i < 60 else 0],
             "split": "val" if i >= 90 else "train"} for i in range(100)]
    df = pd.DataFrame(rows)
    m = make_matched_subset(df, 30)
    assert len(m) == 30 and set(m.image_id) <= set(df[df.split == "train"].image_id)
    m2 = make_matched_subset(df, 30)
    assert m.equals(m2)                                                # deterministic
    pairs = pd.DataFrame({"query_id": ["A1", "A3"], "ref_id": ["A2", "A4"], "hamming": [0, 1]})
    ids = [f"A{i}" for i in range(1, 41)]
    sp = make_pool_split(ids, {"A40"}, pairs)
    assert "A40" not in set(sp.image_id) and abs((sp.split == "val").mean() - 0.1) < 0.06
    g = sp.set_index("image_id")
    assert g.loc["A1", "split"] == g.loc["A2", "split"] and g.loc["A3", "split"] == g.loc["A4", "split"]


def test_config_hash_ignores_paths_and_evaluation_keys():
    from dermshift.config import config_hash, load_config

    a = load_config(ROOT / "configs" / "models" / "cascade.yaml", use_local=False)
    b = load_config(ROOT / "configs" / "models" / "cascade.yaml", use_local=False)
    b["paths"] = {"datasets_root": "/data", "project_root": "/proj", "local_root": None}
    b["eval"]["sets"] = ["ph2"]
    b["data"]["num_workers"] = 0
    assert config_hash(a) == config_hash(b)
    c = load_config(ROOT / "configs" / "models" / "cascade.yaml", {"data.train_domain": "isic2018"}, use_local=False)
    assert config_hash(c) != config_hash(a)


def test_cascade_forward_and_gate_gradient():
    from dermshift.models.cascade import Cascade

    m = Cascade(pretrained=False)
    out = m(torch.randn(2, 3, 96, 128))
    assert out["logits"].shape == out["coarse_logits"].shape == (2, 1, 96, 128) and out["gate"].shape == (2, 1024)
    out["logits"].sum().backward()
    assert m.fusion.W1.weight.grad.abs().sum() > 0


def test_all_model_configs_build():
    from dermshift.config import load_config
    from dermshift.models.registry import MODELS, build_model

    for name in MODELS:
        cfg = load_config(ROOT / "configs" / "models" / f"{name}.yaml",
                          {"model.pretrained": False, "data.height": 96, "data.width": 128}, use_local=False)
        assert cfg["experiment"] == name
        out = build_model(cfg).eval()(torch.randn(1, 3, 96, 128))
        assert out["logits"].shape == (1, 1, 96, 128)


def test_packed_masks_give_identical_metrics():
    from dermshift.evaluate import _metrics_job, pack_mask

    rng = np.random.default_rng(1)
    g = np.zeros((301, 457), bool); g[60:200, 100:330] = True
    p = np.roll(g, 7, axis=1) & (rng.random(g.shape) > 0.02)
    a = _metrics_job(("x", p, g, (2, 5), (1,)))
    b = _metrics_job(("x", pack_mask(p), pack_mask(g), (2, 5), (1,)))
    assert a == b


def test_analysis_tolerates_missing_evaluations(tmp_path):
    """Without intervention or BCN evaluations the analyses return empty tables."""
    from dermshift.analysis import bcn_risk, collect, indicators, intervention_tests, seed_avg

    assert len(seed_avg(collect(tmp_path, ["ham"], ["unet"], [42], ["ph2"]))) == 0
    assert len(intervention_tests(tmp_path, ["ham"], ["unet"], [42], ["ph2"], ["sog"])) == 0
    assert len(indicators(tmp_path, "ham", ["unet"], [42], "bcn20000")) == 0
    assert len(bcn_risk(tmp_path, tmp_path, "ham", ["unet"], [42], {})) == 0


def test_duplicate_mask_comparison():
    """S5 helpers: flip recovery, signed offset sign, paired comparison on a mirrored copy."""
    import cv2
    import pandas as pd

    from dermshift.annotation_pairs import best_flip, compare_pairs, signed_offset, summarize

    rng = np.random.default_rng(0)
    img = (rng.random((64, 80, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    big = np.zeros((64, 80), np.uint8); cv2.circle(big, (40, 32), 16, 1, -1)
    small = np.zeros((64, 80), np.uint8); cv2.circle(small, (40, 32), 12, 1, -1)
    assert best_flip(img, img[:, ::-1])[0] == "h"
    assert signed_offset(big, small) < 0 < signed_offset(small, big)

    class Fake:
        def __init__(self, ids, imgs, masks):
            self.meta = pd.DataFrame({"image_id": ids}); self._i, self._m = imgs, masks
        def image(self, k): return self._i[k]
        def mask(self, k): return self._m[k]

    tr = Fake(["a"], [img], [big])
    te = Fake(["b"], [img[:, ::-1].copy()], [small[:, ::-1].copy()])     # mirrored copy with a tighter mask
    d = compare_pairs(pd.DataFrame({"query_id": ["a"], "ref_id": ["b"], "ncc": [1.0]}), tr, te)
    assert d.loc[0, "flip"] == "h" and d.loc[0, "log_area_ratio_test_vs_train"] < 0
    assert d.loc[0, "offset_test_vs_train_px"] < 0
    s = summarize(pd.concat([d] * 4, ignore_index=True))
    assert len(s) and "area_ratio_median" in s.columns


def test_duplicate_registration_and_review(tmp_path):
    """S5: a re-exported copy (5% zoom, brighter, lower contrast) with the SAME mask must give area ratio ~ 1
    after registration; the Excel review sheet may use ';' and a BOM."""
    import cv2
    import pandas as pd

    from dermshift.annotation_pairs import compare_pairs, read_review

    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((192, 256, 3)) * 255).astype(np.uint8), (9, 9), 0)
    m = np.zeros((192, 256), np.uint8); cv2.ellipse(m, (128, 96), (50, 35), 20, 0, 360, 1, -1)
    base[m > 0] = (base[m > 0] * 0.5).astype(np.uint8)
    A = cv2.getRotationMatrix2D((128, 96), 0, 1.05)                    # zoom in 5 %
    img2 = cv2.warpAffine(base, A, (256, 192), borderMode=cv2.BORDER_REFLECT)
    img2 = np.clip(img2.astype(np.float32) * 0.8 + 30, 0, 255).astype(np.uint8)
    m2 = cv2.warpAffine(m, A, (256, 192), flags=cv2.INTER_NEAREST)

    class Fake:
        def __init__(self, ids, imgs, masks):
            self.meta = pd.DataFrame({"image_id": ids}); self._i, self._m = imgs, masks
        def image(self, k): return self._i[k]
        def mask(self, k): return self._m[k]

    d = compare_pairs(pd.DataFrame({"query_id": ["a"], "ref_id": ["b"], "ncc": [0.95]}),
                      Fake(["a"], [base], [m]), Fake(["b"], [img2], [m2]))
    r = d.iloc[0]
    assert r.registered and abs(r.scale_test_to_train - 1.05) < 0.02
    assert abs(r.log_area_ratio_test_vs_train) < 0.03 and r.dice_train_test > 0.95

    f = tmp_path / "isic2018_train_test_review.csv"
    f.write_text("query_id;ref_id;hamming;ncc;Decision\nISIC_1;ISIC_2;3;0,95;same\n", encoding="utf-8-sig")
    rv = read_review(f)
    assert list(rv.columns)[-1] == "decision" and rv.loc[0, "decision"] == "same"
