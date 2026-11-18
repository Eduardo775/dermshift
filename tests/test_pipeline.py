"""End-to-end tests on synthetic data (CPU): ``python -m pytest -q tests`` from the repository root.

The synthetic datasets are stored under a folder name with non-ASCII characters to exercise path handling.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "configs" / "models"


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    from tests.make_synthetic import make

    base = tmp_path_factory.mktemp("e2e")
    make(base / "Datos_Investigación")
    ov = {"paths.datasets_root": str(base / "Datos_Investigación"), "paths.project_root": str(base / "DermShift"),
          "data.height": 96, "data.width": 128, "data.num_workers": 0, "data.n_folds": 5,
          "optim.batch_size": 4, "optim.max_epochs": 1, "optim.t_max": 1, "optim.amp": False,
          "model.pretrained": False, "run.deterministic": True}
    from dermshift.config import load_config
    from dermshift.pipeline import prepare_data

    cfg = load_config(ROOT / "configs" / "base.yaml", ov)
    rep = prepare_data(cfg, threads=2, published=None)
    return {"base": base, "ov": ov, "cfg": cfg, "report": rep, "proj": base / "DermShift"}


def test_data_preparation(env):
    rep, proj = env["report"], env["proj"]
    assert rep["ham10000"] == 40 and rep["isic2018_test"] == 6 and rep["ph2"] == 4 and rep["isic2018_pool"] == 20
    assert rep["bcn20000"] == 10                       # unlabelled rows and the clinical image are dropped
    man = proj / "manifests"
    # caches are read in place and the validation fold keeps its original-resolution GT
    from dermshift.data.cache import CacheArrays
    val = CacheArrays(proj / "cache" / "ham10000_val")
    split = pd.read_csv(man / "ham10000_split.csv")
    assert sorted(val.meta["image_id"]) == sorted(split.loc[split.split == "val", "image_id"])
    assert val.original_gt(val.meta["image_id"][0]).shape == (150, 200)
    assert not CacheArrays(proj / "cache" / "bcn20000").has_masks
    bm = pd.read_csv(man / "bcn20000.csv", keep_default_na=False)
    assert (bm["anatom_site"] != "").all() and bm["image_id"].str.startswith("ISIC_").all()
    # the planted train/test duplicate is excluded; the in-pool duplicate stays in one partition
    ex = pd.read_csv(man / "isic2018_pool_excluded.csv")
    assert "ISIC_0000000" in set(ex["image_id"])
    ps = pd.read_csv(man / "isic2018_split.csv").set_index("image_id")
    assert "ISIC_0000000" not in ps.index
    assert ps.loc["ISIC_0000001", "split"] == ps.loc["ISIC_0000002", "split"]
    assert ps.loc["ISIC_0000001", "group"] == ps.loc["ISIC_0000002", "group"]
    # size-matched HAM subset: same size as the ISIC train partition, only HAM train-fold images
    mt = pd.read_csv(man / "ham10000_matched.csv")
    assert len(mt) == (ps["split"] == "train").sum() == rep["isic2018_partition"]["train"]
    assert set(mt["image_id"]) <= set(split.loc[split.split == "train", "image_id"])
    from dermshift.domains import domain_indices
    for d in ("ham", "ham_matched", "isic2018"):
        cache, tr, va = domain_indices(man, proj / "cache", d)
        assert len(set(tr) & set(va)) == 0 and len(tr) and len(va)


def test_existing_split_is_kept(env, tmp_path):
    """A split file already in manifests/ (e.g. installed from data/) is kept, not regenerated."""
    from dermshift.config import load_config
    from dermshift.pipeline import prepare_data

    man = env["proj"] / "manifests"
    s = pd.read_csv(man / "ham10000_split.csv")
    rep = json.loads((man / "data_report.json").read_text())
    assert rep["ham_split"]["source"].startswith("regenerated")
    proj2 = tmp_path / "proj2"
    (proj2 / "manifests").mkdir(parents=True)
    flipped = s.copy()
    i = flipped.index[flipped.split == "train"][0]
    flipped.loc[i, "split"] = "val"                    # a split that differs from the regenerated one
    flipped.to_csv(proj2 / "manifests" / "ham10000_split.csv", index=False)
    cfg = load_config(ROOT / "configs" / "base.yaml", {**env["ov"], "paths.project_root": str(proj2)})
    r2 = prepare_data(cfg, dedup=False, threads=2, bcn="no", published=None)
    assert r2["ham_split"] == {"source": "existing file", "identical_to_regenerated": False}
    assert pd.read_csv(proj2 / "manifests" / "ham10000_split.csv").equals(flipped)


def test_train_resume_is_exact_with_workers(env):
    """With two persistent DataLoader workers, an interrupted and resumed run is identical to an uninterrupted one."""
    from dermshift.pipeline import run_experiment

    ov = {**env["ov"], "optim.max_epochs": 3, "optim.t_max": 3, "data.num_workers": 2}
    run_experiment(MODEL_DIR / "deeplabv3plus.yaml", 1, "isic2018", evaluate=False, root_overrides={**ov, "experiment": "ref"})
    p = run_experiment(MODEL_DIR / "deeplabv3plus.yaml", 1, "isic2018", max_minutes=0, evaluate=False, root_overrides=ov)
    assert p["status"] == "paused"
    r = run_experiment(MODEL_DIR / "deeplabv3plus.yaml", 1, "isic2018", root_overrides=ov)
    assert r["status"] == "max_epochs" and set(r["eval"]) == {"ham10000_val", "isic2018_test", "ph2"}
    runs = env["proj"] / "runs" / "isic2018"
    a = torch.load(runs / "ref" / "seed1" / "best.pt", weights_only=False)["model"]
    b = torch.load(runs / "deeplabv3plus" / "seed1" / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a), "resumed run differs from uninterrupted run"
    assert pd.read_csv(runs / "deeplabv3plus" / "seed1" / "train_log.csv")["epoch"].tolist() == [0, 1, 2]
    pi = pd.read_csv(runs / "deeplabv3plus" / "seed1" / "eval" / "isic2018_test" / "per_image.csv")
    assert {"nsd_1pct", "nsd_5px", "log_area_ratio", "hd95_pct"} <= set(pi.columns)
    s = json.loads((runs / "deeplabv3plus" / "seed1" / "eval" / "ph2" / "summary.json").read_text())
    assert s["metrics_version"] == 2 and s["n_images"] == 4


def test_epoch_travels_with_index(env):
    """Augmentation depends on the epoch carried by the sampler, not on dataset.set_epoch."""
    from dermshift.data.dataset import SegDataset
    from dermshift.engine import EpochSampler

    cfg = env["cfg"]
    ds = SegDataset(env["proj"] / "cache" / "ham10000", cfg, np.arange(8), train=True, seed=3)
    a0, a1 = ds[(2, 0)]["image"], ds[(2, 1)]["image"]
    ds.set_epoch(1)
    assert torch.equal(ds[(2, 0)]["image"], a0) and not torch.equal(a0, a1)
    s = EpochSampler(8, seed=3)
    s.set_epoch(5); o5 = list(s)
    s.set_epoch(6); o6 = list(s)
    assert {e for _, e in o5} == {5} and [k for k, _ in o5] != [k for k, _ in o6]


def test_queue_and_analysis(env):
    """Queue for seed 42 (train, ISIC val, BCN20000, test-time transforms), then the analyses."""
    from dermshift import queue

    st = queue.run_queue(env["ov"], seeds=(42,))
    assert (st.state == "done").all(), st[st.state != "done"]
    rd = env["proj"] / "runs" / "isic2018" / "cascade" / "seed42" / "eval"
    assert {"ham10000_val", "isic2018_test", "ph2", "isic2018_val", "bcn20000"} <= {p.name for p in rd.iterdir()}
    assert (rd / "bcn20000" / "unlabelled.csv").exists() and (rd / "ph2__reinhard" / "per_image.csv").exists()
    assert (env["proj"] / "manifests" / "color_stats_isic2018.json").exists()
    assert not (env["proj"] / "runs" / "ham" / "unet" / "seed42" / "last.pt").exists()

    from dermshift.analysis import run_all
    res = run_all(env["cfg"], seeds=(42,))
    out = env["proj"] / "results" / "analysis"
    g1 = pd.read_csv(out / "G1_gap_tests.csv")
    assert set(g1.direction) == {"HAM->ISIC", "ISIC->HAM"} and g1.p_holm.between(0, 1).all()
    assert len(pd.read_csv(out / "G2_area_bias.csv")) and len(pd.read_csv(out / "G2_sign_reversal.csv")) == 7
    dom = pd.read_csv(out / "G3G4_dominance.csv")
    assert set(dom.group) == {"acquisition", "population", "architecture", "dataset"}
    assert dom.general_dominance.sum() == pytest.approx(dom.R2m_full.iloc[0], rel=1e-6)
    assert len(pd.read_csv(out / "G4_reweighting.csv")) == 14
    assert len(pd.read_csv(out / "G3_interventions.csv"))
    fp = pd.read_csv(out / "G6_failure_prediction.csv")
    assert {"arch_disagreement", "ensemble_entropy"} <= set(fp.indicator)
    assert len(pd.read_csv(out / "G6_bcn_expected_dice.csv"))
    dm = pd.read_csv(out / "domain_map.csv")
    assert len(dm) == 10 and (dm.fid_pca >= -1e-6).all() and (dm.pca_dims >= 1).all()
    assert "dominance" in res
    # secondary analyses
    assert {"acquisition", "population", "architecture", "dataset"} >= set(pd.read_csv(out / "S1_gap_dominance_dice.csv").group)
    d = pd.read_csv(out / "S3_gt_annotation_descriptors.csv")
    assert {"ham10000_val", "isic2018_test", "ph2", "isic2018_val", "isic2018_train"} <= set(d["set"])
    assert "isic2018_val" not in set(pd.read_csv(out / "G2_area_bias.csv")["set"])
    assert (out / "S4_failure_rate.csv").exists()


def test_verify_all_duplicates(env):
    from dermshift.analysis import load_exclusions
    from dermshift.pipeline import verify_duplicates

    summ = verify_duplicates(env["cfg"], threshold=0.9)       # synthetic re-encodings: NCC ~0.9-0.99
    man = env["proj"] / "manifests"
    assert {"set_a", "set_b", "candidates", "ncc_gt_0.9"} <= set(summ.columns) and len(summ) >= 10
    bex = pd.read_csv(man / "bcn20000_excluded.csv")
    assert "ISIC_0060000" in set(bex.image_id)                       # BCN copy of an ISIC test image
    ev = pd.read_csv(man / "eval_excluded.csv")
    split = pd.read_csv(man / "ham10000_split.csv").set_index("image_id")
    leak = ("ph2", "IMD002") in set(zip(ev["set"], ev["image_id"]))
    assert leak == (split.loc["ISIC_0024306", "split"] == "train")    # only duplicates of TRAINING images count
    ev_ex, bcn_ex = load_exclusions(man)
    assert "ISIC_0060000" in bcn_ex
    rev = pd.read_csv(man / "duplicates_review.csv")                  # a 'different' decision is respected
    i = rev.index[(rev.set_b == "bcn20000") & (rev.id_b == "ISIC_0060000") | (rev.set_a == "bcn20000") & (rev.id_a == "ISIC_0060000")][0]
    rev.loc[i, "decision"] = "different"
    rev.to_csv(man / "duplicates_review.csv", index=False)
    assert "ISIC_0060000" not in load_exclusions(man)[1]
    verify_duplicates(env["cfg"], threshold=0.9)                      # re-running keeps the entered decision
    assert (pd.read_csv(man / "duplicates_review.csv")["decision"] == "different").sum() == 1
    ncc = pd.read_csv(man / "near_duplicates_isic2018_pool_vs_isic2018_test_ncc.csv")
    assert (ncc.iloc[0].query_id, ncc.iloc[0].ref_id) == ("ISIC_0000000", "ISIC_0036000") and ncc.iloc[0].ncc > 0.9
    tr = pd.read_csv(man / "isic2018_train_test_review.csv", keep_default_na=False)
    assert len(tr) == len(ncc) and (tr["decision"] == "").all()


def test_cli(env):
    args = [f"--set=paths.datasets_root={env['ov']['paths.datasets_root']}",
            f"--set=paths.project_root={env['ov']['paths.project_root']}"]
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), *args, "status"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "train" in out.stdout and "done" in out.stdout


def test_paper_figures_and_panel(env):
    """Figures and tables from the analysis CSVs; review and typical-case panels from caches and probabilities."""
    from dermshift.figures import annotation_panel, make_figures

    res = env["proj"] / "results" / "analysis"
    out = env["proj"] / "results" / "figures"
    done = make_figures(res, out)
    for k in ("design", "gap_vs_tolerance", "area_bias", "annotation_descriptors", "table_gaps", "table_dominance"):
        assert done[k], k
        assert (out / f"{k}.pdf").exists() or (out / f"{k}.tex").exists()
    r = annotation_panel(env["cfg"], experiment="unet", seeds=(42,), n_extreme=2, n_random=2)
    assert r["cases"] > 0 and any(f.endswith(".png") for f in r["files"])
    cases = pd.read_csv(Path(r["out"]) / "panel_cases.csv")
    assert {"set", "image_id", "kind", "review"} <= set(cases.columns)
    r = annotation_panel(env["cfg"], experiment="unet", seeds=(42,), mode="typical", n_typical=1, min_dice=0.0, min_area=0.0)
    assert r["cases"] > 0 and any(f.endswith("typical_cases.png") for f in r["files"])


def test_robustness_thresholds(env):
    from dermshift.analysis import area_bias_no_failures

    avg = pd.read_csv(env["proj"] / "results" / "analysis" / "per_image_seed_avg.csv")
    b, fr = area_bias_no_failures(avg, thresholds=(0.0, 0.5))
    assert len(fr) and {"fail_0", "fail_50"} <= set(fr.columns)
    if len(b):
        assert np.allclose(b[b.threshold == 0.0]["share_kept"], 1.0)      # nothing removed at threshold 0


def test_duplicate_annotation_s5(env):
    """S5 on the planted duplicate (ISIC_0000000 in the pool is ISIC_0036000 in the test set)."""
    from dermshift.annotation_pairs import run_duplicate_annotation

    r = run_duplicate_annotation(env["cfg"], ncc_min=0.9)
    assert r["pairs"] >= 1 and r["selection"].startswith("ncc")
    d = pd.read_csv(env["proj"] / "results" / "analysis" / "S5_duplicate_pairs.csv")
    assert {"train_id", "test_id", "flip", "log_area_ratio_test_vs_train", "dice_train_test"} <= set(d.columns)
    assert ("ISIC_0000000", "ISIC_0036000") in set(zip(d.train_id, d.test_id))
    assert any(f.endswith("duplicate_pairs.png") for f in r["files"])
