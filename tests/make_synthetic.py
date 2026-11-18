"""Create a tiny synthetic copy of the raw dataset layout (same folder/file naming as the
real datasets) to test the whole pipeline end-to-end without real data."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _imwrite(path: Path, img) -> None:
    """Unicode-safe write (the tests put the datasets under a non-ASCII folder, as can happen on Windows)."""
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    buf.tofile(str(path))


def _sample(rng, h, w):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = rng.integers(150, 225, 3)
    low = cv2.resize(rng.normal(0, 25, (6, 8, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img + low + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)   # distinct pHash per image
    m = np.zeros((h, w), np.uint8)
    c = (int(rng.integers(w // 3, 2 * w // 3)), int(rng.integers(h // 3, 2 * h // 3)))
    ax = (int(rng.integers(w // 10, w // 4)), int(rng.integers(h // 10, h // 4)))
    cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    col = rng.integers(40, 110, 3)
    img[m > 0] = np.clip(col + rng.normal(0, 10, (int((m > 0).sum()), 3)), 0, 255).astype(np.uint8)
    return img, m


def make(root: Path, n_ham: int = 40, n_isic: int = 6, n_ph2: int = 4, n_xval=(16, 4), n_bcn: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    ham = root / "HAM10000"
    (ham / "HAM10000_images").mkdir(parents=True, exist_ok=True)
    (ham / "HAM10000_segmentations_lesion_tschandl").mkdir(parents=True, exist_ok=True)
    (ham / "masks_S").mkdir(exist_ok=True)          # must be ignored
    meta = []
    dxs = ["nv", "mel", "bkl", "bcc"]
    for i in range(n_ham):
        iid = f"ISIC_{24306 + i:07d}"
        img, m = _sample(rng, 150, 200)
        _imwrite((ham / "HAM10000_images" / f"{iid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _imwrite((ham / "HAM10000_segmentations_lesion_tschandl" / f"{iid}_segmentation.png"), m)
        meta.append({"lesion_id": f"HAM_{i // 2:07d}", "image_id": iid, "dx": dxs[(i // 2) % 4],
                     "dx_type": "histo", "age": 50, "sex": "male", "localization": "back"})
    pd.DataFrame(meta).to_csv(ham / "HAM10000_metadata.csv", index=False)
    isic = root / "ISIC2018"
    ti, tg = isic / "ISIC2018_Task1-2_Test_Input", isic / "ISIC2018_Task1_Test_GroundTruth"
    ti.mkdir(parents=True, exist_ok=True); tg.mkdir(parents=True, exist_ok=True)
    (isic / "processed").mkdir(exist_ok=True)
    for i in range(n_isic):
        iid = f"ISIC_{36000 + i:07d}"
        h, w = [(160, 240), (200, 200), (300, 400), (120, 220), (250, 330), (180, 260)][i % 6]
        img, m = _sample(rng, h, w)
        _imwrite((ti / f"{iid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _imwrite((tg / f"{iid}_segmentation.png"), m)
    (ti / "LICENSE.txt").write_text("x")
    # ISIC 2018 Task 1 training and validation images (the ISIC training pool)
    k = 0
    for folder_in, folder_gt, n in (("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth", n_xval[0]),
                                    ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth", n_xval[1])):
        (isic / folder_in).mkdir(exist_ok=True); (isic / folder_gt).mkdir(exist_ok=True)
        (isic / folder_in / "ATTRIBUTION.txt").write_text("x")
        for _ in range(n):
            iid = f"ISIC_{k:07d}"; k += 1
            img, m = _sample(rng, 180, 240)
            _imwrite((isic / folder_in / f"{iid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _imwrite((isic / folder_gt / f"{iid}_segmentation.png"), m)
    # planted near-duplicates: training image ISIC_0000000 := test image ISIC_0036000;
    # ISIC_0000002 := ISIC_0000001 (duplicate inside the pool -> same partition)
    import shutil
    shutil.copy(ti / "ISIC_0036000.jpg", isic / "ISIC2018_Task1-2_Training_Input" / "ISIC_0000000.jpg")
    _m = cv2.imdecode(np.fromfile(str(tg / "ISIC_0036000_segmentation.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    _imwrite(isic / "ISIC2018_Task1_Training_GroundTruth" / "ISIC_0000000_segmentation.png",
             cv2.resize(_m, (240, 180), interpolation=cv2.INTER_NEAREST))
    _img = cv2.imdecode(np.fromfile(str(ti / "ISIC_0036000.jpg"), np.uint8), cv2.IMREAD_COLOR)
    _imwrite(isic / "ISIC2018_Task1-2_Training_Input" / "ISIC_0000000.jpg", cv2.resize(_img, (240, 180)))
    shutil.copy(isic / "ISIC2018_Task1-2_Training_Input" / "ISIC_0000001.jpg", isic / "ISIC2018_Task1-2_Training_Input" / "ISIC_0000002.jpg")
    shutil.copy(isic / "ISIC2018_Task1_Training_GroundTruth" / "ISIC_0000001_segmentation.png",
                isic / "ISIC2018_Task1_Training_GroundTruth" / "ISIC_0000002_segmentation.png")
    # BCN20000 (isic-cli layout): images + metadata.csv; 2 rows without diagnosis (test split), 1 clinical image
    bcn = root / "BCN20000"                       # current ISIC Archive export: images next to metadata.csv
    bcn.mkdir(parents=True, exist_ok=True)
    rows = []
    sites = ["anterior torso", "head/neck", "lower extremity", "palms/soles"]
    for i in range(n_bcn + 3):
        iid = f"ISIC_{60000 + i:07d}"
        img, _ = _sample(rng, 256, 256)
        _imwrite(bcn / f"{iid}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        labelled = i < n_bcn
        rows.append({"isic_id": iid, "image_type": "clinical: overview" if i == n_bcn + 2 else "dermoscopic",
                     "diagnosis_1": ("Benign" if i % 3 else "Malignant") if labelled else "",
                     "diagnosis_3": ("Nevus" if i % 3 else "Melanoma, NOS") if labelled else "",
                     "anatom_site_1": sites[i % 4], "anatom_site_2": "", "lesion_id": f"IL_{i // 2:07d}", "age_approx": 50, "sex": "female"})
    pd.DataFrame(rows).to_csv(root / "BCN20000" / "metadata.csv", index=False)
    # planted BCN duplicate: ISIC_0060000 := ISIC test image ISIC_0036002 exported as a 1:1 resize
    _t = cv2.imdecode(np.fromfile(str(ti / "ISIC_0036002.jpg"), np.uint8), cv2.IMREAD_COLOR)
    _imwrite(bcn / "ISIC_0060000.jpg", cv2.resize(_t, (256, 256), interpolation=cv2.INTER_AREA))
    ph2 = root / "PH2Dataset" / "PH2 Dataset images"
    for i in range(n_ph2):
        cid = f"IMD{i + 2:03d}"
        (ph2 / cid / f"{cid}_Dermoscopic_Image").mkdir(parents=True, exist_ok=True)
        (ph2 / cid / f"{cid}_lesion").mkdir(parents=True, exist_ok=True)
        img, m = _sample(rng, 172, 230)
        _imwrite((ph2 / cid / f"{cid}_Dermoscopic_Image" / f"{cid}.bmp"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _imwrite((ph2 / cid / f"{cid}_lesion" / f"{cid}_lesion.bmp"), m)
    # planted evaluation leak: PH2 case IMD002 := HAM10000 image ISIC_0024306 (re-saved at PH2 size)
    hi = cv2.imdecode(np.fromfile(str(ham / "HAM10000_images" / "ISIC_0024306.jpg"), np.uint8), cv2.IMREAD_COLOR)
    hm = cv2.imdecode(np.fromfile(str(ham / "HAM10000_segmentations_lesion_tschandl" / "ISIC_0024306_segmentation.png"), np.uint8),
                      cv2.IMREAD_GRAYSCALE)
    c = ph2 / "IMD002"
    _imwrite(c / "IMD002_Dermoscopic_Image" / "IMD002.bmp", cv2.resize(hi, (230, 172), interpolation=cv2.INTER_AREA))
    _imwrite(c / "IMD002_lesion" / "IMD002_lesion.bmp", cv2.resize(hm, (230, 172), interpolation=cv2.INTER_NEAREST))
    return root


if __name__ == "__main__":
    make(Path(sys.argv[1]))
