"""Every path used by the project, derived from three roots set in ``configs/local.yaml``.

* ``datasets_root``  raw datasets (read-only)
* ``project_root``   manifests, caches, runs and results written by the pipeline
* ``local_root``     optional fast-disk copy of the caches (``null``: read them from ``project_root``)

Expected layout under ``datasets_root``::

    HAM10000/HAM10000_images/ISIC_xxxxxxx.jpg
    HAM10000/HAM10000_segmentations_lesion_tschandl/ISIC_xxxxxxx_segmentation.png
    HAM10000/HAM10000_metadata.csv
    ISIC2018/*Task1-2_Training_Input*/   + *Task1_Training_GroundTruth*/
    ISIC2018/*Task1-2_Validation_Input*/ + *Task1_Validation_GroundTruth*/
    ISIC2018/*Task1-2_Test_Input*/       + *Task1_Test_GroundTruth*/
    PH2Dataset/PH2 Dataset images/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp (+ IMDxxx_lesion/)
    BCN20000/images/ISIC_xxxxxxx.jpg + BCN20000/metadata.csv

Runs are stored in ``<project_root>/runs/<train_domain>/<experiment>/seed<k>/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    datasets_root: Path
    project_root: Path
    local_root: Path | None = None

    # ---------------- raw datasets ----------------
    @property
    def ham_dir(self) -> Path:
        return self.datasets_root / "HAM10000"

    @property
    def ham_images(self) -> Path:
        return self.ham_dir / "HAM10000_images"

    @property
    def ham_masks(self) -> Path:
        return self.ham_dir / "HAM10000_segmentations_lesion_tschandl"

    @property
    def ham_metadata(self) -> Path:
        return self.ham_dir / "HAM10000_metadata.csv"

    @property
    def isic_dir(self) -> Path:
        return self.datasets_root / "ISIC2018"

    def isic_test_images(self) -> Path:
        return _unique_glob(self.isic_dir, "*Task1-2_Test_Input*")

    def isic_test_masks(self) -> Path:
        return _unique_glob(self.isic_dir, "*Task1_Test_GroundTruth*")

    def isic_trainval_dirs(self) -> list[tuple[Path, Path]]:
        """ISIC 2018 Task 1 training + validation (image dir, mask dir): the source pool of the
        reverse direction (train_domain = isic2018). Disjoint by ID from the Task 1 test set."""
        return [(_unique_glob(self.isic_dir, "*Task1-2_Training_Input*"), _unique_glob(self.isic_dir, "*Task1_Training_GroundTruth*")),
                (_unique_glob(self.isic_dir, "*Task1-2_Validation_Input*"), _unique_glob(self.isic_dir, "*Task1_Validation_GroundTruth*"))]

    @property
    def ph2_cases(self) -> Path:
        return self.datasets_root / "PH2Dataset" / "PH2 Dataset images"

    @property
    def bcn_dir(self) -> Path:
        return self.datasets_root / "BCN20000"

    # ---------------- project outputs ----------------
    @property
    def manifests(self) -> Path:
        return self.project_root / "manifests"

    @property
    def caches(self) -> Path:
        return self.project_root / "cache"

    @property
    def caches_local(self) -> Path:
        return self.caches if self.local_root is None else self.local_root / "cache"

    @property
    def runs(self) -> Path:
        return self.project_root / "runs"

    @property
    def results(self) -> Path:
        return self.project_root / "results"

    @property
    def analysis_dir(self) -> Path:
        return self.results / "analysis"

    @property
    def figures_dir(self) -> Path:
        return self.results / "figures"

    def run_dir(self, domain: str, experiment: str, seed: int) -> Path:
        return self.runs / domain / experiment / f"seed{seed}"


def _unique_glob(root: Path, pattern: str) -> Path:
    hits = sorted(p for p in root.glob(pattern) if p.is_dir())
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one folder matching '{pattern}' in {root}, found {hits}")
    return hits[0]


def from_config(cfg: dict) -> ProjectPaths:
    p = cfg["paths"]
    lr = p.get("local_root")
    return ProjectPaths(Path(p["datasets_root"]), Path(p["project_root"]), Path(lr) if lr else None)
