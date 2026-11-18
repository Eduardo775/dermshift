# DermShift

Code for *Where the boundary is drawn: annotation conventions drive cross-dataset shift in dermoscopic lesion
segmentation*.

The study trains seven segmentation architectures on HAM10000, on a size-matched HAM10000 subset and on
ISIC 2018 (three seeds each), evaluates them on HAM10000, ISIC 2018 and PH2 with paired per-image statistics,
and estimates label-free risk on BCN20000. The repository reproduces every number, table and figure of the paper.

## Layout

```
configs/      base.yaml (shared protocol), models/*.yaml (one file per architecture), local.example.yaml
dermshift/    package: data, models, training, evaluation, analysis, figures
scripts/      run.py, the command-line interface
data/         released partitions and duplicate reviews (installed by `prepare`)
results/      analysis outputs of the paper (CSV)
tests/        unit tests and an end-to-end test on synthetic data
```

## Installation

Python 3.10 or newer. Install PyTorch for your CUDA version first, then the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q tests          # about 6 minutes on CPU
```

## Data

Download the datasets and place them under one folder (`datasets_root`):

| Dataset | Source | Folder |
|---|---|---|
| HAM10000 images, metadata and lesion masks | Harvard Dataverse, doi:10.7910/DVN/DBW86T | `HAM10000/` |
| ISIC 2018 Task 1 (training, validation, test) | ISIC Challenge 2018 data page | `ISIC2018/` |
| PH2 | PH2 database (Mendonça et al., 2013) | `PH2Dataset/` |
| BCN20000 (optional, no masks) | ISIC Archive, collection 249: `isic image download --collections 249 BCN20000/images` | `BCN20000/` |

The expected file layout is documented in `dermshift/paths.py`. Copy `configs/local.example.yaml` to
`configs/local.yaml` and set `datasets_root` and `project_root` (where caches, runs and results are written).

## Reproducing the study

```bash
python scripts/run.py env                 # check PyTorch, GPU and paths
python scripts/run.py prepare             # manifests, caches, duplicate screening, partitions
python scripts/run.py verify-duplicates   # NCC of cross-set candidates, exclusion lists
python scripts/run.py queue               # 63 training runs + evaluations (resumable)
python scripts/run.py analyze             # results/analysis/*.csv
python scripts/run.py duplicate-masks     # analysis S5
python scripts/run.py figures             # figures and tables -> results/figures
python scripts/run.py panel --mode typical
```

`prepare` installs the partitions released in `data/`, so the same images are used in every set. `queue` can be
stopped at any time and resumes from the last checkpoint. Training took about 75 GPU hours on a 16 GB
RTX 5060 Ti. Single runs can be launched with `train` and `evaluate` (see `python scripts/run.py -h`); any
configuration value can be overridden with `--set key=value`.

## Outputs and the paper

| Paper item | Output |
|---|---|
| Study design and domain map | `results/figures/design` |
| Paired gap vs boundary tolerance | `gap_vs_tolerance`, table `table_gaps` |
| Area bias per evaluation set | `area_bias`, table `table_area_bias` |
| Reference-mask descriptors (S3) | `annotation_descriptors` |
| Typical cases | `typical_cases` (`panel --mode typical`) |
| Same photograph, two masks (S5) | `duplicate_pairs`, `duplicate_masks/` |
| Test-time interventions | `interventions` |
| Factor attribution and reweighting | tables `table_dominance`, `table_reweighting` |
| Label-free failure indicators, BCN20000 risk | `failure_indicators` |
| Robustness to gross failures (S4) | `area_bias_no_failures`, table `table_robustness` |

The CSV files behind these outputs are in `results/` (see `results/README.md`).

## Citation

See `CITATION.cff`. The code is released under the MIT license; the datasets keep their own licenses.
