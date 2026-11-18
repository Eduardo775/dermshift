# Results

Output of `python scripts/run.py analyze` and `duplicate-masks` for the three seeds of the runs reported in the
paper. Training is not bit-wise deterministic (cuDNN autotuning), so retraining with this code reproduces these
values up to seed-level variation. Per-image tables (`per_image_seed_avg.csv`) and the detailed S4 tables of the
original runs are not included; S4 summary values are reported in the paper. File prefixes follow Table 2 of the
paper.

| Files | Analysis |
|---|---|
| `G1_gap_tests.csv` | paired in-domain vs out-of-domain gap, all metrics |
| `G2_gap_vs_tolerance.csv`, `G2_area_bias.csv`, `G2_sign_reversal.csv` | boundary tolerance and area bias |
| `G3G4_mixed_model_coefficients.csv`, `G3G4_dominance.csv` | mixed-effects model and dominance analysis |
| `G3_interventions.csv` | test-time colour normalisation and hair removal |
| `G4_reweighting.csv` | population reweighting |
| `G6_failure_prediction.csv`, `G6_bcn_expected_dice.csv` | label-free failure indicators and BCN20000 risk |
| `S1_*.csv` | mixed model of the paired gap |
| `S2_*.csv` | evaluation on the ISIC 2018 validation partition |
| `S3_*.csv` | reference-mask descriptors and tightness |
| `S4_*.csv` | area bias after removing gross failures |
| `S5_*.csv` | train vs test masks of the same photograph |
| `domain_map.csv` | Fréchet distance and MMD between datasets |
| `per_image_seed_avg.csv` | per-image metrics averaged over seeds (input of all tests) |
