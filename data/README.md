# Released partitions and duplicate reviews

`prepare` copies these files into `<project_root>/manifests/` before building the caches, so the study is
reproduced with exactly the same images in every partition.

| File | Content |
|---|---|
| `ham10000_split.csv` | HAM10000 train / validation fold (grouped by lesion) |
| `isic2018_split.csv` | ISIC 2018 training-pool partition (train / val) after removing near-duplicates of the evaluation sets |
| `ham10000_matched.csv` | size-matched HAM10000 training subset (2,182 images) |
| `eval_excluded.csv` | evaluation images excluded as duplicates of training images (empty in this study) |
| `bcn20000_excluded.csv` | BCN20000 images excluded as duplicates of labelled images |
| `isic2018_train_test_review.csv` | ISIC 2018 train-test pairs confirmed on visual review as the same photograph (61 pairs) used in analysis S5 |
