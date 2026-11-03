# Dataset provenance

## Source

Hou N, Li M, He L, et al. **Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost.** Journal of Translational Medicine 18, 462 (2020).

- DOI: https://doi.org/10.1186/s12967-020-02620-5
- Official article and CSV supplement (Additional file 1): https://link.springer.com/article/10.1186/s12967-020-02620-5
- Supplement filename: `12967_2020_2620_MOESM1_ESM.csv`.

The project author obtained the source data from the official public channel. The article's rights statement applies CC0 to data made available with the article unless otherwise stated in a data credit line. This provenance applies to the article's published supplement and the two prepared files below; it is not a statement that the full MIMIC-III database may be redistributed without its separate access conditions.

## Prepared files included here

| File | Rows | Columns |
| --- | ---: | ---: |
| `data/split/train_data.csv` | 3,648 | 12 |
| `data/split/test_data.csv` | 911 | 12 |

These are the exact fixed input files retained from this project's experiments, rather than a new randomized split. Each has a binary outcome column (`label`, 1 = 30-day death) and 11 predictors. No patient/admission identifiers, names, contact information, or free-text notes are included.

Local file comparisons establish the following transformations: select the 11 predictors listed in README.md; rename `thirtyday_expire_flag` to `label`; fill missing numeric values with whole-dataset column medians. The original split seed is not recorded; shipping the existing split preserves its row membership and order without guessing that seed. Both files together contain all 4,559 processed rows, with no repeated rows.

This is the project's own prepared version, not the paper authors' exact modeling data and split. No claim is made to reproduce their best XGBoost AUC. The full raw table, additional generated data, individual predictions, and trained models are excluded. See README.md for methodological limitations and PUBLIC_FILES.json for SHA-256 checksums of all published files.
