# AutoScore Ensemble and Distillation

A CEG5003 research project investigating how bootstrap ensembles and distillation affect the stability of interpretable clinical scoring tables. The pipeline compares an AutoScore ensemble, a distilled scoring table, and a single-model baseline across random seeds. It reports discrimination (AUC), pairwise decision disagreement (PDR), and sample-level stability metrics.

This is research code. Stability improvements are experimental findings to evaluate, rather than guarantees of the method.

## What the experiment does

For each seed, the pipeline removes 10% of the fixed training set, trains 30 AutoScore models on bootstrap samples, and averages their scores. It resamples 20,000 training rows, assigns ensemble-generated labels, and fits one distilled scoring table. A single AutoScore baseline is trained on the same reduced training set. All three methods are evaluated on the fixed test set.

The main pipeline is implemented in R. Python is used for analysis and for creating independent demonstration data.

## Install

Run commands from the repository root. Tested with **R 4.5.2**, **AutoScore 1.1.0**, and **Python 3.10.19**. Have `Rscript` and `python` available in your terminal.

```sh
python -m pip install -r requirements.txt
Rscript scripts/setup_r.R
```

Alternatively, create the Python environment with `conda env create -f environment.yml`, then `conda activate autoscore-ensemble`. R is installed separately. The R setup script installs missing CRAN dependencies and the bundled AutoScore source. On systems requiring source compilation, R's package build tools may also be needed.

`requirements.txt` pins the four direct Python dependencies. `requirements-r.csv` records the tested R package versions; the setup script installs available CRAN dependencies, not a complete historical environment lock. Each experiment saves `sessionInfo.txt` so its actual R environment can be checked. Package/platform changes can affect results.

## Quick check without patient data

```sh
python scripts/make_demo_data.py
Rscript run_experiment.R --demo
python scripts/analysis/analysis.py --seed-start 1001 --seed-end 1002 --results-dir outputs/demo/results --output-root outputs/demo/analysis
```

The generator creates independent toy data (960 training rows and 240 test rows), without reading clinical files. `--demo` uses seeds 1001 and 1002 and writes only to `outputs/demo/results/`. It retains the 30-model / 20,000-row experiment settings. The toy data checks execution; its metrics are not clinical findings or a reproduction of the original study. Generating demo data again will refuse to overwrite existing demo CSVs.

## Run with your own fixed split

The original prepared split is included at:

```text
data/split/train_data.csv
data/split/test_data.csv
```

Both files must have the same columns in the same order: a numeric `label` column (0 = survival, 1 = death) and numeric feature columns. Both classes must be present in each file. Missing or infinite values must be handled before training. For the original experiment, the 11 features are:

```text
urineoutput, lactate_min, bun_mean, sysbp_min, inr_max, age,
metastatic_cancer, spo2_mean, sodium_max, aniongap_max, creatinine_min
```

Start with two seeds:

```sh
Rscript run_experiment.R --seed-start 1001 --seed-end 1002
python scripts/analysis/analysis.py --seed-start 1001 --seed-end 1002
```

For the original 500-run seed range:

```sh
Rscript run_experiment.R --seed-start 1001 --seed-end 1500
python scripts/analysis/analysis.py --seed-start 1001 --seed-end 1500
```

Edit `config.R` for model parameters. The default runs only seeds **1001-1002**. Use the full-range command above for 500 runs. Running an existing seed again overwrites that seed's result files; choose `--results-dir outputs/new_run/results` to keep runs separate and pass the same directory as `--results-dir` to the analysis command. Custom input files can be selected with `--train-data PATH --test-data PATH`.

The analysis uses a fixed score threshold of 46 by default; change it with `--threshold`. This is a score threshold, not a probability. Keep the same test samples and labels across runs; the PDR analysis aligns rows by `sample_id` and rejects inconsistent sample sets or labels.

## Outputs and additional analysis

Each `outputs/results/exp_seed<N>/` contains model RDS files, per-sample predictions for all three methods, `summary.csv`, a valid `config.json`, a feature-drop summary, and `sessionInfo.txt`. Analysis writes a new timestamped folder under `outputs/analysis/` with AUC tables, a distribution plot, and PDR summaries.

Two additional scripts scan all available seeds in `outputs/results/`:

```sh
python scripts/analysis/sample_score_metrics_and_boxplots.py
python scripts/analysis/sample_threshold_decision_metrics.py
```

The first computes score variability and hard-sample comparisons. The second scans score thresholds 30-60 and chooses a representative threshold per method by requiring sensitivity >= 0.8, then maximizing balanced accuracy. It also reports split strength, correctness, signed reliability, and cost-sensitive reliability. Their settings, including paths, seed filters, and plot options, are at the top of each script. For demo results, change `RESULTS_DIR` to `outputs/demo/results` before using these two scripts.

## Data source and scope of reproduction

The local research data was derived from the public CSV supplement to:

Hou N, Li M, He L, et al. **Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost.** Journal of Translational Medicine 18, 462 (2020). [Article and Additional file 1](https://doi.org/10.1186/s12967-020-02620-5).

The project's prepared dataset has 4,559 rows, with 3,648 training rows and 911 test rows. Local file comparisons show selection of the 11 features above, renaming `thirtyday_expire_flag` to `label`, and filling missing numeric values with full-dataset column medians. These are this project's processing choices, not a claim to reproduce the paper's exact processing, split, or best XGBoost AUC.

The two original prepared CSVs are bundled so the commands above use the exact retained split, without guessing a split seed. See `DATA_SOURCE.md` for attribution, the article's data rights statement, processing notes, and scope. Compatible package versions are still needed for numerical agreement. You may replace the files with your own fixed split in the same format; doing so runs a new experiment.

## Known methodological limitations

- The historical pipeline chooses an ensemble threshold using test-set labels and uses that threshold to label distillation data. The test set therefore influences training, which can bias evaluation.
- Historical missing-value filling used medians from the whole dataset before splitting. A separate validation study should fit preprocessing on training data only.
- Training data is reused as the AutoScore validation input. Threshold selection in the additional analysis also uses test outcomes.

These behaviors are documented to keep the historical experiment reproducible. An evaluation with independent training, validation, and test sets would be a separate experimental protocol. No clinical deployment claim is made.

## Public files

The publication package contains the main training code, three analysis scripts, dependency files, the demo generator, and AutoScore runtime source/documentation. It includes only the two documented public-source input CSVs; it excludes the raw table, generated data, predictions, trained models, old experiments, local editor settings, and the original Git history. `.gitignore` is supplementary; it does not erase files from historical commits.

## Acknowledgements

Built on [AutoScore](https://github.com/nliulab/AutoScore): Xie F, Chakraborty B, Ong MEH, Goldstein BA, Liu N. *AutoScore: A Machine Learning-Based Automatic Clinical Score Generator and Its Application to Mortality Prediction Using Electronic Health Records.* JMIR Medical Informatics (2020). The bundled AutoScore source declares GPL (>= 2); see `THIRD_PARTY_NOTICES.md` and its `DESCRIPTION` file.
