"""
Analysis script:
  Task 1: Compare ensemble_auc, final_table_auc, baseline_auc across experiment folders.
  Task 2: Compute pairwise disagreement rate (PDR) for final table, ensemble and baseline predictions.
"""

import argparse
import os
import re
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============== Configuration ==============
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
THRESHOLD = 46  # Classification threshold for pred_score
DEFAULT_SEED_START = 1001
DEFAULT_SEED_END = 1500
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")
# ===========================================

def get_experiment_folders(results_dir, seed_start=None, seed_end=None):
    """Get experiment subdirectories in results_dir, optionally filtered by seed range."""
    pattern = re.compile(r"^exp_seed(\d+)$")
    folders = []
    for name in os.listdir(results_dir):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue

        match = pattern.match(name)
        if match is None:
            continue

        seed = int(match.group(1))
        if seed_start is not None and seed_end is not None and not (seed_start <= seed <= seed_end):
            continue

        folders.append((seed, name, path))

    folders.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in folders]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze experiment outputs in outputs/results/exp_seedXXXX.")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="Start seed (inclusive), e.g., 1001")
    parser.add_argument("--seed-end", type=int, default=DEFAULT_SEED_END, help="End seed (inclusive), e.g., 1050")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Classification threshold for pred_score")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory containing exp_seedXXXX folders")
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT_DIR,
        help="Root directory where output folder will be created",
    )
    args = parser.parse_args()

    if (args.seed_start is None) != (args.seed_end is None):
        parser.error("--seed-start and --seed-end must be set together.")
    if args.seed_start is not None and args.seed_start > args.seed_end:
        parser.error("--seed-start must be <= --seed-end.")
    return args


def create_run_output_dir(output_root, seed_start=None, seed_end=None):
    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"seed{seed_start}-{seed_end}" if seed_start is not None else "seed_all"
    base_name = f"analysis_auc_pwdr_{seed_tag}_{timestamp}"
    output_dir = os.path.join(output_root, base_name)

    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(output_root, f"{base_name}_{suffix}")
        suffix += 1

    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def read_metric_from_summary(summary_file, metric_name):
    """Read one metric value from summary.csv, return None if missing."""
    if not os.path.exists(summary_file):
        return None

    df = pd.read_csv(summary_file)
    row = df[df["metric"] == metric_name]
    if row.empty:
        return None
    return float(row["value"].values[0])


# ===================== Task 1: AUC Variance Comparison =====================
def task1_auc_variance_comparison(folders, output_dir):
    """
    Compare ensemble_auc, final_table_auc and baseline_auc from each experiment summary.csv.
    Output per-experiment table + summary stats + visualization.
    """
    records = []

    for name, path in folders:
        summary_file = os.path.join(path, "summary.csv")
        ensemble_auc = read_metric_from_summary(summary_file, "ensemble_auc")
        final_auc = read_metric_from_summary(summary_file, "final_table_auc")
        baseline_auc = read_metric_from_summary(summary_file, "baseline_auc")

        if final_auc is None:
            print(f"[Task1] Skipping {name}: final_table_auc not found")
            continue

        records.append(
            {
                "experiment": name,
                "ensemble_auc": ensemble_auc,
                "final_table_auc": final_auc,
                "baseline_auc": baseline_auc,
            }
        )
        ensemble_str = "MISSING" if ensemble_auc is None else f"{ensemble_auc:.6f}"
        baseline_str = "MISSING" if baseline_auc is None else f"{baseline_auc:.6f}"
        print(
            f"[Task1] {name}: ensemble_auc = {ensemble_str}, "
            f"final_table_auc = {final_auc:.6f}, baseline_auc = {baseline_str}"
        )

    if not records:
        print("[Task1] No valid experiment summaries found.")
        return

    df_auc = pd.DataFrame(records)
    compare_csv = os.path.join(output_dir, "auc_comparison_per_experiment.csv")
    df_auc.to_csv(compare_csv, index=False)
    print(f"[Task1] Per-experiment AUC table saved to: {compare_csv}")

    # Use experiments that contain all 3 AUC metrics for fair variance comparison.
    df_common = df_auc.dropna(subset=["ensemble_auc", "final_table_auc", "baseline_auc"]).copy()
    if df_common.empty:
        print("[Task1] Missing AUC metrics, cannot compare variance across all three methods.")
        return

    ensemble_values = df_common["ensemble_auc"].values
    final_values = df_common["final_table_auc"].values
    baseline_values = df_common["baseline_auc"].values

    def build_stat_row(metric_name, values):
        return {
            "metric": metric_name,
            "n": len(values),
            "mean": np.mean(values),
            "std": np.std(values, ddof=0),
            "min": np.min(values),
            "max": np.max(values),
        }

    summary = pd.DataFrame(
        [
            build_stat_row("ensemble_auc", ensemble_values),
            build_stat_row("final_table_auc", final_values),
            build_stat_row("baseline_auc", baseline_values),
        ]
    )
    summary_path = os.path.join(output_dir, "auc_variance_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[Task1] Variance summary saved to: {summary_path}")

    ensemble_std = summary.loc[summary["metric"] == "ensemble_auc", "std"].values[0]
    final_std = summary.loc[summary["metric"] == "final_table_auc", "std"].values[0]
    baseline_std = summary.loc[summary["metric"] == "baseline_auc", "std"].values[0]
    print("\n[Task1] Variance comparison on matched experiments:")
    print(
        f"[Task1] ensemble_auc:    mean={np.mean(ensemble_values):.6f}, std={ensemble_std:.6f}, "
        f"min={np.min(ensemble_values):.6f}, max={np.max(ensemble_values):.6f}"
    )
    print(
        f"[Task1] final_table_auc: mean={np.mean(final_values):.6f}, std={final_std:.6f}, "
        f"min={np.min(final_values):.6f}, max={np.max(final_values):.6f}"
    )
    print(
        f"[Task1] baseline_auc:    mean={np.mean(baseline_values):.6f}, std={baseline_std:.6f}, "
        f"min={np.min(baseline_values):.6f}, max={np.max(baseline_values):.6f}"
    )
    print(f"[Task1] std(final_table_auc) - std(baseline_auc) = {final_std - baseline_std:.6f}")
    print(f"[Task1] std(final_table_auc) - std(ensemble_auc) = {final_std - ensemble_std:.6f}")

    # Draw a simple distribution comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(df_common) >= 2:
        sns.kdeplot(ensemble_values, fill=True, alpha=0.30, label="Ensemble AUC", ax=ax)
        sns.kdeplot(final_values, fill=True, alpha=0.35, label="Final Table AUC", ax=ax)
        sns.kdeplot(baseline_values, fill=True, alpha=0.35, label="Baseline AUC", ax=ax)
    else:
        ax.scatter(ensemble_values, [1.2], label="Ensemble AUC")
        ax.scatter(final_values, [1], label="Final Table AUC")
        ax.scatter(baseline_values, [0.8], label="Baseline AUC")

    ax.axvline(np.mean(ensemble_values), linestyle="-.", linewidth=1.2, label="Ensemble mean")
    ax.axvline(np.mean(final_values), linestyle="--", linewidth=1.2, label="Final mean")
    ax.axvline(np.mean(baseline_values), linestyle=":", linewidth=1.2, label="Baseline mean")
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Ensemble vs Final Table vs Baseline AUC Distribution", fontsize=14)
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "kde_ensemble_final_baseline_auc.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[Task1] Comparison plot saved to: {fig_path}")


# ===================== Task 2: Pairwise Disagreement =====================
def load_binary_prediction(pred_file, threshold, prediction_type):
    """
    Load prediction file and return binary labels based on threshold.

    prediction_type:
      - "final_table": use pred_score
      - "baseline": use pred_score
      - "ensemble": prefer ensemble_mean, then mean(model_*), then pred_score
    """
    df = pd.read_csv(pred_file)
    required = {"sample_id", "true_label"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("A nonempty prediction file with sample_id and true_label is required")
    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be nonmissing and unique")
    if not df["true_label"].isin([0, 1]).all():
        raise ValueError("true_label must contain only 0 and 1")
    df = df.set_index("sample_id").sort_index()

    if prediction_type in {"final_table", "baseline"}:
        if "pred_score" not in df.columns:
            raise ValueError("pred_score column not found")
        score = df["pred_score"].to_numpy()
        score_col = "pred_score"
    elif prediction_type == "ensemble":
        if "ensemble_mean" in df.columns:
            score = df["ensemble_mean"].to_numpy()
            score_col = "ensemble_mean"
        else:
            model_cols = [c for c in df.columns if c.startswith("model_")]
            if model_cols:
                score = df[model_cols].mean(axis=1).to_numpy()
                score_col = "mean(model_*)"
            elif "pred_score" in df.columns:
                score = df["pred_score"].to_numpy()
                score_col = "pred_score"
            else:
                raise ValueError(
                    "No usable score column found. Need one of: ensemble_mean, model_*, pred_score."
                )
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    if not np.isfinite(score).all():
        raise ValueError("Prediction scores must be finite")
    classified = pd.Series((score >= threshold).astype(int), index=df.index)
    return classified, score_col, df["true_label"]


def task2_pairwise_disagreement_single(
    folders,
    threshold,
    output_dir,
    prediction_type,
    pred_filename,
):
    """Compute mean pairwise disagreement rate for one prediction source."""
    all_predictions = {}
    reference_labels = None
    log_prefix = f"[Task2-{prediction_type}]"

    for name, path in folders:
        pred_file = os.path.join(path, pred_filename)
        if not os.path.exists(pred_file):
            print(f"{log_prefix} Skipping {name}: {pred_filename} not found")
            continue

        try:
            classified, score_col, labels = load_binary_prediction(pred_file, threshold, prediction_type)
        except Exception as exc:
            print(f"{log_prefix} Skipping {name}: failed to parse scores ({exc})")
            continue

        if reference_labels is None:
            reference_labels = labels
        elif not labels.index.equals(reference_labels.index) or not np.array_equal(labels, reference_labels):
            raise ValueError(f"{name}: sample IDs or true labels differ across experiments; PDR would be invalid")
        all_predictions[name] = classified
        print(
            f"{log_prefix} {name}: {len(classified)} samples, "
            f"{classified.sum()} predicted positive (threshold={threshold}, score={score_col})"
        )

    if len(all_predictions) < 2:
        print(f"{log_prefix} Need at least 2 experiments for pairwise comparison.")
        return None

    names = list(all_predictions.keys())
    disagreement_rows = []
    total_disagreement = 0.0

    for name_a, name_b in combinations(names, 2):
        pred_a = all_predictions[name_a]
        pred_b = all_predictions[name_b]

        disagreement = float(np.mean(pred_a.to_numpy() != pred_b.to_numpy()))
        disagreement_rows.append(
            {"exp_a": name_a, "exp_b": name_b, "disagreement_rate": disagreement}
        )
        total_disagreement += disagreement

    n_pairs = len(disagreement_rows)
    mean_disagreement = total_disagreement / n_pairs

    print(f"\n{log_prefix} Threshold = {threshold}")
    print(f"{log_prefix} Number of experiments: {len(names)}")
    print(f"{log_prefix} Number of pairs: {n_pairs}")
    print(f"{log_prefix} Mean pairwise disagreement rate: {mean_disagreement:.6f}")

    df_pairs = pd.DataFrame(disagreement_rows)
    pair_paths = [os.path.join(output_dir, f"pairwise_disagreement_{prediction_type}.csv")]
    for pair_path in pair_paths:
        df_pairs.to_csv(pair_path, index=False)
        print(f"{log_prefix} Pairwise details saved to: {pair_path}")

    df_summary = pd.DataFrame(
        [
            {
                "prediction_type": prediction_type,
                "threshold": threshold,
                "n_experiments": len(names),
                "n_pairs": n_pairs,
                "mean_disagreement_rate": mean_disagreement,
            }
        ]
    )
    print(f"{log_prefix} Summary prepared for combined output.")

    return df_summary.iloc[0].to_dict()


def task2_pairwise_disagreement(folders, threshold, output_dir):
    """
    Compute PDR for final table, ensemble and baseline predictions.
    """
    settings = [
        ("final_table", "final_table_predictions.csv"),
        ("ensemble", "ensemble_predictions.csv"),
        ("baseline", "baseline_predictions.csv"),
    ]

    # Keep only combined summary output to avoid duplicated information files.
    obsolete_summary_files = [
        "disagreement_summary.csv",
        "disagreement_summary_final_table.csv",
        "disagreement_summary_ensemble.csv",
        "disagreement_summary_baseline.csv",
    ]
    for filename in obsolete_summary_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"[Task2] Removed obsolete summary file: {path}")

    summary_rows = []

    for prediction_type, pred_filename in settings:
        print("\n" + "-" * 60)
        print(f"[Task2] Running PDR for {prediction_type} ({pred_filename})")
        print("-" * 60)
        metric = task2_pairwise_disagreement_single(
            folders=folders,
            threshold=threshold,
            output_dir=output_dir,
            prediction_type=prediction_type,
            pred_filename=pred_filename,
        )
        if metric is not None:
            summary_rows.append(metric)

    if summary_rows:
        combined_summary_path = os.path.join(output_dir, "disagreement_summary_all_models.csv")
        pd.DataFrame(summary_rows).to_csv(combined_summary_path, index=False)
        print(f"\n[Task2] Combined summary saved to: {combined_summary_path}")
    else:
        print("\n[Task2] No valid predictions found for final_table / ensemble / baseline.")


# ===================== Main =====================
if __name__ == "__main__":
    args = parse_args()
    if not np.isfinite(args.threshold):
        raise SystemExit("--threshold must be finite.")
    if not os.path.isdir(args.results_dir):
        raise SystemExit("Results directory not found. Run the experiment first; see README.md.")
    folders = get_experiment_folders(args.results_dir, args.seed_start, args.seed_end)
    if not folders:
        raise SystemExit("No experiment folders found for the selected seed range.")
    output_dir = create_run_output_dir(args.output_root, args.seed_start, args.seed_end)
    if args.seed_start is not None and args.seed_end is not None:
        expected_n = args.seed_end - args.seed_start + 1
        print(
            f"Using seed range {args.seed_start}-{args.seed_end} "
            f"(expected {expected_n} seeds), found {len(folders)} folders in {args.results_dir}"
        )
        print(f"Output directory: {output_dir}\n")
    else:
        print(f"Found {len(folders)} experiment folders in {args.results_dir}")
        print(f"Output directory: {output_dir}\n")

    print("=" * 60)
    print("Task 1: AUC Variance Comparison")
    print("=" * 60)
    task1_auc_variance_comparison(folders, output_dir)

    print("\n" + "=" * 60)
    print(f"Task 2: Pairwise Disagreement (threshold={args.threshold})")
    print("=" * 60)
    task2_pairwise_disagreement(folders, args.threshold, output_dir)
