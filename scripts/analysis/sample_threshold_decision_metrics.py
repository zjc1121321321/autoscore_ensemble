"""
Threshold-centered mortality analysis for exp_seedXXXX outputs.

This script scans a threshold range, reports method-level curves, and
selects a representative threshold t* with a predefined rule:
1) sensitivity >= SENSITIVITY_MIN
2) within candidates, balanced accuracy is maximized.

Sample-level details are then reported at t* only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===================== User Configuration =====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Directory that directly contains exp_seedXXXX folders.
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

# Seed filter (inclusive). Set both to None to use all detected seeds.
SEED_START: Optional[int] = None
SEED_END: Optional[int] = None

# Threshold set T = [THRESHOLD_START, THRESHOLD_END] with step THRESHOLD_STEP.
THRESHOLD_START = 30
THRESHOLD_END = 60
THRESHOLD_STEP = 1.0

# Representative-threshold selection rule B.
# First keep thresholds with sensitivity >= SENSITIVITY_MIN, then maximize BA.
SENSITIVITY_MIN = 0.8

# Cost-sensitive utility settings: TP/TN=+1, FP=-alpha, FN=-beta.
# beta must be > alpha > 0.
COST_ALPHA = 1.0
COST_BETA = 2.0

# Plot controls.
GENERATE_PLOTS = True
PLOT_DPI = 160

# Root output directory. Each run creates a new timestamped folder.
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")

# Optional hard limit for score sanity check.
# If abs(score) exceeds the limit in one seed file, that seed is skipped.
# Set to None to disable.
ENSEMBLE_SCORE_ABS_HARD_LIMIT: Optional[float] = 1000.0
FINAL_TABLE_SCORE_ABS_HARD_LIMIT: Optional[float] = 1000.0
BASELINE_SCORE_ABS_HARD_LIMIT: Optional[float] = 1000.0

# Baseline toggle.
ENABLE_BASELINE = True
BASELINE_REQUIRED = False
# ==============================================================


@dataclass
class SourceMatrix:
    source_name: str
    sample_ids: np.ndarray
    true_labels: np.ndarray
    score_matrix: np.ndarray
    score_columns: List[str]
    used_seeds: List[int]
    skipped_seeds: List[Dict[str, str]]


def validate_config() -> None:
    if not os.path.isdir(RESULTS_DIR):
        raise ValueError(f"RESULTS_DIR does not exist or is not a directory: {RESULTS_DIR}")

    if (SEED_START is None) != (SEED_END is None):
        raise ValueError("SEED_START and SEED_END must be both set or both None.")
    if SEED_START is not None and SEED_END is not None and SEED_START > SEED_END:
        raise ValueError("SEED_START must be <= SEED_END.")

    if THRESHOLD_STEP <= 0:
        raise ValueError("THRESHOLD_STEP must be > 0.")
    if THRESHOLD_START > THRESHOLD_END:
        raise ValueError("THRESHOLD_START must be <= THRESHOLD_END.")
    if not (0.0 <= SENSITIVITY_MIN <= 1.0):
        raise ValueError("SENSITIVITY_MIN must be in [0, 1].")
    if COST_ALPHA <= 0 or COST_BETA <= 0:
        raise ValueError("COST_ALPHA and COST_BETA must be > 0.")
    if not COST_BETA > COST_ALPHA:
        raise ValueError("COST_BETA must be strictly greater than COST_ALPHA.")
    if PLOT_DPI <= 0:
        raise ValueError("PLOT_DPI must be > 0.")

    for name, value in [
        ("ENSEMBLE_SCORE_ABS_HARD_LIMIT", ENSEMBLE_SCORE_ABS_HARD_LIMIT),
        ("FINAL_TABLE_SCORE_ABS_HARD_LIMIT", FINAL_TABLE_SCORE_ABS_HARD_LIMIT),
        ("BASELINE_SCORE_ABS_HARD_LIMIT", BASELINE_SCORE_ABS_HARD_LIMIT),
    ]:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 or None.")


def build_thresholds(start: float, end: float, step: float) -> np.ndarray:
    thresholds = []
    value = float(start)
    eps = abs(step) * 1e-9
    while value <= end + eps:
        thresholds.append(round(value, 10))
        value += step
    if not thresholds:
        raise ValueError("No thresholds generated. Please check threshold config.")
    return np.asarray(thresholds, dtype=float)


def get_experiment_folders(
    results_dir: str,
    seed_start: Optional[int] = None,
    seed_end: Optional[int] = None,
) -> List[Tuple[int, str, str]]:
    pattern = re.compile(r"^exp_seed(\d+)$")
    folders: List[Tuple[int, str, str]] = []

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
    return folders


def create_run_output_dir(
    output_root: str,
    seed_start: Optional[int],
    seed_end: Optional[int],
) -> str:
    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"seed{seed_start}-{seed_end}" if seed_start is not None else "seed_all"
    base_name = f"sample_threshold_metrics_{seed_tag}_{timestamp}"
    output_dir = os.path.join(output_root, base_name)

    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(output_root, f"{base_name}_{suffix}")
        suffix += 1

    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def _read_prediction_frame(exp_dir: str, source_name: str) -> pd.DataFrame:
    if source_name == "ensemble":
        pred_path = os.path.join(exp_dir, "ensemble_predictions.csv")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing file: {pred_path}")
        df = pd.read_csv(pred_path)

        if "ensemble_mean" in df.columns:
            score = pd.to_numeric(df["ensemble_mean"], errors="coerce")
        else:
            model_cols = [c for c in df.columns if c.startswith("model_")]
            if model_cols:
                score = df[model_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            elif "pred_score" in df.columns:
                score = pd.to_numeric(df["pred_score"], errors="coerce")
            else:
                raise ValueError(
                    "ensemble_predictions.csv needs one of: ensemble_mean, model_*, pred_score."
                )
    elif source_name in {"final_table", "baseline"}:
        filename = "final_table_predictions.csv" if source_name == "final_table" else "baseline_predictions.csv"
        pred_path = os.path.join(exp_dir, filename)
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing file: {pred_path}")
        df = pd.read_csv(pred_path)
        if "pred_score" not in df.columns:
            raise ValueError(f"{filename} must contain pred_score column.")
        score = pd.to_numeric(df["pred_score"], errors="coerce")
    else:
        raise ValueError(f"Unknown source_name: {source_name}")

    if "sample_id" in df.columns:
        sample_id = df["sample_id"]
    else:
        sample_id = pd.Series(np.arange(1, len(df) + 1), name="sample_id")

    if "true_label" not in df.columns:
        raise ValueError(f"{source_name} prediction file must contain true_label column.")
    true_label = pd.to_numeric(df["true_label"], errors="coerce")

    out = pd.DataFrame(
        {
            "sample_id": sample_id,
            "true_label": true_label,
            "score": score,
        }
    )
    out = out.drop_duplicates(subset=["sample_id"], keep="first")
    out = out.sort_values("sample_id")
    return out


def _coerce_binary_labels(labels: np.ndarray) -> np.ndarray:
    unique_vals = np.unique(labels[~np.isnan(labels)])
    if len(unique_vals) == 0:
        raise ValueError("No valid labels found.")
    if len(unique_vals) > 2:
        raise ValueError(f"Expected binary labels, found values: {unique_vals.tolist()}")

    if set(unique_vals.tolist()).issubset({0.0, 1.0}):
        return labels.astype(int)

    if len(unique_vals) == 2:
        high = float(unique_vals[1])
        mapped = np.where(labels == high, 1, 0)
        return mapped.astype(int)

    return np.zeros_like(labels, dtype=int)


def _get_score_abs_hard_limit(source_name: str) -> Optional[float]:
    if source_name == "ensemble":
        return ENSEMBLE_SCORE_ABS_HARD_LIMIT
    if source_name == "final_table":
        return FINAL_TABLE_SCORE_ABS_HARD_LIMIT
    if source_name == "baseline":
        return BASELINE_SCORE_ABS_HARD_LIMIT
    return None


def build_source_matrix(
    folders: List[Tuple[int, str, str]],
    source_name: str,
) -> SourceMatrix:
    merged_df: Optional[pd.DataFrame] = None
    score_columns: List[str] = []
    used_seeds: List[int] = []
    skipped_seeds: List[Dict[str, str]] = []
    score_abs_hard_limit = _get_score_abs_hard_limit(source_name)

    for seed, folder_name, folder_path in folders:
        try:
            frame = _read_prediction_frame(folder_path, source_name)
        except Exception as exc:
            skipped_seeds.append(
                {"seed": str(seed), "folder": folder_name, "reason": str(exc)}
            )
            continue

        if score_abs_hard_limit is not None:
            max_abs_score = pd.to_numeric(frame["score"], errors="coerce").abs().max()
            if pd.isna(max_abs_score):
                skipped_seeds.append(
                    {
                        "seed": str(seed),
                        "folder": folder_name,
                        "reason": "All score values are missing or non-numeric.",
                    }
                )
                continue
            if float(max_abs_score) > score_abs_hard_limit:
                skipped_seeds.append(
                    {
                        "seed": str(seed),
                        "folder": folder_name,
                        "reason": (
                            f"Score abs max {float(max_abs_score):.6f} exceeds hard limit "
                            f"{score_abs_hard_limit}."
                        ),
                    }
                )
                continue

        seed_col = f"seed_{seed}"
        frame = frame.rename(columns={"score": seed_col}).set_index("sample_id")

        if merged_df is None:
            merged_df = frame[["true_label", seed_col]].copy()
            score_columns.append(seed_col)
            used_seeds.append(seed)
            continue

        common_ids = merged_df.index.intersection(frame.index)
        if common_ids.empty:
            skipped_seeds.append(
                {
                    "seed": str(seed),
                    "folder": folder_name,
                    "reason": "No overlapping sample_id with previous seeds.",
                }
            )
            continue

        merged_df = merged_df.loc[common_ids].sort_index()
        frame = frame.loc[common_ids].sort_index()

        mismatch = merged_df["true_label"].to_numpy() != frame["true_label"].to_numpy()
        if np.any(mismatch):
            raise ValueError(
                f"{source_name}: true_label mismatch in seed {seed} for {int(np.sum(mismatch))} samples."
            )

        merged_df[seed_col] = frame[seed_col].to_numpy()
        score_columns.append(seed_col)
        used_seeds.append(seed)

    if merged_df is None or not score_columns:
        raise ValueError(f"No valid seeds found for source={source_name}.")

    keep_cols = ["true_label"] + score_columns
    merged_df = merged_df.dropna(subset=keep_cols)

    sample_ids = merged_df.index.to_numpy()
    true_labels_raw = merged_df["true_label"].to_numpy(dtype=float)
    true_labels = _coerce_binary_labels(true_labels_raw)
    score_matrix = merged_df[score_columns].to_numpy(dtype=float)

    return SourceMatrix(
        source_name=source_name,
        sample_ids=sample_ids,
        true_labels=true_labels,
        score_matrix=score_matrix,
        score_columns=score_columns,
        used_seeds=used_seeds,
        skipped_seeds=skipped_seeds,
    )


def _source_order_value(source_name: str) -> int:
    order = {"ensemble": 0, "final_table": 1, "baseline": 2}
    return order.get(source_name, 999)


def _sorted_sources(source_names: List[str]) -> List[str]:
    return sorted(source_names, key=_source_order_value)


def _pretty_source_name(source_name: str) -> str:
    names = {
        "ensemble": "Ensemble",
        "final_table": "Final Table",
        "baseline": "Baseline",
    }
    return names.get(source_name, source_name)


def compute_sample_metrics_at_threshold(
    score_matrix: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
    cost_alpha: float,
    cost_beta: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    n_samples, n_models = score_matrix.shape
    if n_samples == 0 or n_models == 0:
        raise ValueError("score_matrix must be non-empty.")

    label_2d = true_labels.reshape(-1, 1)
    pred_pos = score_matrix >= threshold

    # Keep definitions 1/2/3 identical to the previous implementation.
    k_pos = pred_pos.sum(axis=1)
    k_neg = n_models - k_pos
    major = np.maximum(k_pos, k_neg)
    minor = np.minimum(k_pos, k_neg)
    split_strength = np.divide(
        minor,
        major,
        out=np.zeros(n_samples, dtype=float),
        where=major > 0,
    )
    correctness = (pred_pos == label_2d).mean(axis=1)
    r_signed = (2.0 * correctness - 1.0) * (1.0 - split_strength)

    utility_matrix = np.where(
        label_2d == 1,
        np.where(pred_pos, 1.0, -cost_beta),
        np.where(pred_pos, -cost_alpha, 1.0),
    )
    cost_utility = utility_matrix.mean(axis=1)
    cost_utility_normalized = cost_utility / max(1.0, cost_alpha, cost_beta)
    r_cost = cost_utility_normalized * (1.0 - split_strength)

    pos_mask = true_labels == 1
    neg_mask = true_labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative classes are required for threshold metrics.")

    tp = float(pred_pos[pos_mask, :].sum())
    fn = float((~pred_pos[pos_mask, :]).sum())
    fp = float(pred_pos[neg_mask, :].sum())
    tn = float((~pred_pos[neg_mask, :]).sum())

    sensitivity = tp / (n_models * n_pos)
    specificity = tn / (n_models * n_neg)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    method_cost_utility = (tp + tn - cost_alpha * fp - cost_beta * fn) / (n_models * n_samples)

    sample_metrics = {
        "split_strength": split_strength,
        "correctness": correctness,
        "r_signed": r_signed,
        "cost_utility": cost_utility,
        "cost_utility_normalized": cost_utility_normalized,
        "r_cost": r_cost,
    }
    aggregate_metrics = {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "cost_utility": float(method_cost_utility),
        "mean_positive_fraction": float(np.mean(k_pos / n_models)),
        "mean_split_strength": float(np.mean(split_strength)),
        "mean_correctness": float(np.mean(correctness)),
        "mean_r_signed": float(np.mean(r_signed)),
        "mean_r_cost": float(np.mean(r_cost)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_samples": int(n_samples),
        "n_models": int(n_models),
        "n_positive_samples": int(n_pos),
        "n_negative_samples": int(n_neg),
    }
    return sample_metrics, aggregate_metrics


def compute_threshold_curve_summary(
    score_matrix: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray,
    cost_alpha: float,
    cost_beta: float,
) -> pd.DataFrame:
    threshold_rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        _, aggregate = compute_sample_metrics_at_threshold(
            score_matrix=score_matrix,
            true_labels=true_labels,
            threshold=float(threshold),
            cost_alpha=cost_alpha,
            cost_beta=cost_beta,
        )
        threshold_rows.append(aggregate)

    return pd.DataFrame(threshold_rows).sort_values("threshold")


def select_representative_threshold(
    threshold_summary: pd.DataFrame,
    sensitivity_min: float,
) -> Dict[str, float]:
    if threshold_summary.empty:
        raise ValueError("threshold_summary is empty.")

    rule_candidates = threshold_summary[threshold_summary["sensitivity"] >= sensitivity_min].copy()
    n_rule_candidates = int(rule_candidates.shape[0])

    if n_rule_candidates > 0:
        selection_pool = rule_candidates
        selection_status = "rule_b"
        sort_by = ["balanced_accuracy", "sensitivity", "cost_utility", "threshold"]
        ascending = [False, False, False, True]
    else:
        selection_pool = threshold_summary.copy()
        selection_status = "fallback_max_sensitivity"
        sort_by = ["sensitivity", "balanced_accuracy", "cost_utility", "threshold"]
        ascending = [False, False, False, True]

    best = (
        selection_pool.sort_values(
            by=sort_by,
            ascending=ascending,
            kind="mergesort",
        )
        .iloc[0]
        .to_dict()
    )
    best["selection_status"] = selection_status
    best["sensitivity_min"] = float(sensitivity_min)
    best["n_rule_candidates"] = n_rule_candidates
    best["n_selection_pool"] = int(selection_pool.shape[0])
    best["constraint_satisfied"] = int(float(best["sensitivity"]) >= sensitivity_min)
    return best


def build_representative_sample_table(
    source_matrix: SourceMatrix,
    representative_threshold: float,
    cost_alpha: float,
    cost_beta: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sample_metrics, aggregate = compute_sample_metrics_at_threshold(
        score_matrix=source_matrix.score_matrix,
        true_labels=source_matrix.true_labels,
        threshold=float(representative_threshold),
        cost_alpha=cost_alpha,
        cost_beta=cost_beta,
    )

    sample_df = pd.DataFrame(
        {
            "sample_id": source_matrix.sample_ids,
            "true_label": source_matrix.true_labels,
            "n_models": source_matrix.score_matrix.shape[1],
            "representative_threshold": float(representative_threshold),
            "split_strength": sample_metrics["split_strength"],
            "correctness": sample_metrics["correctness"],
            "r_signed": sample_metrics["r_signed"],
            "cost_utility": sample_metrics["cost_utility"],
            "cost_utility_normalized": sample_metrics["cost_utility_normalized"],
            "r_cost": sample_metrics["r_cost"],
        }
    )
    return sample_df, aggregate


def merge_prefixed_sample_tables(source_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged_df: Optional[pd.DataFrame] = None

    for source_name in _sorted_sources(list(source_tables.keys())):
        df = source_tables[source_name]
        prefixed_df = df.rename(
            columns={c: f"{source_name}_{c}" for c in df.columns if c not in {"sample_id", "true_label"}}
        )
        if merged_df is None:
            merged_df = prefixed_df
        else:
            merged_df = pd.merge(merged_df, prefixed_df, on=["sample_id", "true_label"], how="outer")

    if merged_df is None:
        return pd.DataFrame(columns=["sample_id", "true_label"])
    return merged_df.sort_values("sample_id")


def plot_threshold_curves(
    source_threshold_tables: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    if not source_threshold_tables:
        raise ValueError("No threshold tables provided for plotting.")

    metrics = [
        ("sensitivity", "Sensitivity (Death Recall)", (0.0, 1.0)),
        ("balanced_accuracy", "Balanced Accuracy", (0.0, 1.0)),
        ("cost_utility", f"Cost Utility (alpha={COST_ALPHA}, beta={COST_BETA})", None),
    ]
    colors = {
        "ensemble": "#1f77b4",
        "final_table": "#ff7f0e",
        "baseline": "#2ca02c",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for source_name in _sorted_sources(list(source_threshold_tables.keys())):
        df = source_threshold_tables[source_name].sort_values("threshold")
        x = df["threshold"].to_numpy(dtype=float)
        for ax, (metric, title, ylim) in zip(axes, metrics):
            y = df[metric].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=_pretty_source_name(source_name),
                color=colors.get(source_name),
            )
            ax.set_title(title)
            ax.set_xlabel("Threshold")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25)

    axes[0].axhline(SENSITIVITY_MIN, color="red", linestyle="--", linewidth=1.2, label="Sensitivity Min")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Threshold Curves Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_representative_threshold_comparison(
    representative_summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    if representative_summary_df.empty:
        raise ValueError("No representative-threshold summary found for plotting.")

    plot_df = representative_summary_df.copy()
    plot_df = plot_df.sort_values("source_name", key=lambda s: s.map(_source_order_value))
    methods = plot_df["source_name"].tolist()
    xtick_labels = [
        f"{_pretty_source_name(source)}\nt*={threshold:.1f}"
        for source, threshold in zip(methods, plot_df["representative_threshold"].to_numpy(dtype=float))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(methods))

    rate_specs = [
        ("sensitivity", "Sensitivity", "#4c78a8"),
        ("specificity", "Specificity", "#72b7b2"),
        ("balanced_accuracy", "Balanced Accuracy", "#f58518"),
    ]
    width_left = 0.24
    for idx, (col, label, color) in enumerate(rate_specs):
        offset = (idx - 1) * width_left
        axes[0].bar(
            x + offset,
            plot_df[col].to_numpy(dtype=float),
            width=width_left,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xtick_labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Rates At Representative Threshold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)

    utility_specs = [
        ("cost_utility", "Cost Utility", "#54a24b"),
        ("mean_r_cost", "Mean R_cost", "#e45756"),
    ]
    width_right = 0.34
    for idx, (col, label, color) in enumerate(utility_specs):
        offset = (idx - 0.5) * width_right
        axes[1].bar(
            x + offset,
            plot_df[col].to_numpy(dtype=float),
            width=width_right,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xtick_labels)
    axes[1].set_title("Utility/Cost-Reliability At Representative Threshold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.suptitle("Representative Threshold Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_sample_reliability_distributions(
    source_sample_tables: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    if not source_sample_tables:
        raise ValueError("No sample tables provided for plotting.")

    ordered_sources = _sorted_sources(list(source_sample_tables.keys()))
    n_rows = len(ordered_sources)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.5, 3.3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    colors = {
        "ensemble": "#1f77b4",
        "final_table": "#ff7f0e",
        "baseline": "#2ca02c",
    }
    bins = np.linspace(-1.0, 1.0, 31)

    for row_idx, source_name in enumerate(ordered_sources):
        df = source_sample_tables[source_name]
        color = colors.get(source_name, "#4c78a8")
        ax_signed = axes[row_idx, 0]
        ax_cost = axes[row_idx, 1]

        r_signed = df["r_signed"].to_numpy(dtype=float)
        r_cost = df["r_cost"].to_numpy(dtype=float)

        ax_signed.hist(r_signed, bins=bins, color=color, alpha=0.85)
        ax_signed.axvline(np.mean(r_signed), color="black", linestyle="--", linewidth=1.1)
        ax_signed.set_xlim(-1.0, 1.0)
        ax_signed.grid(alpha=0.2)

        ax_cost.hist(r_cost, bins=bins, color=color, alpha=0.85)
        ax_cost.axvline(np.mean(r_cost), color="black", linestyle="--", linewidth=1.1)
        ax_cost.set_xlim(-1.0, 1.0)
        ax_cost.grid(alpha=0.2)

        ax_signed.set_ylabel(_pretty_source_name(source_name))

        if row_idx == 0:
            ax_signed.set_title("R_signed Distribution")
            ax_cost.set_title("R_cost Distribution")

    axes[-1, 0].set_xlabel("R_signed")
    axes[-1, 1].set_xlabel("R_cost")
    fig.suptitle("Sample Reliability Distributions At Representative Threshold", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    validate_config()
    thresholds = build_thresholds(THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP)
    folders = get_experiment_folders(RESULTS_DIR, SEED_START, SEED_END)
    if not folders:
        raise ValueError(f"No exp_seedXXXX folders found in: {RESULTS_DIR}")

    output_dir = create_run_output_dir(OUTPUT_ROOT_DIR, SEED_START, SEED_END)

    print(f"Results directory: {RESULTS_DIR}")
    if SEED_START is not None:
        print(f"Seed range: {SEED_START}-{SEED_END}")
    else:
        print(f"Seed range: all detected ({len(folders)} folders)")
    print(
        f"Thresholds: start={THRESHOLD_START}, end={THRESHOLD_END}, "
        f"step={THRESHOLD_STEP}, count={len(thresholds)}"
    )
    print(
        f"Rule B: sensitivity >= {SENSITIVITY_MIN}, then maximize balanced accuracy "
        f"(cost alpha={COST_ALPHA}, beta={COST_BETA})"
    )
    print(f"Output directory: {output_dir}")

    source_order = ["ensemble", "final_table"]
    if ENABLE_BASELINE:
        source_order.append("baseline")

    source_matrices: Dict[str, SourceMatrix] = {}
    for source_name in source_order:
        try:
            source_matrices[source_name] = build_source_matrix(folders, source_name)
        except Exception as exc:
            if source_name == "baseline" and not BASELINE_REQUIRED:
                print(f"[WARN] Baseline not available, continue without baseline: {exc}")
                continue
            raise

    if "ensemble" not in source_matrices or "final_table" not in source_matrices:
        raise ValueError("Both ensemble and final_table must be available.")

    source_threshold_tables: Dict[str, pd.DataFrame] = {}
    source_representative_sample_tables: Dict[str, pd.DataFrame] = {}
    representative_rows: List[Dict[str, float]] = []

    for source_name in _sorted_sources(list(source_matrices.keys())):
        matrix = source_matrices[source_name]
        threshold_df = compute_threshold_curve_summary(
            score_matrix=matrix.score_matrix,
            true_labels=matrix.true_labels,
            thresholds=thresholds,
            cost_alpha=COST_ALPHA,
            cost_beta=COST_BETA,
        )
        source_threshold_tables[source_name] = threshold_df

        selected = select_representative_threshold(threshold_df, SENSITIVITY_MIN)
        representative_threshold = float(selected["threshold"])

        sample_df, sample_aggregate = build_representative_sample_table(
            source_matrix=matrix,
            representative_threshold=representative_threshold,
            cost_alpha=COST_ALPHA,
            cost_beta=COST_BETA,
        )
        source_representative_sample_tables[source_name] = sample_df

        representative_rows.append(
            {
                "source_name": source_name,
                "representative_threshold": representative_threshold,
                "selection_status": str(selected["selection_status"]),
                "sensitivity_min": float(selected["sensitivity_min"]),
                "constraint_satisfied": int(selected["constraint_satisfied"]),
                "n_rule_candidates": int(selected["n_rule_candidates"]),
                "n_selection_pool": int(selected["n_selection_pool"]),
                "sensitivity": float(selected["sensitivity"]),
                "specificity": float(selected["specificity"]),
                "balanced_accuracy": float(selected["balanced_accuracy"]),
                "cost_utility": float(selected["cost_utility"]),
                "mean_positive_fraction": float(selected["mean_positive_fraction"]),
                "mean_split_strength": float(sample_aggregate["mean_split_strength"]),
                "mean_correctness": float(sample_aggregate["mean_correctness"]),
                "mean_r_signed": float(sample_aggregate["mean_r_signed"]),
                "mean_r_cost": float(sample_aggregate["mean_r_cost"]),
                "tp": int(sample_aggregate["tp"]),
                "fp": int(sample_aggregate["fp"]),
                "tn": int(sample_aggregate["tn"]),
                "fn": int(sample_aggregate["fn"]),
                "n_samples": int(sample_aggregate["n_samples"]),
                "n_models": int(sample_aggregate["n_models"]),
                "n_positive_samples": int(sample_aggregate["n_positive_samples"]),
                "n_negative_samples": int(sample_aggregate["n_negative_samples"]),
            }
        )

    representative_summary_df = pd.DataFrame(representative_rows)
    representative_summary_df = representative_summary_df.sort_values(
        "source_name", key=lambda s: s.map(_source_order_value)
    )

    merged_representative_sample_df = merge_prefixed_sample_tables(source_representative_sample_tables)

    seed_manifest_path = os.path.join(output_dir, "seed_manifest.csv")
    config_path = os.path.join(output_dir, "run_config.json")

    threshold_curve_paths = {}
    representative_summary_paths = {}
    representative_sample_paths = {}
    for source_name in _sorted_sources(list(source_matrices.keys())):
        threshold_curve_path = os.path.join(output_dir, f"{source_name}_threshold_curve_summary.csv")
        representative_summary_path = os.path.join(
            output_dir, f"{source_name}_representative_threshold_summary.csv"
        )
        representative_sample_path = os.path.join(
            output_dir, f"{source_name}_representative_sample_metrics.csv"
        )

        threshold_curve_paths[source_name] = threshold_curve_path
        representative_summary_paths[source_name] = representative_summary_path
        representative_sample_paths[source_name] = representative_sample_path

        source_threshold_tables[source_name].to_csv(threshold_curve_path, index=False)
        representative_summary_df[
            representative_summary_df["source_name"] == source_name
        ].to_csv(representative_summary_path, index=False)
        source_representative_sample_tables[source_name].to_csv(representative_sample_path, index=False)

    representative_summary_all_path = os.path.join(
        output_dir, "representative_threshold_summary_all_sources.csv"
    )
    merged_representative_path = os.path.join(output_dir, "representative_sample_metrics_merged.csv")
    representative_summary_df.to_csv(representative_summary_all_path, index=False)
    merged_representative_sample_df.to_csv(merged_representative_path, index=False)

    seed_manifest_rows = []
    for seed, folder_name, _ in folders:
        row = {"seed": seed, "folder": folder_name}
        for source_name, matrix in source_matrices.items():
            row[f"used_in_{source_name}"] = int(seed in matrix.used_seeds)
        seed_manifest_rows.append(row)
    pd.DataFrame(seed_manifest_rows).to_csv(seed_manifest_path, index=False)

    plot_paths: Dict[str, str] = {}
    if GENERATE_PLOTS:
        threshold_curves_plot_path = os.path.join(output_dir, "threshold_curves_overview.png")
        representative_plot_path = os.path.join(output_dir, "representative_threshold_comparison.png")
        sample_plot_path = os.path.join(output_dir, "sample_reliability_at_tstar.png")

        try:
            plot_threshold_curves(source_threshold_tables, threshold_curves_plot_path)
            plot_paths["threshold_curves"] = threshold_curves_plot_path
        except Exception as exc:
            print(f"[WARN] Threshold curve plot generation failed: {exc}")

        try:
            plot_representative_threshold_comparison(representative_summary_df, representative_plot_path)
            plot_paths["representative_threshold_comparison"] = representative_plot_path
        except Exception as exc:
            print(f"[WARN] Representative threshold comparison plot generation failed: {exc}")

        try:
            plot_sample_reliability_distributions(source_representative_sample_tables, sample_plot_path)
            plot_paths["sample_reliability"] = sample_plot_path
        except Exception as exc:
            print(f"[WARN] Sample reliability plot generation failed: {exc}")

    representative_by_source = {
        row["source_name"]: row for row in representative_summary_df.to_dict(orient="records")
    }
    run_config = {
        "results_dir": os.path.abspath(RESULTS_DIR),
        "seed_start": SEED_START,
        "seed_end": SEED_END,
        "n_detected_seed_folders": len(folders),
        "threshold_start": THRESHOLD_START,
        "threshold_end": THRESHOLD_END,
        "threshold_step": THRESHOLD_STEP,
        "threshold_count": int(len(thresholds)),
        "thresholds": thresholds.tolist(),
        "sensitivity_min": SENSITIVITY_MIN,
        "cost_alpha": COST_ALPHA,
        "cost_beta": COST_BETA,
        "generate_plots": GENERATE_PLOTS,
        "plot_dpi": PLOT_DPI,
        "ensemble_score_abs_hard_limit": ENSEMBLE_SCORE_ABS_HARD_LIMIT,
        "final_table_score_abs_hard_limit": FINAL_TABLE_SCORE_ABS_HARD_LIMIT,
        "baseline_score_abs_hard_limit": BASELINE_SCORE_ABS_HARD_LIMIT,
        "enable_baseline": ENABLE_BASELINE,
        "baseline_required": BASELINE_REQUIRED,
        "output_dir": os.path.abspath(output_dir),
        "plot_paths": plot_paths,
        "sources": {},
        "generated_at": datetime.now().isoformat(),
    }
    for source_name, matrix in source_matrices.items():
        source_row = representative_by_source.get(source_name, {})
        run_config["sources"][source_name] = {
            "n_used_seeds": len(matrix.used_seeds),
            "used_seeds": matrix.used_seeds,
            "n_samples": int(matrix.score_matrix.shape[0]),
            "n_models": int(matrix.score_matrix.shape[1]),
            "skipped_seeds": matrix.skipped_seeds,
            "representative_threshold": source_row.get("representative_threshold"),
            "selection_status": source_row.get("selection_status"),
            "constraint_satisfied": source_row.get("constraint_satisfied"),
        }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    for source_name in _sorted_sources(list(source_matrices.keys())):
        print(f"- {threshold_curve_paths[source_name]}")
        print(f"- {representative_summary_paths[source_name]}")
        print(f"- {representative_sample_paths[source_name]}")
    print(f"- {representative_summary_all_path}")
    print(f"- {merged_representative_path}")
    print(f"- {seed_manifest_path}")
    print(f"- {config_path}")
    for _, plot_path in sorted(plot_paths.items()):
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
