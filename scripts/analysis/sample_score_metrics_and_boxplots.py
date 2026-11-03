"""
Score-only analysis for exp_seedXXXX outputs.

This script does NOT compute threshold/decision metrics.
It computes sample-level score statistics and focuses on:
1) overall method-level stability comparison
2) consensus hard-sample analysis across methods.
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

# Root output directory. Each run creates a new timestamped folder.
OUTPUT_ROOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")

# Scatter plot settings.
# Each sample is one point:
#   x-axis = central tendency metric
#   y-axis = variability metric
SCATTER_X_METRIC = "score_mean"   # options: score_mean, score_median
SCATTER_Y_METRIC = "score_iqr"    # options: score_iqr, score_std, score_range
ANNOTATE_TOP_VOLATILE_N = 20      # annotate top-N by y metric; set 0 to disable
GENERATE_SCATTER = False

# Main outputs.
TOP_K_CONSENSUS_HARD = 30
GENERATE_OVERALL_COMPARISON = True
GENERATE_CONSENSUS_HEATMAP = True

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

    if GENERATE_SCATTER:
        if SCATTER_X_METRIC not in {"score_mean", "score_median"}:
            raise ValueError("SCATTER_X_METRIC must be one of: score_mean, score_median.")
        if SCATTER_Y_METRIC not in {"score_iqr", "score_std", "score_range"}:
            raise ValueError("SCATTER_Y_METRIC must be one of: score_iqr, score_std, score_range.")
        if ANNOTATE_TOP_VOLATILE_N < 0:
            raise ValueError("ANNOTATE_TOP_VOLATILE_N must be >= 0.")
    if TOP_K_CONSENSUS_HARD <= 0:
        raise ValueError("TOP_K_CONSENSUS_HARD must be > 0.")

    for name, value in [
        ("ENSEMBLE_SCORE_ABS_HARD_LIMIT", ENSEMBLE_SCORE_ABS_HARD_LIMIT),
        ("FINAL_TABLE_SCORE_ABS_HARD_LIMIT", FINAL_TABLE_SCORE_ABS_HARD_LIMIT),
        ("BASELINE_SCORE_ABS_HARD_LIMIT", BASELINE_SCORE_ABS_HARD_LIMIT),
    ]:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0 or None.")


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
    base_name = f"sample_score_metrics_{seed_tag}_{timestamp}"
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


def compute_boxplot_stats(score_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    q1 = np.percentile(score_matrix, 25, axis=1)
    median = np.percentile(score_matrix, 50, axis=1)
    q3 = np.percentile(score_matrix, 75, axis=1)
    score_min = np.min(score_matrix, axis=1)
    score_max = np.max(score_matrix, axis=1)
    iqr = q3 - q1
    whisker_low = np.maximum(score_min, q1 - 1.5 * iqr)
    whisker_high = np.minimum(score_max, q3 + 1.5 * iqr)

    return {
        "mean": np.mean(score_matrix, axis=1),
        "std": np.std(score_matrix, axis=1, ddof=0),
        "min": score_min,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": score_max,
        "iqr": iqr,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


def build_score_metric_table(source_matrix: SourceMatrix) -> pd.DataFrame:
    stats = compute_boxplot_stats(source_matrix.score_matrix)
    score_range = stats["max"] - stats["min"]
    return pd.DataFrame(
        {
            "sample_id": source_matrix.sample_ids,
            "true_label": source_matrix.true_labels,
            "n_models": source_matrix.score_matrix.shape[1],
            "score_mean": stats["mean"],
            "score_std": stats["std"],
            "score_min": stats["min"],
            "score_q1": stats["q1"],
            "score_median": stats["median"],
            "score_q3": stats["q3"],
            "score_max": stats["max"],
            "score_iqr": stats["iqr"],
            "score_range": score_range,
            "score_whisker_low": stats["whisker_low"],
            "score_whisker_high": stats["whisker_high"],
        }
    )


def merge_score_tables(source_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged_df: Optional[pd.DataFrame] = None

    for source_name in _sorted_sources(list(source_tables.keys())):
        df = source_tables[source_name]
        pref = df.rename(
            columns={c: f"{source_name}_{c}" for c in df.columns if c not in {"sample_id", "true_label"}}
        )
        if merged_df is None:
            merged_df = pref
        else:
            merged_df = pd.merge(merged_df, pref, on=["sample_id", "true_label"], how="outer")

    if merged_df is None:
        return pd.DataFrame(columns=["sample_id", "true_label"])
    return merged_df.sort_values("sample_id")


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


def _source_color(source_name: str) -> str:
    colors = {
        "ensemble": "#4E79A7",
        "final_table": "#F28E2B",
        "baseline": "#59A14F",
    }
    return colors.get(source_name, "#7f7f7f")


def add_difficulty_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range_percentile"] = out["score_range"].rank(pct=True)
    out["std_percentile"] = out["score_std"].rank(pct=True)
    out["difficulty"] = np.sqrt(out["range_percentile"] * out["std_percentile"])
    return out


def build_method_score_summary(source_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for source_name in _sorted_sources(list(source_tables.keys())):
        df = source_tables[source_name]
        row: Dict[str, float] = {
            "source_name": source_name,
            "n_samples": int(df.shape[0]),
            "n_models": int(df["n_models"].iloc[0]) if not df.empty else 0,
        }
        for metric in ["score_range", "score_std", "score_iqr", "difficulty"]:
            values = df[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q75"] = float(np.percentile(values, 75))
            row[f"{metric}_q90"] = float(np.percentile(values, 90))
        row["hard_sample_rate_80"] = float(np.mean(df["difficulty"] >= 0.80))
        rows.append(row)
    return pd.DataFrame(rows)


def build_consensus_hard_samples_table(
    source_tables: Dict[str, pd.DataFrame],
    top_k: int,
) -> pd.DataFrame:
    merged_df: Optional[pd.DataFrame] = None
    all_sources = ["ensemble", "final_table", "baseline"]

    for source_name in all_sources:
        if source_name not in source_tables:
            continue
        df = source_tables[source_name][
            ["sample_id", "true_label", "score_range", "score_std", "difficulty"]
        ].copy()
        df = df.rename(
            columns={
                "score_range": f"{source_name}_score_range",
                "score_std": f"{source_name}_score_std",
                "difficulty": f"{source_name}_difficulty",
            }
        )
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=["sample_id", "true_label"], how="outer")

    if merged_df is None:
        return pd.DataFrame(columns=["sample_id", "true_label"])

    for source_name in all_sources:
        for metric in ["score_range", "score_std", "difficulty"]:
            col = f"{source_name}_{metric}"
            if col not in merged_df.columns:
                merged_df[col] = np.nan

    available_difficulty_cols = [
        f"{source_name}_difficulty"
        for source_name in all_sources
        if source_name in source_tables
    ]
    merged_df["consensus_mean_difficulty"] = merged_df[available_difficulty_cols].mean(axis=1, skipna=True)
    merged_df["consensus_min_difficulty"] = merged_df[available_difficulty_cols].min(axis=1, skipna=True)

    merged_df = merged_df.sort_values(
        ["consensus_min_difficulty", "consensus_mean_difficulty", "sample_id"],
        ascending=[False, False, True],
    )
    top_k = min(int(top_k), int(merged_df.shape[0]))
    merged_df = merged_df.head(top_k).reset_index(drop=True)
    return merged_df


def plot_overall_method_comparison(
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    if summary_df.empty:
        raise ValueError("method summary table is empty.")

    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values("source_name", key=lambda s: s.map(_source_order_value))
    methods = plot_df["source_name"].tolist()
    labels = [_pretty_source_name(s) for s in methods]
    colors = [_source_color(s) for s in methods]
    x = np.arange(len(methods))

    panel_metrics = [
        ("score_range_mean", "A. Score Range Mean"),
        ("score_std_mean", "B. Score Std Mean"),
        ("score_iqr_mean", "C. Score IQR Mean"),
        ("difficulty_mean", "D. Difficulty Mean"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    for ax, (metric_col, panel_title) in zip(axes_flat, panel_metrics):
        values = plot_df[metric_col].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.25)
        ylim_top = float(max(values) * 1.20 if len(values) > 0 else 1.0)
        if ylim_top <= 0:
            ylim_top = 1.0
        ax.set_ylim(0, ylim_top)
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + ylim_top * 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Overall Score Stability Comparison", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_consensus_hard_samples_grouped_points(
    consensus_df: pd.DataFrame,
    source_tables: Dict[str, pd.DataFrame],
    output_path: str,
    top_k: int,
) -> None:
    if consensus_df.empty:
        raise ValueError("consensus hard-sample table is empty.")

    source_order = [s for s in ["ensemble", "final_table", "baseline"] if s in source_tables]
    difficulty_cols = [f"{s}_difficulty" for s in source_order]
    row_labels = [
        f"{int(row.sample_id)} (y={int(row.true_label)})"
        for row in consensus_df[["sample_id", "true_label"]].itertuples(index=False)
    ]
    y_pos = np.arange(len(row_labels))

    fig_h = max(6.5, 0.40 * min(top_k, len(row_labels)) + 2.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    marker_map = {
        "ensemble": "o",
        "final_table": "s",
        "baseline": "^",
    }

    for source_name, diff_col in zip(source_order, difficulty_cols):
        values = consensus_df[diff_col].to_numpy(dtype=float)
        valid = ~np.isnan(values)
        ax.scatter(
            values[valid],
            y_pos[valid],
            s=60,
            marker=marker_map.get(source_name, "o"),
            color=_source_color(source_name),
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            label=_pretty_source_name(source_name),
            zorder=3,
        )

    # Light guideline per sample row for easier method-to-method visual comparison.
    for idx in range(len(row_labels)):
        row_vals = consensus_df.iloc[idx][difficulty_cols].to_numpy(dtype=float)
        row_vals = row_vals[~np.isnan(row_vals)]
        if row_vals.size >= 2:
            ax.hlines(
                y=idx,
                xmin=float(np.min(row_vals)),
                xmax=float(np.max(row_vals)),
                color="#bdbdbd",
                linewidth=0.9,
                alpha=0.8,
                zorder=2,
            )

    # Dynamic x-range based on observed difficulty values in the selected Top-K rows.
    all_vals = consensus_df[difficulty_cols].to_numpy(dtype=float).ravel()
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size > 0:
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        span = x_max - x_min
        if span <= 1e-12:
            pad = max(0.02, 0.05 * max(1e-6, abs(x_max)))
        else:
            pad = max(0.01, span * 0.08)
        left = max(0.0, x_min - pad)
        right = min(1.0, x_max + pad)
        if right - left < 0.05:
            center = 0.5 * (left + right)
            left = max(0.0, center - 0.025)
            right = min(1.0, center + 0.025)
        ax.set_xlim(left, right)
    else:
        ax.set_xlim(0.0, 1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top-K Consensus Hard Samples")
    ax.set_xlabel("Difficulty d(i,m)")
    ax.set_ylabel("Sample ID (Label)")
    ax.grid(axis="x", alpha=0.25, zorder=1)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_top_volatile_table(
    df: pd.DataFrame,
    y_metric: str,
    top_n: int,
) -> pd.DataFrame:
    cols = ["sample_id", "true_label", "score_mean", "score_median", "score_iqr", "score_std", "score_range"]
    work_df = df.dropna(subset=[y_metric]).copy()
    work_df = work_df.sort_values([y_metric, "sample_id"], ascending=[False, True]).head(top_n)
    out = work_df[[c for c in cols if c in work_df.columns]].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def _scatter_by_label(ax: plt.Axes, df: pd.DataFrame, x_metric: str, y_metric: str) -> None:
    color_map = {0: "#4E79A7", 1: "#E15759"}
    label_values = sorted(pd.unique(df["true_label"]))

    for lbl in label_values:
        sub = df[df["true_label"] == lbl]
        color = color_map.get(int(lbl), "#7f7f7f") if pd.notna(lbl) else "#7f7f7f"
        ax.scatter(
            sub[x_metric],
            sub[y_metric],
            s=18,
            alpha=0.70,
            c=color,
            edgecolors="none",
            label=f"label={int(lbl)}" if pd.notna(lbl) else "label=NA",
        )


def plot_mean_variability_scatter(
    source_tables: Dict[str, pd.DataFrame],
    x_metric: str,
    y_metric: str,
    annotate_top_n: int,
    output_path: str,
) -> None:
    if not source_tables:
        raise ValueError("No source tables provided for scatter plotting.")

    required_cols = {"sample_id", "true_label", x_metric, y_metric}
    plot_tables: Dict[str, pd.DataFrame] = {}
    for source_name, df in source_tables.items():
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{source_name} metrics missing columns: {missing}")
        plot_tables[source_name] = df.dropna(subset=[x_metric, y_metric]).copy()

    n_panels = len(plot_tables)
    fig_w = max(7 * n_panels, 10)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 6))
    if n_panels == 1:
        axes = [axes]

    def draw_one(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
        _scatter_by_label(ax, df, x_metric, y_metric)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="best", fontsize=8, frameon=False)

        if annotate_top_n > 0 and not df.empty:
            top_df = df.sort_values(y_metric, ascending=False).head(annotate_top_n)
            for row in top_df.itertuples(index=False):
                ax.annotate(
                    str(row.sample_id),
                    (float(getattr(row, x_metric)), float(getattr(row, y_metric))),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                    alpha=0.85,
                )

    title_map = {
        "ensemble": "Ensemble",
        "final_table": "Final Table",
        "baseline": "Baseline",
    }
    for ax, source_name in zip(axes, sorted(plot_tables.keys())):
        pretty_name = title_map.get(source_name, source_name)
        draw_one(ax, plot_tables[source_name], f"{pretty_name}: Sample Mean vs Variability")

    fig.suptitle(f"Sample-Level Scatter ({x_metric} vs {y_metric})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    validate_config()
    folders = get_experiment_folders(RESULTS_DIR, SEED_START, SEED_END)
    if not folders:
        raise ValueError(f"No exp_seedXXXX folders found in: {RESULTS_DIR}")

    output_dir = create_run_output_dir(OUTPUT_ROOT_DIR, SEED_START, SEED_END)

    print(f"Results directory: {RESULTS_DIR}")
    if SEED_START is not None:
        print(f"Seed range: {SEED_START}-{SEED_END}")
    else:
        print(f"Seed range: all detected ({len(folders)} folders)")
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

    source_tables: Dict[str, pd.DataFrame] = {}
    for source_name in _sorted_sources(list(source_matrices.keys())):
        matrix = source_matrices[source_name]
        source_tables[source_name] = add_difficulty_columns(build_score_metric_table(matrix))

    merged_df = merge_score_tables(source_tables)
    method_summary_df = build_method_score_summary(source_tables)
    consensus_hard_df = build_consensus_hard_samples_table(
        source_tables=source_tables,
        top_k=TOP_K_CONSENSUS_HARD,
    )

    merged_path = os.path.join(output_dir, "score_metrics_merged.csv")
    method_summary_path = os.path.join(output_dir, "method_score_summary.csv")
    consensus_hard_path = os.path.join(output_dir, "consensus_hard_samples.csv")
    seed_manifest_path = os.path.join(output_dir, "seed_manifest.csv")
    overall_comparison_plot_path = os.path.join(output_dir, "overall_method_comparison.png")
    consensus_heatmap_plot_path = os.path.join(output_dir, "consensus_hard_samples_heatmap.png")
    scatter_plot_path = os.path.join(
        output_dir, f"scatter_{SCATTER_X_METRIC}_vs_{SCATTER_Y_METRIC}.png"
    )
    config_path = os.path.join(output_dir, "run_config.json")

    per_source_metric_paths = {}
    for source_name in _sorted_sources(list(source_tables.keys())):
        df = source_tables[source_name]
        path = os.path.join(output_dir, f"{source_name}_score_metrics.csv")
        per_source_metric_paths[source_name] = path
        df.to_csv(path, index=False)
    merged_df.to_csv(merged_path, index=False)
    method_summary_df.to_csv(method_summary_path, index=False)
    consensus_hard_df.to_csv(consensus_hard_path, index=False)

    seed_manifest_rows = []
    for seed, folder_name, _ in folders:
        row = {"seed": seed, "folder": folder_name}
        for source_name, matrix in source_matrices.items():
            row[f"used_in_{source_name}"] = int(seed in matrix.used_seeds)
        seed_manifest_rows.append(row)
    pd.DataFrame(seed_manifest_rows).to_csv(seed_manifest_path, index=False)

    overall_comparison_generated = False
    if GENERATE_OVERALL_COMPARISON:
        try:
            plot_overall_method_comparison(
                summary_df=method_summary_df,
                output_path=overall_comparison_plot_path,
            )
            overall_comparison_generated = True
        except Exception as exc:
            print(f"[WARN] Overall comparison plot generation failed: {exc}")

    consensus_heatmap_generated = False
    if GENERATE_CONSENSUS_HEATMAP:
        try:
            plot_consensus_hard_samples_grouped_points(
                consensus_df=consensus_hard_df,
                source_tables=source_tables,
                output_path=consensus_heatmap_plot_path,
                top_k=TOP_K_CONSENSUS_HARD,
            )
            consensus_heatmap_generated = True
        except Exception as exc:
            print(f"[WARN] Consensus hard-sample grouped-point plot generation failed: {exc}")

    scatter_generated = False
    if GENERATE_SCATTER:
        try:
            plot_mean_variability_scatter(
                source_tables=source_tables,
                x_metric=SCATTER_X_METRIC,
                y_metric=SCATTER_Y_METRIC,
                annotate_top_n=ANNOTATE_TOP_VOLATILE_N,
                output_path=scatter_plot_path,
            )
            scatter_generated = True
        except Exception as exc:
            print(f"[WARN] Scatter plot generation failed: {exc}")

    run_config = {
        "results_dir": os.path.abspath(RESULTS_DIR),
        "seed_start": SEED_START,
        "seed_end": SEED_END,
        "n_detected_seed_folders": len(folders),
        "output_dir": os.path.abspath(output_dir),
        "top_k_consensus_hard": TOP_K_CONSENSUS_HARD,
        "generate_overall_comparison": GENERATE_OVERALL_COMPARISON,
        "generate_consensus_heatmap": GENERATE_CONSENSUS_HEATMAP,
        "generate_scatter": GENERATE_SCATTER,
        "scatter_x_metric": SCATTER_X_METRIC,
        "scatter_y_metric": SCATTER_Y_METRIC,
        "annotate_top_volatile_n": ANNOTATE_TOP_VOLATILE_N,
        "ensemble_score_abs_hard_limit": ENSEMBLE_SCORE_ABS_HARD_LIMIT,
        "final_table_score_abs_hard_limit": FINAL_TABLE_SCORE_ABS_HARD_LIMIT,
        "baseline_score_abs_hard_limit": BASELINE_SCORE_ABS_HARD_LIMIT,
        "enable_baseline": ENABLE_BASELINE,
        "baseline_required": BASELINE_REQUIRED,
        "overall_comparison_output_generated": overall_comparison_generated,
        "consensus_heatmap_output_generated": consensus_heatmap_generated,
        "scatter_output_generated": scatter_generated,
        "method_score_summary_path": os.path.abspath(method_summary_path),
        "consensus_hard_samples_path": os.path.abspath(consensus_hard_path),
        "sources": {},
        "generated_at": datetime.now().isoformat(),
    }
    for source_name, matrix in source_matrices.items():
        run_config["sources"][source_name] = {
            "n_used_seeds": len(matrix.used_seeds),
            "used_seeds": matrix.used_seeds,
            "n_samples": int(matrix.score_matrix.shape[0]),
            "n_models": int(matrix.score_matrix.shape[1]),
            "skipped_seeds": matrix.skipped_seeds,
        }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    for source_name in _sorted_sources(list(per_source_metric_paths.keys())):
        print(f"- {per_source_metric_paths[source_name]}")
    print(f"- {merged_path}")
    print(f"- {method_summary_path}")
    print(f"- {consensus_hard_path}")
    print(f"- {seed_manifest_path}")
    if overall_comparison_generated:
        print(f"- {overall_comparison_plot_path}")
    if consensus_heatmap_generated:
        print(f"- {consensus_heatmap_plot_path}")
    if scatter_generated:
        print(f"- {scatter_plot_path}")
    print(f"- {config_path}")


if __name__ == "__main__":
    main()
