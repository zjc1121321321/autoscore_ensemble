# ==================== Global Experiment Config ====================

# === Experiment seeds ===
# Preferred: configure an inclusive seed range.
# Example: start=1, end=5 will run seeds 1,2,3,4,5.
# Start with two runs; use --seed-start 1001 --seed-end 1500 for the full study.
experiment_seed_start <- 1001
experiment_seed_end <- 1002

# Optional: explicit seed list. If non-empty, it overrides the range above.
# experiment_seeds <- c(35, 45, 55, 65)
experiment_seeds <- NULL

resolve_experiment_seeds <- function(seed_start, seed_end, seed_list) {
  if (!is.null(seed_list) && length(seed_list) > 0) {
    seed_list <- as.integer(seed_list)
    if (any(is.na(seed_list))) stop("experiment_seeds must be integers.")
    return(seed_list)
  }

  if (!is.numeric(seed_start) || !is.numeric(seed_end) ||
      !is.finite(seed_start) || !is.finite(seed_end)) {
    stop("experiment_seed_start and experiment_seed_end must be finite numbers.")
  }

  seed_start <- as.integer(seed_start)
  seed_end <- as.integer(seed_end)

  if (seed_start > seed_end) stop("experiment_seed_start must be <= experiment_seed_end.")

  seed_start:seed_end
}

experiment_seeds <- resolve_experiment_seeds(
  seed_start = experiment_seed_start,
  seed_end = experiment_seed_end,
  seed_list = experiment_seeds
)

# Seed derivation rule for one experiment round.
derive_seeds <- function(exp_seed) {
  list(
    experiment_seed = exp_seed,
    bootstrap_seed = exp_seed,
    autoscore_seed = exp_seed + 10000
  )
}

# === Model parameters ===
n_bootstrap <- 30
num_var <- 11
ntree <- 100
max_score <- 100
quantiles <- c(0, 0.05, 0.2, 0.8, 0.95, 1)

# === Feature drop for ensemble diversity (run_experiment.R) ===
# For each scoring table, randomly drop k feature columns before training.
# Example: c(1, 5) means each model drops 1 to 5 columns.
feature_drop_enable <- FALSE
feature_drop_range <- c(2, 3)
feature_drop_protected <- c()
feature_drop_seed_offset <- 20000

# === Synthetic data parameters (for Final Table) ===
n_synthetic_rows <- 20000
synthetic_add_noise <- FALSE
synthetic_ensure_non_negative <- TRUE
noise_level <- 0.1

# === Random sample removal on training set ===
train_drop_ratio <- 0.1

# === Baseline script parameters (run_baseline.R) ===
baseline_repeats <- 1
baseline_seed_start <- 777

# === Project paths (all relative to repo root) ===
project_root <- "."

data_root <- "data"
data_raw_dir <- file.path(data_root, "raw")
data_split_dir <- file.path(data_root, "split")
data_generated_dir <- file.path(data_root, "generated")

outputs_root <- "outputs"
results_root <- file.path(outputs_root, "results")
analysis_output_root <- file.path(outputs_root, "analysis")
test_reports_root <- file.path(outputs_root, "test_reports")

archive_root <- "archive"
archive_results_root <- file.path(archive_root, "results")
archive_analysis_root <- file.path(archive_root, "analysis")

# === Data files ===
train_data_path <- file.path(data_split_dir, "train_data.csv")
test_data_path <- file.path(data_split_dir, "test_data.csv")
generated_data_path <- file.path(data_generated_dir, "generated_data.csv")

# === Common output files ===
predict_output_path <- file.path(results_root, "predictions.csv")
