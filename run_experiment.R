library(AutoScore)
library(pROC)

source("config.R")
source("utils.R")
configure_experiment(commandArgs(trailingOnly = TRUE))

if (!exists("synthetic_add_noise")) synthetic_add_noise <- TRUE
if (!exists("synthetic_ensure_non_negative")) synthetic_ensure_non_negative <- TRUE
if (!exists("feature_drop_enable")) feature_drop_enable <- FALSE
if (!exists("feature_drop_range")) feature_drop_range <- c(0, 0)
if (!exists("feature_drop_protected")) feature_drop_protected <- c()
if (!exists("feature_drop_seed_offset")) feature_drop_seed_offset <- 20000

# ==================== 数据加载（所有实验共用）====================
input_data <- load_experiment_data(train_data_path, test_data_path)
train_set_full <- input_data$train
test_set <- input_data$test

# ==================== 单次实验函数 ====================
run_one_experiment <- function(experiment_seed, bootstrap_seed, autoscore_seed) {

if (train_drop_ratio < 0 || train_drop_ratio >= 1) {
  stop("train_drop_ratio must be in [0, 1).")
}
train_set <- train_set_full
n_train_full <- nrow(train_set_full)
drop_n <- floor(n_train_full * train_drop_ratio)
if (drop_n > 0) {
  set.seed(experiment_seed)
  drop_idx <- sample(seq_len(n_train_full), size = drop_n, replace = FALSE)
  train_set <- train_set_full[-drop_idx, , drop = FALSE]
}
n_train_reduced <- nrow(train_set)
actual_drop_ratio <- (n_train_full - n_train_reduced) / n_train_full

cat("========================================\n")
cat("  AutoScore Ensemble Experiment\n")
cat("  experiment_seed:", experiment_seed, "\n")
cat("  bootstrap_seed: ", bootstrap_seed, "\n")
cat("  autoscore_seed: ", autoscore_seed, "\n")
cat("========================================\n")
cat("训练集:", nrow(train_set), "行 | 测试集:", nrow(test_set), "行\n\n")

# 创建输出目录
cat("  Train rows (full -> reduced): ", n_train_full, " -> ", n_train_reduced, "\n", sep = "")
cat("  Actual drop ratio: ", round(actual_drop_ratio, 4), "\n\n", sep = "")

exp_dir <- create_exp_dir(experiment_seed)

if (!is.logical(feature_drop_enable) || length(feature_drop_enable) != 1 || is.na(feature_drop_enable)) {
  stop("feature_drop_enable must be TRUE or FALSE.")
}

if (length(feature_drop_range) != 2 || any(!is.finite(feature_drop_range))) {
  stop("feature_drop_range must contain exactly two finite numbers, e.g., c(1, 5).")
}
feature_drop_range <- sort(as.integer(feature_drop_range))
if (any(is.na(feature_drop_range)) || feature_drop_range[1] < 0) {
  stop("feature_drop_range must be non-negative integers.")
}
configured_min_drop <- feature_drop_range[1]
configured_max_drop <- feature_drop_range[2]

if (!is.numeric(feature_drop_seed_offset) || length(feature_drop_seed_offset) != 1 ||
    !is.finite(feature_drop_seed_offset)) {
  stop("feature_drop_seed_offset must be one finite number.")
}
feature_drop_seed_offset <- as.integer(feature_drop_seed_offset)

feature_cols_all <- setdiff(colnames(train_set), "label")
n_total_features <- length(feature_cols_all)
if (n_total_features == 0) {
  stop("No feature columns found in train_set (only label).")
}

protected_cols <- unique(as.character(feature_drop_protected))
protected_cols <- protected_cols[nzchar(protected_cols)]
unknown_protected <- setdiff(protected_cols, feature_cols_all)
if (length(unknown_protected) > 0) {
  stop("feature_drop_protected contains unknown columns: ",
       paste(unknown_protected, collapse = ", "))
}

candidate_drop_cols <- setdiff(feature_cols_all, protected_cols)
# Keep at least one feature so AutoScore can still train.
min_keep_features <- 1
max_drop_by_keep <- n_total_features - min_keep_features
max_drop_by_candidate <- length(candidate_drop_cols)

if (isTRUE(feature_drop_enable)) {
  effective_max_drop <- min(configured_max_drop, max_drop_by_keep, max_drop_by_candidate)
  effective_min_drop <- min(configured_min_drop, effective_max_drop)
} else {
  effective_min_drop <- 0
  effective_max_drop <- 0
}

cat("  Feature drop enabled: ", isTRUE(feature_drop_enable), "\n", sep = "")
cat("  Feature drop configured range: [", configured_min_drop, ", ", configured_max_drop, "]\n", sep = "")
cat("  Feature drop effective range: [", effective_min_drop, ", ", effective_max_drop, "]\n", sep = "")
cat("  Feature drop protected columns: ",
    if (length(protected_cols) == 0) "none" else paste(protected_cols, collapse = ", "),
    "\n\n", sep = "")

# ==================== Step 1: Ensemble 训练 ====================
cat("===== Step 1: Ensemble 训练 (", n_bootstrap, "个模型) =====\n")

scoring_tables <- list()
cut_vecs <- list()
final_variables_list <- list()
feature_drop_k <- integer(n_bootstrap)
feature_drop_dropped_columns <- vector("list", n_bootstrap)
feature_drop_kept_columns <- vector("list", n_bootstrap)

pdf(NULL)
for (i in 1:n_bootstrap) {
  cat("  模型", i, "/", n_bootstrap, "\n")
  
  # Bootstrap 采样（独立 seed）
  set.seed(bootstrap_seed + i)
  n <- nrow(train_set)
  boot_indices <- sample(1:n, n, replace = TRUE)
  boot_train <- train_set[boot_indices, ]
  dropped_cols_i <- character(0)
  boot_train_sub <- boot_train
  if (effective_max_drop > 0) {
    set.seed(autoscore_seed + feature_drop_seed_offset + i)
    if (effective_min_drop == effective_max_drop) {
      drop_k <- effective_max_drop
    } else {
      drop_k <- sample.int(effective_max_drop - effective_min_drop + 1, size = 1) + effective_min_drop - 1
    }
    if (drop_k > 0) {
      dropped_cols_i <- sample(candidate_drop_cols, size = drop_k, replace = FALSE)
      keep_cols_i <- setdiff(colnames(boot_train), dropped_cols_i)
      boot_train_sub <- boot_train[, keep_cols_i, drop = FALSE]
    }
  }
  feature_drop_k[i] <- length(dropped_cols_i)
  feature_drop_dropped_columns[[i]] <- dropped_cols_i
  feature_drop_kept_columns[[i]] <- setdiff(feature_cols_all, dropped_cols_i)
  cat("    drop_k:", feature_drop_k[i], "\n")
  
  # AutoScore 训练（独立 seed）
  set.seed(autoscore_seed + i)
  ranking <- suppressMessages(
    AutoScore_rank(train_set = boot_train_sub, method = "rf", ntree = ntree)
  )
  use_num_var <- min(num_var, length(ranking))
  final_variables <- names(ranking)[seq_len(use_num_var)]
  
  cut_vec <- suppressMessages(
    AutoScore_weighting(
      train_set = boot_train_sub, validation_set = boot_train_sub,
      final_variables = final_variables, max_score = max_score,
      categorize = "quantile", quantiles = quantiles
    )
  )
  
  scoring_table <- suppressMessages(
    AutoScore_fine_tuning(
      train_set = boot_train_sub, validation_set = boot_train_sub,
      final_variables = final_variables, cut_vec = cut_vec, max_score = max_score
    )
  )
  
  scoring_tables[[i]] <- scoring_table
  cut_vecs[[i]] <- cut_vec
  final_variables_list[[i]] <- final_variables
}
dev.off()
feature_drop_summary_df <- data.frame(
  model_index = seq_len(n_bootstrap),
  drop_k = feature_drop_k,
  n_features_before = n_total_features,
  n_features_after = n_total_features - feature_drop_k,
  dropped_columns = vapply(
    feature_drop_dropped_columns,
    function(x) if (length(x) == 0) "" else paste(x, collapse = ";"),
    character(1)
  ),
  kept_columns = vapply(
    feature_drop_kept_columns,
    function(x) if (length(x) == 0) "" else paste(x, collapse = ";"),
    character(1)
  ),
  stringsAsFactors = FALSE
)
write.csv(feature_drop_summary_df, file.path(exp_dir, "feature_drop_summary.csv"), row.names = FALSE)
cat("Ensemble 训练完成！\n\n")

# ==================== Step 2: Ensemble 预测测试集 ====================
cat("===== Step 2: Ensemble 预测测试集 =====\n")

ensemble_pred_matrix <- matrix(0, nrow = nrow(test_set), ncol = n_bootstrap)
individual_aucs <- numeric(n_bootstrap)

for (i in 1:n_bootstrap) {
  pred <- suppressMessages(
    AutoScore_testing(
      test_set = test_set,
      final_variables = final_variables_list[[i]],
      cut_vec = cut_vecs[[i]],
      scoring_table = scoring_tables[[i]],
      threshold = "best", with_label = FALSE
    )
  )
  ensemble_pred_matrix[, i] <- pred$pred_score
  individual_aucs[i] <- as.numeric(auc(roc(test_set$label, pred$pred_score, quiet = TRUE)))
}

ensemble_mean_score <- rowMeans(ensemble_pred_matrix)
ensemble_roc <- roc(test_set$label, ensemble_mean_score, quiet = TRUE)
ensemble_auc <- as.numeric(auc(ensemble_roc))

cat("  单模型 AUC: [", round(min(individual_aucs), 4), "-",
    round(max(individual_aucs), 4), "], 均值", round(mean(individual_aucs), 4), "\n")
cat("  Ensemble AUC:", round(ensemble_auc, 4), "\n\n")

# 保存 ensemble 逐样本预测
save_predictions(test_set$label, ensemble_pred_matrix,
                 file.path(exp_dir, "ensemble_predictions.csv"))

# 保存 ensemble 模型对象
best_threshold <- coords(ensemble_roc, "best", ret = "threshold", quiet = TRUE)$threshold
ensemble_model <- list(
  scoring_tables = scoring_tables,
  cut_vecs = cut_vecs,
  final_variables_list = final_variables_list,
  feature_drop = list(
    enable = isTRUE(feature_drop_enable),
    configured_range = c(configured_min_drop, configured_max_drop),
    effective_range = c(effective_min_drop, effective_max_drop),
    protected_columns = protected_cols,
    seed_offset = feature_drop_seed_offset,
    summary = feature_drop_summary_df
  ),
  n_bootstrap = n_bootstrap,
  threshold = best_threshold,
  auc = ensemble_auc
)
saveRDS(ensemble_model, file.path(exp_dir, "ensemble_model.rds"))

# ==================== Step 3: 生成合成数据 ====================
cat("===== Step 3: 生成合成数据 & 集成标注 =====\n")

# 用训练集的特征列生成合成数据（去掉 label）
train_features <- train_set[, colnames(train_set) != "label"]
synthetic_features <- generate_synthetic_data(
  train_features,
  n_synthetic_rows,
  noise_level,
  seed = experiment_seed,
  add_noise = synthetic_add_noise,
  ensure_non_negative = synthetic_ensure_non_negative
)

# 用 ensemble 模型为合成数据标注
synthetic_with_dummy <- cbind(label = 0, synthetic_features)
synthetic_preds <- matrix(0, nrow = n_synthetic_rows, ncol = n_bootstrap)

for (i in 1:n_bootstrap) {
  pred <- suppressMessages(
    AutoScore_testing(
      test_set = synthetic_with_dummy,
      final_variables = final_variables_list[[i]],
      cut_vec = cut_vecs[[i]],
      scoring_table = scoring_tables[[i]],
      threshold = "best", with_label = FALSE
    )
  )
  synthetic_preds[, i] <- pred$pred_score
}

synthetic_mean <- rowMeans(synthetic_preds)
synthetic_labels <- ifelse(synthetic_mean > best_threshold, 1, 0)
synthetic_train <- cbind(label = synthetic_labels, synthetic_features)

cat("  生成", n_synthetic_rows, "行合成数据\n")
cat("  标签分布: 0 =", sum(synthetic_labels == 0),
    ", 1 =", sum(synthetic_labels == 1), "\n\n")

# ==================== Step 4: Final Table 训练 ====================
cat("===== Step 4: Final Scoring Table 训练 =====\n")

set.seed(autoscore_seed)
pdf(NULL)

ft_ranking <- suppressMessages(
  AutoScore_rank(train_set = synthetic_train, method = "rf", ntree = ntree)
)
ft_use_num_var <- min(num_var, length(ft_ranking))
ft_variables <- names(ft_ranking)[seq_len(ft_use_num_var)]

ft_cut_vec <- suppressMessages(
  AutoScore_weighting(
    train_set = synthetic_train, validation_set = synthetic_train,
    final_variables = ft_variables, max_score = max_score,
    categorize = "quantile", quantiles = quantiles
  )
)

ft_scoring_table <- suppressMessages(
  AutoScore_fine_tuning(
    train_set = synthetic_train, validation_set = synthetic_train,
    final_variables = ft_variables, cut_vec = ft_cut_vec, max_score = max_score
  )
)
dev.off()

# Final Table 预测测试集
ft_pred <- suppressMessages(
  AutoScore_testing(
    test_set = test_set,
    final_variables = ft_variables,
    cut_vec = ft_cut_vec,
    scoring_table = ft_scoring_table,
    threshold = "best", with_label = FALSE
  )
)

ft_auc <- as.numeric(auc(roc(test_set$label, ft_pred$pred_score, quiet = TRUE)))
cat("  Final Table AUC:", round(ft_auc, 4), "\n\n")

# 保存 final table 逐样本预测
save_predictions(test_set$label, ft_pred$pred_score,
                 file.path(exp_dir, "final_table_predictions.csv"))

# 保存 final scoring table
saveRDS(list(
  scoring_table = ft_scoring_table,
  cut_vec = ft_cut_vec,
  final_variables = ft_variables
), file.path(exp_dir, "final_scoring_table.rds"))

# ==================== Baseline on reduced train set ====================
cat("===== Step 5: Baseline on Reduced Train Set =====\n")

pdf(NULL)
baseline_ranking <- suppressMessages(
  AutoScore_rank(train_set = train_set, method = "rf", ntree = ntree)
)
baseline_use_num_var <- min(num_var, length(baseline_ranking))
baseline_variables <- names(baseline_ranking)[seq_len(baseline_use_num_var)]

baseline_cut_vec <- suppressMessages(
  AutoScore_weighting(
    train_set = train_set, validation_set = train_set,
    final_variables = baseline_variables, max_score = max_score,
    categorize = "quantile", quantiles = quantiles
  )
)

baseline_scoring_table <- suppressMessages(
  AutoScore_fine_tuning(
    train_set = train_set, validation_set = train_set,
    final_variables = baseline_variables, cut_vec = baseline_cut_vec, max_score = max_score
  )
)
dev.off()

baseline_pred <- suppressMessages(
  AutoScore_testing(
    test_set = test_set,
    final_variables = baseline_variables,
    cut_vec = baseline_cut_vec,
    scoring_table = baseline_scoring_table,
    threshold = "best", with_label = FALSE
  )
)
baseline_auc <- as.numeric(auc(roc(test_set$label, baseline_pred$pred_score, quiet = TRUE)))
cat("  Baseline AUC:", round(baseline_auc, 4), "\n\n")
# Save baseline per-sample predictions
save_predictions(test_set$label, baseline_pred$pred_score,
                 file.path(exp_dir, "baseline_predictions.csv"))


# ==================== Step 5: 汇总输出 ====================
cat("===== 汇总 =====\n")

summary_df <- data.frame(
  metric = c("ensemble_auc", "final_table_auc", "baseline_auc", "ensemble_threshold",
             "individual_auc_mean", "individual_auc_min",
             "individual_auc_max", "individual_auc_sd",
             paste0("individual_auc_", 1:n_bootstrap)),
  value = c(ensemble_auc, ft_auc, baseline_auc, best_threshold,
            mean(individual_aucs), min(individual_aucs),
            max(individual_aucs), sd(individual_aucs),
            individual_aucs)
)
write.csv(summary_df, file.path(exp_dir, "summary.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(exp_dir, "sessionInfo.txt"))

# 保存完整配置
save_config_json(list(
  experiment_seed   = experiment_seed,
  bootstrap_seed    = bootstrap_seed,
  autoscore_seed    = autoscore_seed,
  n_bootstrap       = n_bootstrap,
  num_var           = num_var,
  ntree             = ntree,
  max_score         = max_score,
  quantiles         = quantiles,
  feature_drop_enable = isTRUE(feature_drop_enable),
  feature_drop_range = c(configured_min_drop, configured_max_drop),
  feature_drop_effective_range = c(effective_min_drop, effective_max_drop),
  feature_drop_protected = if (length(protected_cols) == 0) "" else paste(protected_cols, collapse = ";"),
  feature_drop_seed_offset = feature_drop_seed_offset,
  feature_drop_summary_file = "feature_drop_summary.csv",
  feature_drop_k = feature_drop_k,
  train_drop_ratio  = train_drop_ratio,
  n_train_full      = n_train_full,
  n_train_reduced   = n_train_reduced,
  n_synthetic_rows  = n_synthetic_rows,
  synthetic_add_noise = synthetic_add_noise,
  synthetic_ensure_non_negative = synthetic_ensure_non_negative,
  noise_level       = noise_level,
  ensemble_auc      = round(ensemble_auc, 6),
  final_table_auc   = round(ft_auc, 6),
  ensemble_threshold = round(best_threshold, 6)
), file.path(exp_dir, "config.json"))

cat("  Ensemble AUC:     ", round(ensemble_auc, 4), "\n")
cat("  Final Table AUC:  ", round(ft_auc, 4), "\n")
cat("  Baseline AUC:     ", round(baseline_auc, 4), "\n")
cat("  Ensemble 阈值:    ", round(best_threshold, 4), "\n")
cat("  单模型 AUC 范围:   [", round(min(individual_aucs), 4), "-",
    round(max(individual_aucs), 4), "]\n")
cat("\n结果已保存到:", exp_dir, "/\n")
cat("========================================\n")

} # end of run_one_experiment

# ==================== 批量执行 ====================
cat("共", length(experiment_seeds), "组实验\n\n")
for (exp_seed in experiment_seeds) {
  seeds <- derive_seeds(exp_seed)
  run_one_experiment(seeds$experiment_seed, seeds$bootstrap_seed, seeds$autoscore_seed)
  cat("\n")
}
cat("全部实验完成！\n")
