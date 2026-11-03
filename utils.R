
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  path
}

create_exp_dir <- function(experiment_seed, results_dir = NULL) {
  if (is.null(results_dir)) {
    if (exists("results_root", inherits = TRUE)) {
      results_dir <- get("results_root", inherits = TRUE)
    } else {
      results_dir <- file.path("outputs", "results")
    }
  }
  dir_name <- file.path(results_dir, paste0("exp_seed", experiment_seed))
  if (!dir.exists(dir_name)) dir.create(dir_name, recursive = TRUE)
  return(dir_name)
}

save_config_json <- function(config_list, filepath) {
  jsonlite::write_json(config_list, filepath, auto_unbox = TRUE,
                       pretty = TRUE, digits = NA, na = "null")
}

# CLI overrides do not change the research configuration on disk.
configure_experiment <- function(args = character(0), envir = parent.frame()) {
  allowed <- c("--seed-start", "--seed-end", "--results-dir", "--train-data", "--test-data")
  opts <- list()
  demo <- FALSE
  i <- 1L
  while (i <= length(args)) {
    key <- args[i]
    if (key == "--demo") {
      demo <- TRUE
      i <- i + 1L
      next
    }
    if (!key %in% allowed || i == length(args) || startsWith(args[i + 1L], "--")) {
      stop("Usage: Rscript run_experiment.R [--demo] [--seed-start N --seed-end N] ",
           "[--results-dir PATH] [--train-data PATH --test-data PATH]")
    }
    if (!is.null(opts[[key]])) stop("Duplicate option: ", key)
    opts[[key]] <- args[i + 1L]
    i <- i + 2L
  }
  if (demo) {
    assign("train_data_path", file.path("data", "demo", "train_data.csv"), envir)
    assign("test_data_path", file.path("data", "demo", "test_data.csv"), envir)
    assign("results_root", file.path("outputs", "demo", "results"), envir)
    assign("experiment_seeds", 1001:1002, envir)
  }
  if (xor(is.null(opts[["--seed-start"]]), is.null(opts[["--seed-end"]]))) {
    stop("--seed-start and --seed-end must be provided together.")
  }
  if (!is.null(opts[["--seed-start"]])) {
    values <- c(opts[["--seed-start"]], opts[["--seed-end"]])
    if (any(!grepl("^[0-9]+$", values))) stop("Seeds must be non-negative integers.")
    values <- as.numeric(values)
    if (any(values > .Machine$integer.max - 30000)) stop("Seed is too large.")
    assign("experiment_seeds", resolve_experiment_seeds(values[1], values[2], NULL), envir)
  }
  mapping <- c("--results-dir" = "results_root", "--train-data" = "train_data_path",
               "--test-data" = "test_data_path")
  for (key in names(mapping)) {
    if (!is.null(opts[[key]])) assign(mapping[[key]], opts[[key]], envir)
  }
}

load_experiment_data <- function(train_path, test_path) {
  paths <- c(train_path, test_path)
  if (any(!file.exists(paths))) {
    stop("Data file not found: ", paste(paths[!file.exists(paths)], collapse = ", "),
         ". See README.md, or create demo data with python scripts/make_demo_data.py.")
  }
  train <- read.csv(train_path, stringsAsFactors = FALSE, check.names = FALSE)
  test <- read.csv(test_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!identical(names(train), names(test))) stop("Train/test columns must match, including their order.")
  if (anyDuplicated(names(train))) stop("Duplicate column names are not supported.")
  for (df in list(train, test)) {
    if (!"label" %in% names(df) || ncol(df) < 2 || nrow(df) < 2) {
      stop("Each CSV needs a label column, at least one feature, and at least two rows.")
    }
    if (!all(vapply(df, is.numeric, logical(1))) || any(!is.finite(as.matrix(df)))) {
      stop("All input values must be finite numbers; handle missing values before training.")
    }
    if (!setequal(unique(df$label), c(0, 1))) stop("Both label classes 0 and 1 must be present in each CSV.")
  }
  list(train = train, test = test)
}

save_predictions <- function(true_labels, pred_data, filepath, col_prefix = "model") {
  df <- data.frame(sample_id = seq_along(true_labels), true_label = true_labels)
  if (is.matrix(pred_data)) {
    pred_df <- as.data.frame(pred_data)
    colnames(pred_df) <- paste0(col_prefix, "_", 1:ncol(pred_data))
    df <- cbind(df, pred_df)
    df$ensemble_mean <- rowMeans(pred_data)
  } else {
    df$pred_score <- pred_data
  }
  write.csv(df, filepath, row.names = FALSE)
}

generate_synthetic_data <- function(
  data,
  n_rows,
  noise_level = 0.1,
  seed = NULL,
  add_noise = TRUE,
  ensure_non_negative = TRUE
) {
  if (!is.null(seed)) set.seed(seed)

  if (n_rows <= 0) stop("n_rows must be > 0.")
  if (nrow(data) == 0) stop("input data has 0 rows, cannot bootstrap.")

  sampled_idx <- sample.int(nrow(data), size = n_rows, replace = TRUE)
  result <- data[sampled_idx, , drop = FALSE]
  rownames(result) <- NULL

  if (!isTRUE(add_noise) || is.null(noise_level) || !is.finite(noise_level) || noise_level <= 0) {
    return(result)
  }

  for (col in colnames(result)) {
    if (!is.numeric(result[[col]])) next

    ref_col <- data[[col]]
    is_integer_like <- all(ref_col == round(ref_col), na.rm = TRUE)
    n_unique <- length(unique(ref_col[!is.na(ref_col)]))
    if (is_integer_like || n_unique <= 10) next

    col_sd <- sd(ref_col, na.rm = TRUE)
    if (is.na(col_sd) || col_sd == 0) next

    noise <- rnorm(n_rows, mean = 0, sd = col_sd * noise_level)
    noisy <- result[[col]] + noise
    if (isTRUE(ensure_non_negative)) noisy <- pmax(noisy, 0)
    result[[col]] <- noisy
  }

  return(result)
}

generate_synthetic_data_by_feature_bootstrap <- function(
  data,
  n_rows,
  noise_level = 0.1,
  seed = NULL,
  add_noise = TRUE,
  ensure_non_negative = TRUE
) {
  if (!is.null(seed)) set.seed(seed)

  if (n_rows <= 0) stop("n_rows must be > 0.")
  if (nrow(data) == 0) stop("input data has 0 rows, cannot bootstrap.")

  result <- as.data.frame(matrix(nrow = n_rows, ncol = ncol(data)))
  colnames(result) <- colnames(data)

  for (col in colnames(data)) {
    sampled_idx <- sample.int(nrow(data), size = n_rows, replace = TRUE)
    result[[col]] <- data[[col]][sampled_idx]
  }

  if (!isTRUE(add_noise) || is.null(noise_level) || !is.finite(noise_level) || noise_level <= 0) {
    return(result)
  }

  for (col in colnames(result)) {
    if (!is.numeric(result[[col]])) next

    ref_col <- data[[col]]
    is_integer_like <- all(ref_col == round(ref_col), na.rm = TRUE)
    n_unique <- length(unique(ref_col[!is.na(ref_col)]))
    if (is_integer_like || n_unique <= 10) next

    col_sd <- sd(ref_col, na.rm = TRUE)
    if (is.na(col_sd) || col_sd == 0) next

    noise <- rnorm(n_rows, mean = 0, sd = col_sd * noise_level)
    noisy <- result[[col]] + noise
    if (isTRUE(ensure_non_negative)) noisy <- pmax(noisy, 0)
    result[[col]] <- noisy
  }

  return(result)
}
