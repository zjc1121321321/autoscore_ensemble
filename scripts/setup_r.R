# Run from the repository root: Rscript scripts/setup_r.R
if (getRversion() < "4.3.0") stop("Please use R >= 4.3; tested with R 4.5.2.")
if (!file.exists("AutoScore-master/DESCRIPTION")) stop("Run this script from the repository root.")
options(repos = c(CRAN = "https://cloud.r-project.org"))
desc <- read.dcf("AutoScore-master/DESCRIPTION")
imports <- trimws(strsplit(desc[1, "Imports"], ",")[[1]])
imports <- sub("[[:space:]]*\\(.*\\)$", "", imports)
packages <- unique(c(imports, "jsonlite"))
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing)
remaining <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(remaining)) stop("Missing dependencies: ", paste(remaining, collapse = ", "))
install.packages("AutoScore-master", repos = NULL, type = "source")
if (!requireNamespace("AutoScore", quietly = TRUE)) stop("AutoScore installation failed.")
cat("R dependencies ready. See requirements-r.csv for the tested package versions.\n")
