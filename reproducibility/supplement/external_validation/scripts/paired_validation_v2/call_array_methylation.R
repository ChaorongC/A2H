#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(optparse)
})

option_list <- list(
    make_option("--root", default = "external_validation"),
    make_option("--samples", default = "metadata/paired_validation_v2/paired_samples.tsv"),
    make_option("--idats", default = "metadata/paired_validation_v2/paired_array_idat_manifest.tsv"),
    make_option("--study", default = NULL),
    make_option("--platform", default = NULL),
    make_option("--outdir", default = "results/paired_validation_v2/array"),
    make_option("--detection-p", type = "double", default = 0.01),
    make_option("--max-failed-probe-fraction", type = "double", default = 0.05)
)

parser <- OptionParser(
    usage = "%prog --study STUDY --platform 450K|EPIC [options]",
    option_list = option_list,
    description = "Call paired-validation-v2 array beta matrices from sample-level IDAT files."
)
args <- parse_args(parser)

if (is.null(args$study) || is.null(args$platform)) {
    print_help(parser)
    stop("--study and --platform are required", call. = FALSE)
}

suppressPackageStartupMessages({
    library(minfi)
    library(DMRcate)
})

root <- normalizePath(args$root, mustWork = TRUE)
sample_path <- file.path(root, args$samples)
idat_path <- file.path(root, args$idats)
outdir <- file.path(root, args$outdir)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

paired_samples <- read.delim(sample_path, stringsAsFactors = FALSE, check.names = FALSE)
idat_manifest <- read.delim(idat_path, stringsAsFactors = FALSE, check.names = FALSE)

target_samples <- paired_samples[
    paired_samples$study == args$study & paired_samples$platform == args$platform,
]
if (nrow(target_samples) == 0) {
    stop("No paired samples matched study/platform", call. = FALSE)
}

idats <- idat_manifest[
    idat_manifest$study == args$study & idat_manifest$platform == args$platform,
]
if (nrow(idats) != 2 * nrow(target_samples)) {
    stop("Expected two IDAT channels per target sample", call. = FALSE)
}

sample_sheet <- unique(idats[c("array_id")])
sample_sheet$Basename <- vapply(
    sample_sheet$array_id,
    function(array_id) {
        sample_idats <- idats[idats$array_id == array_id, ]
        green_path <- sample_idats$local_path[sample_idats$channel == "Grn"]
        red_path <- sample_idats$local_path[sample_idats$channel == "Red"]
        if (length(green_path) != 1 || length(red_path) != 1) {
            stop(sprintf("Expected one Grn and one Red IDAT for %s", array_id), call. = FALSE)
        }
        green_base <- sub("_Grn\\.idat\\.gz$", "", file.path(root, green_path))
        red_base <- sub("_Red\\.idat\\.gz$", "", file.path(root, red_path))
        if (green_base != red_base) {
            stop(sprintf("IDAT channel basenames do not match for %s", array_id), call. = FALSE)
        }
        green_base
    },
    character(1)
)
sample_sheet$Sample_Name <- sample_sheet$array_id
sample_sheet$Sample_Group <- target_samples$subject_or_cell[
    match(sample_sheet$array_id, target_samples$array_id)
]
sample_sheet$Slide <- sub(".*_([0-9]+)_R[0-9]+C[0-9]+$", "\\1", sample_sheet$Basename)
sample_sheet$Array <- sub(".*_(R[0-9]+C[0-9]+)$", "\\1", sample_sheet$Basename)

rg_set <- read.metharray.exp(targets = sample_sheet, force = TRUE)
det_p <- detectionP(rg_set)
failed_fraction <- colMeans(det_p > args$`detection-p`, na.rm = TRUE)
keep_samples <- failed_fraction <= args$`max-failed-probe-fraction`

qc <- data.frame(
    array_id = colnames(det_p),
    failed_detection_probe_fraction = failed_fraction,
    detection_p_threshold = args$`detection-p`,
    max_failed_probe_fraction = args$`max-failed-probe-fraction`,
    kept = keep_samples,
    stringsAsFactors = FALSE
)

if (!any(keep_samples)) {
    qc_path <- file.path(outdir, sprintf("%s_%s.sample_qc.tsv", args$study, args$platform))
    write.table(qc, qc_path, sep = "\t", quote = FALSE, row.names = FALSE)
    stop("No array samples passed detection-p filtering", call. = FALSE)
}

rg_set <- rg_set[, keep_samples]
det_p <- det_p[, keep_samples, drop = FALSE]

mset <- preprocessQuantile(rg_set)
beta <- getBeta(mset)
is_cg <- grepl("^cg", rownames(beta))
beta <- beta[is_cg, , drop = FALSE]

beta <- rmSNPandCH(beta, dist = 2, mafcut = 0.05)

qc$final_cg_probe_count <- nrow(beta)
qc$normalization_batch <- sprintf("%s_%s", args$study, args$platform)

beta_df <- data.frame(Probe_ID = rownames(beta), beta, check.names = FALSE)
beta_path <- file.path(outdir, sprintf("%s_%s.beta.tsv", args$study, args$platform))
qc_path <- file.path(outdir, sprintf("%s_%s.sample_qc.tsv", args$study, args$platform))
write.table(beta_df, beta_path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
write.table(qc, qc_path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

message("Wrote beta matrix: ", beta_path)
message("Wrote sample QC: ", qc_path)
