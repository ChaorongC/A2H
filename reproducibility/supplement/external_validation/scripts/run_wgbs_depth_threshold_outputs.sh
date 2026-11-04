#!/bin/bash
set -euo pipefail

root="external_validation"
python_bin="python"
a2h_root="legacy_model"
threads="${SLURM_CPUS_PER_TASK:-8}"
cohort=""
min_depth=""
dry_run=0

usage() {
    cat <<'EOF'
usage: run_wgbs_depth_threshold_outputs.sh --cohort gse46644|pgpuk|pidsley|gse240482_prostate_epicv1 --min-depth N [--root DIR] [--python-bin PATH] [--a2h-root DIR] [--threads N] [--dry-run]

Filter existing WGBS methRatio/CHALM text files by CT_count and rebuild direct
WGBS plus A2H-transformed CHALM/CAMDA matrices in depth-specific result roots.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            root="$2"
            shift 2
            ;;
        --python-bin)
            python_bin="$2"
            shift 2
            ;;
        --a2h-root)
            a2h_root="$2"
            shift 2
            ;;
        --threads)
            threads="$2"
            shift 2
            ;;
        --cohort)
            cohort="$2"
            shift 2
            ;;
        --min-depth)
            min_depth="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${cohort}" || -z "${min_depth}" ]]; then
    usage >&2
    exit 1
fi

case "${cohort}" in
    gse46644|pgpuk|pidsley|gse240482_prostate_epicv1)
        ;;
    *)
        echo "Unsupported cohort: ${cohort}" >&2
        exit 1
        ;;
esac

if ! [[ "${min_depth}" =~ ^[0-9]+$ ]] || [[ "${min_depth}" -lt 1 ]]; then
    echo "--min-depth must be a positive integer" >&2
    exit 1
fi

root=$(realpath -m "${root}")
a2h_root=$(realpath -m "${a2h_root}")
cd "${root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_filter() {
    local input_traditional="$1"
    local input_chalm="$2"
    local output_traditional="$3"
    local output_chalm="$4"
    local summary="$5"
    shift 5

    "${python_bin}" "${root}/scripts/filter_wgbs_depth_outputs.py" \
        --input-traditional-dir "${input_traditional}" \
        --input-chalm-dir "${input_chalm}" \
        --output-traditional-dir "${output_traditional}" \
        --output-chalm-dir "${output_chalm}" \
        --min-depth "${min_depth}" \
        --summary "${summary}" \
        "$@"
}

paired_sample_args() {
    local study="$1"
    awk -F'\t' -v study="${study}" 'NR > 1 && $1 == study {print $2}' \
        "${root}/metadata/paired_validation_v2/paired_wgbs_run_manifest.tsv" |
        sort -u |
        awk '{print "--sample-id"; print $0}'
}

paired_cohort_config() {
    case "$1" in
        pgpuk)
            echo "pgpuk 450K pgpuk_450K"
            ;;
        pidsley)
            echo "pidsley 450K pidsley_450K"
            echo "pidsley EPIC pidsley_EPIC"
            ;;
        gse240482_prostate_epicv1)
            echo "epicv2_eval EPIC gse240482_prostate_epicv1_EPIC"
            ;;
        *)
            echo "Unsupported paired cohort: $1" >&2
            return 1
            ;;
    esac
}

run_paired_a2h() {
    local result_root="$1"
    local study="$2"
    local platform="$3"
    local prefix="$4"
    "${python_bin}" "${root}/scripts/generate_paired_validation_a2h_outputs.py" \
        --root "${root}" \
        --namespace paired_validation_v2 \
        --mode wgbs \
        --study "${study}" \
        --platform "${platform}" \
        --input-prefix "${prefix}" \
        --output-prefix "${prefix}" \
        --a2h-root "${a2h_root}" \
        --python-bin "${python_bin}" \
        --threads "${threads}" \
        --wgbs-input-dir "${result_root}/wgbs/probe_space" \
        --a2h-wgbs-dir "${result_root}/wgbs/a2h_from_traditional"
}

if [[ "${cohort}" == "gse46644" ]]; then
    result_root="results/gse46644_depth${min_depth}"
    run_filter \
        "results/traditional" \
        "results/chalm" \
        "${result_root}/traditional" \
        "${result_root}/chalm" \
        "${result_root}/coverage_filter_summary.tsv"

    "${python_bin}" "${root}/scripts/build_original_chalm_camda_matrices.py" \
        --methratio-dir "${result_root}/traditional" \
        --chalm-dir "${result_root}/chalm" \
        --chalm-array-root "${a2h_root}" \
        --output-dir "${result_root}/chalm_array/full" \
        --prefix wgbs_epic_hg38 \
        --beadchip EPIC

    "${python_bin}" "${root}/scripts/generate_gse46644_a2h_outputs.py" \
        --root "${root}" \
        --methratio-dir "${result_root}/traditional" \
        --a2h-root "${a2h_root}" \
        --python-bin "${python_bin}" \
        --output-dir "${result_root}/a2h_from_traditional" \
        --prefix wgbs_epic_hg38 \
        --beadchip EPIC \
        --threads "${threads}" \
        --matrix-path "${result_root}/a2h_from_traditional/wgbs_epic_hg38.matrix.tsv" \
        --rebuild-matrix
else
    result_root="results/paired_validation_v2_depth${min_depth}"
    mapfile -t cohort_configs < <(paired_cohort_config "${cohort}")
    first_config="${cohort_configs[0]}"
    read -r source_study _ _ <<<"${first_config}"
    mapfile -t sample_args < <(paired_sample_args "${source_study}")
    if [[ "${#sample_args[@]}" -eq 0 ]]; then
        echo "No paired-validation-v2 WGBS samples found for ${source_study}" >&2
        exit 1
    fi

    if [[ "${dry_run}" -eq 1 ]]; then
        for config in "${cohort_configs[@]}"; do
            read -r study platform prefix <<<"${config}"
            echo "source_study=${study} platform=${platform} output_prefix=${prefix} result_root=${result_root}"
        done
        exit 0
    fi

    run_filter \
        "results/paired_validation_v2/wgbs/traditional" \
        "results/paired_validation_v2/wgbs/chalm" \
        "${result_root}/wgbs/traditional" \
        "${result_root}/wgbs/chalm" \
        "${result_root}/${cohort}.coverage_filter_summary.tsv" \
        "${sample_args[@]}"

    for config in "${cohort_configs[@]}"; do
        read -r study platform prefix <<<"${config}"
        "${python_bin}" "${root}/scripts/paired_validation_v2/build_wgbs_probe_matrices.py" \
            --root "${root}" \
            --study "${study}" \
            --platform "${platform}" \
            --output-dir "${result_root}/wgbs/probe_space" \
            --traditional-dir "${result_root}/wgbs/traditional" \
            --chalm-dir "${result_root}/wgbs/chalm" \
            --output-prefix "${prefix}"
        run_paired_a2h "${result_root}" "${study}" "${platform}" "${prefix}"
    done
fi

echo "Completed ${cohort} WGBS min-depth ${min_depth}"
