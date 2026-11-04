#!/bin/bash
set -euo pipefail

root="external_validation"
rscript="Rscript"
dependency=""
study_filter=""
platform_filter=""

usage() {
    cat >&2 <<'EOF'
usage: submit_array_processing.sh [--root DIR] [--rscript PATH] [--dependency JOBID] [--study STUDY --platform 450K|EPIC]

Submit minfi beta-calling jobs by study and platform for paired-validation-v2.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            root="$2"
            shift 2
            ;;
        --rscript)
            rscript="$2"
            shift 2
            ;;
        --dependency)
            dependency="$2"
            shift 2
            ;;
        --study)
            study_filter="$2"
            shift 2
            ;;
        --platform)
            platform_filter="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ -n "${study_filter}" || -n "${platform_filter}" ]]; then
    [[ -n "${study_filter}" && -n "${platform_filter}" ]] || { echo "--study and --platform must be provided together" >&2; exit 1; }
fi

mkdir -p "${root}/logs/paired_validation_v2/array"

submit_one() {
    local study="$1"
    local platform="$2"
    dependency_args=()
    if [[ -n "${dependency}" ]]; then
        dependency_args=(--dependency="afterok:${dependency}")
    fi
    sbatch \
        "${dependency_args[@]}" \
        --job-name="pv2_array_${study}_${platform}" \
        -p highmem \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=12:00:00 \
        --out="${root}/logs/paired_validation_v2/array/%x.%j.o" \
        --error="${root}/logs/paired_validation_v2/array/%x.%j.e" \
        --wrap="export PATH=$(dirname "${rscript}"):\$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1; ${rscript} ${root}/scripts/paired_validation_v2/call_array_methylation.R --root ${root} --study ${study} --platform ${platform}"
}

if [[ -n "${study_filter}" ]]; then
    submit_one "${study_filter}" "${platform_filter}"
else
    submit_one pgpuk 450K
    submit_one pidsley 450K
    submit_one pidsley EPIC
    submit_one epicv2_eval EPIC
fi
