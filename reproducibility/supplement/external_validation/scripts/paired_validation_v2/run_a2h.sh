#!/bin/bash
set -euo pipefail

root="external_validation"
a2h_root="legacy_model"
python_bin="python"
threads="${SLURM_CPUS_PER_TASK:-4}"
mode="both"
study_filter=""
platform_filter=""

usage() {
    cat >&2 <<'EOF'
usage: run_a2h.sh [--root DIR] [--a2h-root DIR] [--python-bin PATH] [--threads N] [--mode array|wgbs|both] [--study STUDY --platform 450K|EPIC]

Generate A2H outputs for paired-validation v2 in parallel namespaces.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            root="$2"
            shift 2
            ;;
        --a2h-root)
            a2h_root="$2"
            shift 2
            ;;
        --python-bin)
            python_bin="$2"
            shift 2
            ;;
        --threads)
            threads="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
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

cmd=(
    "${python_bin}"
    "${root}/scripts/generate_paired_validation_a2h_outputs.py"
    --root "${root}"
    --namespace paired_validation_v2
    --mode "${mode}"
    --a2h-root "${a2h_root}"
    --python-bin "${python_bin}"
    --threads "${threads}"
)

if [[ -n "${study_filter}" || -n "${platform_filter}" ]]; then
    [[ -n "${study_filter}" && -n "${platform_filter}" ]] || { echo "--study and --platform must be provided together" >&2; exit 1; }
    cmd+=(--study "${study_filter}" --platform "${platform_filter}")
fi

"${cmd[@]}"
