#!/bin/bash
set -euo pipefail

root="external_validation"
dependency=""
study_filter=""
platform_filter=""

usage() {
    cat >&2 <<'EOF'
usage: submit_wgbs_probe_matrices.sh [--root DIR] [--dependency JOBID[:JOBID...]] [--study STUDY --platform 450K|EPIC]

Submit paired-validation-v2 WGBS probe-space matrix builders for direct array
comparison.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            root="$2"
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

mkdir -p "${root}/logs/paired_validation_v2/probe_space"
dependency_args=()
if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
fi

submit_one() {
    local study="$1"
    local platform="$2"
    sbatch \
        "${dependency_args[@]}" \
        --job-name="pv2_probe_${study}_${platform}" \
        -p highmem \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=32G \
        --time=6:00:00 \
        --out="${root}/logs/paired_validation_v2/probe_space/%x.%j.o" \
        --error="${root}/logs/paired_validation_v2/probe_space/%x.%j.e" \
        --wrap="python ${root}/scripts/paired_validation_v2/build_wgbs_probe_matrices.py --root ${root} --study ${study} --platform ${platform}"
}

if [[ -n "${study_filter}" ]]; then
    submit_one "${study_filter}" "${platform_filter}"
else
    submit_one pgpuk 450K
    submit_one pidsley 450K
    submit_one pidsley EPIC
    submit_one epicv2_eval EPIC
fi
