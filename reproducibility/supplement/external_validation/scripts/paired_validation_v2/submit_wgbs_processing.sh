#!/bin/bash
set -euo pipefail

root="external_validation"
manifest="${root}/metadata/paired_validation_v2/paired_wgbs_run_manifest.tsv"
dependency_map=""
dependency=""
study_filter=""

usage() {
    cat >&2 <<'EOF'
usage: submit_wgbs_processing.sh [--root DIR] [--manifest TSV] [--dependency-map TSV] [--dependency JOBID[:JOBID...]] [--study STUDY]

Submit one WGBS processing job per paired-validation-v2 WGBS ID. If a dependency
map is provided, it must contain two tab-separated columns: run_accession and
slurm_job_id. The optional global dependency is added to every processing job.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            root="$2"
            shift 2
            ;;
        --manifest)
            manifest="$2"
            shift 2
            ;;
        --dependency-map)
            dependency_map="$2"
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

mkdir -p "${root}/logs/paired_validation_v2/process"

mapfile -t samples < <(awk -F'\t' -v study_filter="${study_filter}" 'NR > 1 && (study_filter == "" || $1 == study_filter) {print $2}' "${manifest}" | sort -u)

dependency_for_sample() {
    local sample="$1"
    [[ -n "${dependency_map}" ]] || return 0
    awk -F'\t' -v sample="${sample}" '
        NR == FNR && NR > 1 {job[$1] = $2; next}
        NR > 1 && $2 == sample && $4 in job {deps = deps (deps ? ":" : "") job[$4]}
        END {print deps}
    ' "${dependency_map}" "${manifest}"
}

for sample in "${samples[@]}"; do
    deps="$(dependency_for_sample "${sample}")"
    dependency_args=()
    if [[ -n "${dependency}" ]]; then
        dependency_args=(--dependency="afterok:${dependency}")
    fi
    if [[ -n "${deps}" ]]; then
        if [[ -n "${dependency}" ]]; then
            dependency_args=(--dependency="afterok:${dependency}:${deps}")
        else
            dependency_args=(--dependency="afterok:${deps}")
        fi
    fi
    sbatch \
        "${dependency_args[@]}" \
        --job-name="pv2_proc_${sample}" \
        -p highmem \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=20 \
        --mem=160G \
        --time=5-00:00:00 \
        --out="${root}/logs/paired_validation_v2/process/%x.%j.o" \
        --error="${root}/logs/paired_validation_v2/process/%x.%j.e" \
        --wrap="${root}/scripts/paired_validation_v2/process_wgbs_sample.sh ${sample} --root ${root} --manifest ${manifest}"
done
