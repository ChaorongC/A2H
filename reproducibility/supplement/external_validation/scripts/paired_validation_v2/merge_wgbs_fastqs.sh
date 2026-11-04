#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <WGBS_ID> [--root DIR] [--manifest TSV]" >&2
    exit 1
fi

sample_id="$1"
shift

root="external_validation"
manifest="${root}/metadata/paired_validation_v2/paired_wgbs_run_manifest.tsv"

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
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

sample_fastq_dir="${root}/fastq/paired_validation_v2/${sample_id}"
mkdir -p "${sample_fastq_dir}"

mapfile -t r1_files < <(awk -F'\t' -v root="${root}" -v sample_id="${sample_id}" 'NR > 1 && $2 == sample_id {print root "/raw/paired_validation_v2/fastq_run/" $4 "/" $4 "_1.fastq.gz"}' "${manifest}")
mapfile -t r2_files < <(awk -F'\t' -v root="${root}" -v sample_id="${sample_id}" 'NR > 1 && $2 == sample_id {print root "/raw/paired_validation_v2/fastq_run/" $4 "/" $4 "_2.fastq.gz"}' "${manifest}")

if [[ "${#r1_files[@]}" -eq 0 ]]; then
    echo "No runs found for ${sample_id} in ${manifest}" >&2
    exit 1
fi

for file in "${r1_files[@]}" "${r2_files[@]}"; do
    [[ -s "${file}" ]] || { echo "Missing run FASTQ ${file}" >&2; exit 1; }
done

cat "${r1_files[@]}" > "${sample_fastq_dir}/${sample_id}_R1.fastq.gz"
cat "${r2_files[@]}" > "${sample_fastq_dir}/${sample_id}_R2.fastq.gz"
gzip -t "${sample_fastq_dir}/${sample_id}_R1.fastq.gz" "${sample_fastq_dir}/${sample_id}_R2.fastq.gz"

echo "Merged FASTQs ready for ${sample_id}"
