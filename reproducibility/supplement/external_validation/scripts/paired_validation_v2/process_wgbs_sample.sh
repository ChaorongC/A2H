#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <WGBS_ID> [--root DIR] [--manifest TSV] [--protocol-flags TSV]" >&2
    exit 1
fi

sample_id="$1"
shift

root="external_validation"
manifest="${root}/metadata/paired_validation_v2/paired_wgbs_run_manifest.tsv"
protocol_flags="${root}/metadata/paired_validation_v2/bsmap_protocol_flags.tsv"

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
        --protocol-flags)
            protocol_flags="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

sample_fastq_dir="${root}/fastq/paired_validation_v2/${sample_id}"
trim_dir="${root}/tmp/paired_validation_v2/trim_galore/${sample_id}"
bam_dir="${root}/work/paired_validation_v2/bam"
traditional_dir="${root}/results/paired_validation_v2/wgbs/traditional"
chalm_dir="${root}/results/paired_validation_v2/wgbs/chalm"
status_dir="${root}/logs/paired_validation_v2/process"
thread_count="${SLURM_CPUS_PER_TASK:-8}"
python_bin="python"

module load trimgalore/0.6.7
module load samtools/1.10

bsmap_bin="bsmap"
chalm_py="tools/CHALM_fast/src/CHALM-fast.py"
legacy_samtools_dir="${root}/scripts/samtools_legacy_compat"
reference_fa="reference/hg38.fa"

mkdir -p "${trim_dir}" "${bam_dir}" "${traditional_dir}" "${chalm_dir}" "${status_dir}"

output_has_data() {
    local path="$1"
    [[ -s "${path}" ]] || return 1
    awk 'END { exit (NR > 1 ? 0 : 1) }' "${path}"
}

study="$(awk -F'\t' -v sample_id="${sample_id}" 'NR > 1 && $2 == sample_id {print $1; exit}' "${manifest}")"
if [[ -z "${study}" ]]; then
    echo "No manifest rows found for ${sample_id}" >&2
    exit 1
fi
mapfile -t bsmap_extra_flags < <(awk -F'\t' -v study="${study}" 'NR > 1 && $1 == study {print $2}' "${protocol_flags}")
if [[ "${#bsmap_extra_flags[@]}" -eq 0 || -z "${bsmap_extra_flags[0]}" ]]; then
    echo "No BSMAP flags found for study ${study} in ${protocol_flags}" >&2
    exit 1
fi

for run_id in $(awk -F'\t' -v sample_id="${sample_id}" 'NR > 1 && $2 == sample_id {print $4}' "${manifest}"); do
    if ! awk -F'\t' -v sample_id="${sample_id}" -v run_id="${run_id}" 'NR > 1 && $2 == sample_id && $4 == run_id && $10 == "complete" {found=1} END {exit found ? 0 : 1}' "${manifest}"; then
        echo "Run ${run_id} for ${sample_id} is not marked complete in ${manifest}" >&2
        exit 1
    fi
done

r1_fastq="${sample_fastq_dir}/${sample_id}_R1.fastq.gz"
r2_fastq="${sample_fastq_dir}/${sample_id}_R2.fastq.gz"
unsorted_bam="${bam_dir}/${sample_id}_unsrt.bam"
mapped_unsorted_bam="${bam_dir}/${sample_id}_unsrt_mapped.bam"
sorted_bam="${bam_dir}/${sample_id}_srt_mapped.bam"
traditional_out="${traditional_dir}/${sample_id}_methRatio.txt"
chalm_out="${chalm_dir}/${sample_id}_CHALM.txt"

if [[ -s "${sorted_bam}" && -s "${sorted_bam}.bai" ]] && samtools quickcheck "${sorted_bam}"; then
    echo "Skipping alignment for ${sample_id}; reusing ${sorted_bam}"
else
    "${root}/scripts/paired_validation_v2/merge_wgbs_fastqs.sh" "${sample_id}" --root "${root}" --manifest "${manifest}"

    trim_galore --paired \
        --2colour 20 \
        --clip_R1 10 \
        --clip_R2 20 \
        --three_prime_clip_R1 10 \
        --three_prime_clip_R2 10 \
        --cores "${thread_count}" \
        -o "${trim_dir}" \
        "${r1_fastq}" \
        "${r2_fastq}"

    "${bsmap_bin}" \
        -a "${trim_dir}/${sample_id}_R1_val_1.fq.gz" \
        -b "${trim_dir}/${sample_id}_R2_val_2.fq.gz" \
        -d "${reference_fa}" \
        ${bsmap_extra_flags[0]} \
        -o "${unsorted_bam}" \
        -p "${thread_count}"

    samtools view -F 268 -b "${unsorted_bam}" > "${mapped_unsorted_bam}"
    rm -f "${unsorted_bam}"

    samtools sort -@ "${thread_count}" -o "${sorted_bam}" "${mapped_unsorted_bam}"
    rm -f "${mapped_unsorted_bam}"
    samtools index -@ "${thread_count}" "${sorted_bam}"
fi

if output_has_data "${traditional_out}"; then
    echo "Skipping methRatio extraction for ${sample_id}; reusing ${traditional_out}"
else
    "${python_bin}" "${chalm_py}" trad \
        -d "${reference_fa}" \
        -x CG -i no-action \
        -m 4 -p -r -u -g -q \
        -s "${legacy_samtools_dir}" \
        -o "${traditional_out}" \
        "${sorted_bam}"
fi

if output_has_data "${chalm_out}"; then
    echo "Skipping CHALM extraction for ${sample_id}; reusing ${chalm_out}"
else
    "${python_bin}" "${chalm_py}" trad \
        -d "${reference_fa}" \
        -x CG -i no-action \
        -m 4 -p -r -u -g -q -l 1 \
        -s "${legacy_samtools_dir}" \
        -o "${chalm_out}" \
        "${sorted_bam}"
fi

samtools quickcheck "${sorted_bam}"
[[ -s "${sorted_bam}.bai" ]]
output_has_data "${traditional_out}"
output_has_data "${chalm_out}"

touch "${status_dir}/${sample_id}.complete"

"${python_bin}" - "${manifest}" "${sample_id}" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
sample_id = sys.argv[2]
rows = []
with manifest_path.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = reader.fieldnames
    for row in reader:
        if row["wgbs_id"] == sample_id:
            row["process_status"] = "complete"
        rows.append(row)
with manifest_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
PY

echo "Processing complete for ${sample_id}"
