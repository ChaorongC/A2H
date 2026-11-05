#!/usr/bin/env bash
# Step 0: per-sample WGBS processing for one atlas sample.
# FASTQ (2 lanes, paired-end) -> Trim Galore -> BASAL -> samtools -> CHALM.py
# Outputs: <out_dir>/3.methratio/<sample_id>_methRatio.txt
#          <out_dir>/3.chalm/<sample_id>_CHALM.txt
#
# Usage:
#   bash 00_process_wgbs.sh <sample_id> <fastq_dir> <out_dir> <hg38.fa> <CHALM_src_dir> <samtools-0.1.19_dir> [threads]
#
# <fastq_dir> must contain <...sample_id...>_L001_R1.fastq.gz, _L001_R2, _L002_R1 and _L002_R2.
# Tools on PATH: trim_galore (0.6.7, with cutadapt), basal (1.8), samtools (1.10), python.
set -euo pipefail

sample_id=$1
fastq_dir=$2
out_dir=$3
hg38_ref=$4
chalm_src=$5
samtools_0119=$6
threads=${7:-40}

mkdir -p "$out_dir"/{0.temp,1.trimgalore,2.align,3.methratio,3.chalm}

# Concatenate the two sequencing lanes
find_fastq() { find "$fastq_dir" -maxdepth 2 -type f -name "*${sample_id}*_$1.fastq.gz" -printf '%p\n'; }
cat "$(find_fastq L001_R1)" "$(find_fastq L002_R1)" > "$out_dir/0.temp/${sample_id}_R1.fastq.gz"
cat "$(find_fastq L001_R2)" "$(find_fastq L002_R2)" > "$out_dir/0.temp/${sample_id}_R2.fastq.gz"

# 1. Quality and adapter trimming
trim_galore --paired \
    --2colour 20 \
    --clip_R1 10 \
    --clip_R2 20 \
    --three_prime_clip_R1 10 \
    --three_prime_clip_R2 10 \
    --cores "$threads" \
    -o "$out_dir/1.trimgalore/$sample_id" \
    "$out_dir/0.temp/${sample_id}_R1.fastq.gz" \
    "$out_dir/0.temp/${sample_id}_R2.fastq.gz"

# 2. Paired-end bisulfite alignment to hg38 with BASAL (C>T conversion, default parameters).
#    BASAL is built on BSMAP, and this call is equivalent to `bsmap -R`, which was used for the atlas samples.
basal -a "$out_dir/1.trimgalore/$sample_id/${sample_id}_R1_val_1.fq.gz" \
    -b "$out_dir/1.trimgalore/$sample_id/${sample_id}_R2_val_2.fq.gz" \
    -d "$hg38_ref" \
    -M C:T \
    -R \
    -o "$out_dir/2.align/${sample_id}_unsrt.bam" \
    -p "$threads"

# 3. Keep mapped primary alignments (-F 268: drop unmapped, mate-unmapped and secondary), sort, index
samtools view -F 268 -b "$out_dir/2.align/${sample_id}_unsrt.bam" > "$out_dir/2.align/${sample_id}_unsrt_mapped.bam"
samtools sort -@ "$threads" -o "$out_dir/2.align/${sample_id}_srt_mapped.bam" "$out_dir/2.align/${sample_id}_unsrt_mapped.bam"
samtools index -@ "$threads" "$out_dir/2.align/${sample_id}_srt_mapped.bam"

# CHALM.py trad flags: CpG context only (-x CG), no C/T SNP correction (-i no-action), depth >= 4 (-m 4),
# properly paired (-p), duplicates removed (-r), unique mappings (-u), both strands combined (-g).
# 4. Methylation level (methRatio) at CpGs with >= 4 reads
python "$chalm_src/CHALM.py" trad \
    -d "$hg38_ref" \
    -x CG -i no-action \
    -m 4 -p -r -u -g -q \
    -s "$samtools_0119" \
    -o "$out_dir/3.methratio/${sample_id}_methRatio.txt" \
    "$out_dir/2.align/${sample_id}_srt_mapped.bam"

# 5. CHALM at CpGs with >= 4 reads (-l 1: a read counts as methylated if >= 1 CpG on it is methylated)
python "$chalm_src/CHALM.py" trad \
    -d "$hg38_ref" \
    -x CG -i no-action \
    -m 4 -p -r -u -g -q -l 1 \
    -s "$samtools_0119" \
    -o "$out_dir/3.chalm/${sample_id}_CHALM.txt" \
    "$out_dir/2.align/${sample_id}_srt_mapped.bam"
