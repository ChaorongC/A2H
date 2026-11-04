#!/bin/bash
# Build per-sample symlink roots so the unchanged src/r2_2_e2e3_summarize.py can run as a Slurm array
# (one process per atlas sample or PGP-UK participant instead of ~9 h in one process).
# Each shard root holds <experiment>/<sample> -> the real condition directory; the summarizer
# writes that shard's EPIC parquets to <shard>/epic and its tables to <shard>/out.
# Usage (from R/): bash results/r2_2_e2e3_20261015/build_summary_shards.sh <namespace> [sample ...]
#   namespace: new directory name under ROOT (refused if it exists); samples default to all.
set -euo pipefail
ROOT=work/r2_2_e2e3_20261015
NS=$ROOT/${1:?namespace}; shift
[[ -e $NS ]] && { echo "Refusing to reuse $NS" >&2; exit 1; }
samples=("$@")
if [[ ${#samples[@]} -eq 0 ]]; then
    mapfile -t samples < <(ls "$ROOT/e2" "$ROOT/e3" "$ROOT/pgpuk_c" | grep -E '^(Z|PGPUK)' | sort -u)
fi
mkdir -p "$NS/shards"
manifest=$NS/shards.tsv
printf 'task\tshard\troot\n' > "$manifest"
task=0
for sample in "${samples[@]}"; do
    shard=$NS/shards/$sample
    for experiment in e2 e3 pgpuk_c; do
        [[ -d $ROOT/$experiment/$sample ]] || continue
        mkdir -p "$shard/$experiment"
        ln -s "$ROOT/$experiment/$sample" "$shard/$experiment/$sample"
    done
    [[ -d $shard ]] || { echo "No condition directories for $sample" >&2; exit 1; }
    printf '%s\t%s\t%s\n' "$task" "$sample" "$shard" >> "$manifest"
    task=$((task + 1))
done
echo "$manifest: $task shards"
