#!/bin/bash
# Recreate legacy_model/: the A2H.py version and the pickle resources that the revision analyses ran with
# (A2H commit e66ac58, 2026-05-15). A2H v0.5.0 reads .npz resources only; the coefficient values are the same.
# Run from reproducibility/supplement/ inside a git clone of the A2H repository.
set -euo pipefail
out=${1:-legacy_model}
mkdir -p "$out"
for f in A2H.py A2H_coef.pkl A2H_cpg_info.pkl A2H_cpg_probe.hg38.tsv A2H_cpg_probe.hg19.tsv; do
    git show "e66ac58:$f" > "$out/$f"
done
echo "wrote $out/"
