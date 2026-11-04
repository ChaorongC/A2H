"""Step 1: merge per-sample CHALM and methRatio files into CpG x sample matrices.

Only CpGs targeted by an Illumina array are kept: the union of CpGs in all hg38
manifests (HM450, EPIC, EPIC+, EPICv2). A CpG is identified as
"<chrom>_<position>", the 1-based position of the cytosine (manifest CpG_beg + 1).

Usage:
    python 01_merge_matrices.py \
        --chalm-dir <dir with <sample_id>_CHALM.txt> \
        --methratio-dir <dir with <sample_id>_methRatio.txt> \
        --manifest-dir <dir with *.hg38.manifest.tsv> \
        --out-dir <dir> [--threads N] [--samples <file with one sample_id per line>]

Samples are every <sample_id>_CHALM.txt in --chalm-dir unless --samples is given.

Outputs (pandas pickles, rows = CpGs, columns = sample_id):
    merged_CHALM.pkl, merged_methratio.pkl
"""
import argparse
import glob
import os

import pandas as pd
from joblib import Parallel, delayed

CHROMS = ["chr" + str(c) for c in list(range(1, 23)) + ["X", "Y", "M"]]


def array_cpgs(manifest_dir):
    loci = []
    for path in sorted(glob.glob(os.path.join(manifest_dir, "*.hg38.manifest.tsv"))):
        m = pd.read_table(path, usecols=["Probe_ID", "CpG_chrm", "CpG_beg", "CpG_end"]).dropna(axis=0)
        loci.append(m["CpG_chrm"] + "_" + (m["CpG_beg"].astype(int) + 1).astype(str))
    return pd.concat(loci).drop_duplicates().sort_values(ignore_index=True)


def read_ratio(path, sample_id, cpgs):
    d = pd.read_table(path, usecols=["chr", "pos", "ratio"])
    d = d[d["chr"].isin(CHROMS)]
    d.index = d["chr"] + "_" + d["pos"].astype(str)
    return d[d.index.isin(cpgs)]["ratio"].rename(sample_id)


def merge(sample_ids, pattern, cpgs, threads):
    columns = Parallel(n_jobs=threads, backend="multiprocessing", verbose=10)(
        delayed(read_ratio)(pattern % s, s, cpgs) for s in sample_ids
    )
    return pd.concat(columns, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", help="file with one sample_id per line (default: every <sample_id>_CHALM.txt in --chalm-dir)")
    p.add_argument("--chalm-dir", required=True)
    p.add_argument("--methratio-dir", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threads", type=int, default=-1)
    a = p.parse_args()

    if a.samples:
        sample_ids = [s.strip() for s in open(a.samples) if s.strip()]
    else:
        sample_ids = sorted(f[: -len("_CHALM.txt")] for f in os.listdir(a.chalm_dir) if f.endswith("_CHALM.txt"))
    cpgs = array_cpgs(a.manifest_dir)
    os.makedirs(a.out_dir, exist_ok=True)

    merge(sample_ids, os.path.join(a.chalm_dir, "%s_CHALM.txt"), cpgs, a.threads).to_pickle(
        os.path.join(a.out_dir, "merged_CHALM.pkl"))
    merge(sample_ids, os.path.join(a.methratio_dir, "%s_methRatio.txt"), cpgs, a.threads).to_pickle(
        os.path.join(a.out_dir, "merged_methratio.pkl"))


if __name__ == "__main__":
    main()
