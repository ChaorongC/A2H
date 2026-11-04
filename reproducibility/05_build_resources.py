"""Step 5: build the bundled A2H resource files.

    A2H_coef.npz          featureID, chalm_coef (n, 3), camda_coef (n, 3)
    A2H_cpg_info.npz      featureID, min, max, mean, std, betabinom_a, betabinom_b
    A2H_constant_cpgs.tsv CpGs where the CHALM fit used the constant fallback
                          (methylation level identical in every training sample)

The CpG set and order are those of the CHALM coefficients, sorted by locus ID.
CAMDA coefficients and CpG info are taken for the same CpGs.

Usage:
    python 05_build_resources.py --fit-dir <step 2 out-dir> --cpg-info cpg_info.pkl \
        --probe-map A2H_cpg_probe.hg38.tsv --out-dir <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd

CONSTANT_LIST_PLATFORMS = {"HM450": "HM450.hg38", "EPIC": "EPIC.hg38", "EPICv2": "EPICv2.hg38"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-dir", required=True)
    p.add_argument("--cpg-info", required=True)
    p.add_argument("--probe-map", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    chalm = pd.read_pickle(os.path.join(a.fit_dir, "chalm_coef.pkl")).sort_index()
    loci = chalm.index
    camda = pd.read_pickle(os.path.join(a.fit_dir, "camda_coef.pkl")).loc[loci]
    info = pd.read_pickle(a.cpg_info).loc[loci]
    os.makedirs(a.out_dir, exist_ok=True)

    feature_ids = loci.to_numpy(dtype=str)
    np.savez_compressed(
        os.path.join(a.out_dir, "A2H_coef.npz"),
        featureID=feature_ids,
        chalm_coef=chalm.to_numpy(dtype=np.float64),
        camda_coef=camda.to_numpy(dtype=np.float64),
    )
    np.savez_compressed(
        os.path.join(a.out_dir, "A2H_cpg_info.npz"),
        featureID=feature_ids,
        **{c: info[c].to_numpy(dtype=np.float64) for c in ["min", "max", "mean", "std", "betabinom_a", "betabinom_b"]},
    )

    # Constant-input CpGs: the A2H output there is the intercept, whatever the input.
    fit_info = pd.read_csv(os.path.join(a.fit_dir, "chalm_fit_info.tsv"), sep="\t", index_col="locus")
    constant = fit_info.loc[loci][lambda d: d["constant_input"]].index
    probes = pd.read_csv(a.probe_map, sep="\t")
    table = pd.DataFrame({"n_reference_samples": fit_info.loc[constant, "n_pairs"]}, index=constant.rename("locus"))
    for name, column in CONSTANT_LIST_PLATFORMS.items():
        ids = probes[["loci.hg38", column]].dropna().drop_duplicates()
        ids = ids[ids["loci.hg38"].isin(constant)].groupby("loci.hg38")[column].agg(lambda s: ";".join(sorted(s)))
        table[f"{name}_probe"] = ids.reindex(table.index)
    table["A2H_CHALM_output"] = chalm.loc[constant, 2]
    table["A2H_CAMDA_output"] = camda.loc[constant, 2]
    table.reset_index().to_csv(os.path.join(a.out_dir, "A2H_constant_cpgs.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
