"""Step 4: map array probe IDs to the fitted CpG loci (A2H_cpg_probe.hg38/hg19.tsv).

Starting from the CpGs with CHALM coefficients (sorted), each platform's probe
IDs are attached by a left join on the hg38 locus ID. A locus targeted by
several probes appears in several rows.
    hg38 manifests: locus = CpG_chrm + "_" + (CpG_beg + 1)
    hg19 manifests: CpG_beg is lifted over to hg38 with the `liftover` package
        (get_lifter("hg19", "hg38", one_based=True)); locus = CpG_chrm + "_" +
        (lifted position + 1). Probes that cannot be lifted over are dropped.

Usage:
    python 04_probe_maps.py --chalm-coef chalm_coef.pkl \
        --manifest-dir <dir with <platform>.<build>.manifest.tsv> --out-dir <dir>

Outputs: A2H_cpg_probe.hg38.tsv, A2H_cpg_probe.hg19.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd

PLATFORMS = {
    "hg38": ["EPICv2", "EPIC+", "EPIC", "HM450"],
    "hg19": ["EPIC", "EPICv2", "HM450"],
}


def hg38_loci(manifest):
    m = pd.read_table(manifest, usecols=["Probe_ID", "CpG_chrm", "CpG_beg", "CpG_end"]).dropna(axis=0)
    m["featureID"] = m["CpG_chrm"] + "_" + (m["CpG_beg"].astype(int) + 1).astype(str)
    return m


def hg19_loci(manifest, converter):
    m = pd.read_table(manifest, usecols=["Probe_ID", "CpG_chrm", "CpG_beg"]).dropna(axis=0)
    feature_ids = []
    for _, locus in m[["CpG_chrm", "CpG_beg"]].iterrows():
        try:
            feature_ids.append(
                locus["CpG_chrm"] + "_" + str(converter[locus["CpG_chrm"]][locus["CpG_beg"]][0][1] + 1)
            )
        except Exception:
            feature_ids.append(np.nan)
    m["featureID"] = feature_ids
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chalm-coef", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    loci = pd.DataFrame({"featureID": pd.read_pickle(a.chalm_coef).sort_index().index})
    os.makedirs(a.out_dir, exist_ok=True)

    converter = None
    for build, platforms in PLATFORMS.items():
        table = loci.copy()
        for platform in platforms:
            column = f"{platform}.{build}"
            manifest = os.path.join(a.manifest_dir, f"{column}.manifest.tsv")
            if build == "hg38":
                m = hg38_loci(manifest)
            else:
                if converter is None:
                    from liftover import get_lifter

                    converter = get_lifter("hg19", "hg38", one_based=True)
                m = hg19_loci(manifest, converter)
            table = table.merge(m[["featureID", "Probe_ID"]], on="featureID", how="left").rename(
                columns={"Probe_ID": column}
            )
        table.rename(columns={"featureID": "loci.hg38"}).to_csv(
            os.path.join(a.out_dir, f"A2H_cpg_probe.{build}.tsv"), sep="\t", index=False, header=True
        )


if __name__ == "__main__":
    main()
