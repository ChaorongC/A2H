"""Step 2: fit the per-CpG A2H polynomials for CHALM and CAMDA.

For each CpG, the target (CHALM, or CAMDA = CHALM - methylation level) is
regressed on the WGBS methylation level across the atlas samples with a
degree-2 polynomial (numpy.polyfit). Samples where either value is missing are
dropped. If the methylation level is identical in every remaining sample, no
polynomial can be fitted; the coefficients are then [0, 0, c], where c is the
densest target value: the argmax of a Gaussian kernel density estimate
(bandwidth 0.05) evaluated on 10 evenly spaced points in [0, 1].

CpG filter: CHALM is fitted at CpGs with at least --min-obs (50) samples with a
CHALM value; CAMDA at CpGs with at least --min-obs samples with a CAMDA value.

Usage:
    python 02_fit_coefficients.py --chalm merged_CHALM.pkl \
        --methratio merged_methratio.pkl --out-dir <dir> [--threads N]

Outputs (in --out-dir):
    chalm_coef.pkl, camda_coef.pkl: DataFrame, index = CpG, columns 0, 1, 2 =
        polynomial coefficients, highest degree first (numpy.polyval order).
    chalm_fit_info.tsv, camda_fit_info.tsv: per CpG, the number of samples used
        in the fit (n_pairs) and whether the constant fallback was used
        (constant_input).
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

DEGREE = 2
BINS = 10
BANDWIDTH = 0.05

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


def densest_value(y):
    kde = KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH).fit(y.reshape(-1, 1))
    log_dens = kde.score_samples(np.linspace(0, 1, BINS).reshape(-1, 1))
    return np.linspace(0, 1, BINS)[np.exp(log_dens).argmax()]


def fit_cpg(m, y):
    temp = pd.DataFrame([m, y]).dropna(axis=1).T.sort_values(by=0).T.values
    x_d, y_d = temp[0], temp[1]
    constant_input = all(v == x_d[0] for v in x_d)
    if constant_input:
        coef = np.array([0] * DEGREE + [densest_value(y_d)])
    else:
        coef = np.polyfit(x_d, y_d, DEGREE)
    return coef, len(x_d), constant_input


def fit(methylation, target, threads):
    results = Parallel(n_jobs=threads, verbose=1, backend="threading")(
        delayed(fit_cpg)(m, y) for m, y in zip(methylation.values, target.values)
    )
    coef = pd.DataFrame(np.vstack([r[0] for r in results]), index=methylation.index.values)
    info = pd.DataFrame(
        {"n_pairs": [r[1] for r in results], "constant_input": [r[2] for r in results]},
        index=pd.Index(methylation.index.values, name="locus"),
    )
    return coef, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chalm", required=True)
    p.add_argument("--methratio", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-obs", type=int, default=50)
    p.add_argument("--threads", type=int, default=-1)
    a = p.parse_args()

    chalm = pd.read_pickle(a.chalm)
    meth = pd.read_pickle(a.methratio)
    if not (chalm.index.equals(meth.index) and chalm.columns.equals(meth.columns)):
        raise ValueError("CHALM and methRatio matrices must have the same CpGs and samples.")
    camda = chalm.subtract(meth)
    max_missing = chalm.shape[1] - a.min_obs
    os.makedirs(a.out_dir, exist_ok=True)

    for name, target in (("chalm", chalm), ("camda", camda)):
        target = target.loc[target.isna().sum(axis=1) <= max_missing]
        coef, info = fit(meth.loc[target.index], target, a.threads)
        coef.to_pickle(os.path.join(a.out_dir, f"{name}_coef.pkl"))
        info.to_csv(os.path.join(a.out_dir, f"{name}_fit_info.tsv"), sep="\t")


if __name__ == "__main__":
    main()
