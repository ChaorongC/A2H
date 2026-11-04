"""Step 3: per-CpG summaries of the training methylation levels (A2H CpG info).

For the CpGs fitted for CHALM (at least --min-obs samples with a CHALM value),
summarize the WGBS methylation level across atlas samples:
    min, max, mean, std (missing values skipped; std with ddof = 1)
    betabinom_a, betabinom_b: maximum-likelihood beta-binomial parameters
        (n = 100) for the methylation levels expressed as integer percentages
        (int(100 * level), i.e. truncated), fitted by Nelder-Mead starting
        from a = b = mean percentage.
A2H uses betabinom_a and betabinom_b to compute the confidence level of an
input beta value.

Usage:
    python 03_cpg_info.py --chalm merged_CHALM.pkl --methratio merged_methratio.pkl \
        --out cpg_info.pkl [--threads N]

Output: DataFrame (pandas pickle), index = CpG (sorted, named "loci.hg38"),
columns min, max, mean, std, betabinom_a, betabinom_b.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import betabinom

N = 100


def neg_log_likelihood(params, k, n):
    alpha, beta = params
    return -betabinom.logpmf(k, n, alpha, beta).sum()


def beta_binom_fit(x):
    init_guess = [np.mean(x)] * 2
    result = minimize(
        neg_log_likelihood,
        init_guess,
        args=(x, N),
        method="Nelder-Mead",
        options={"maxiter": 1000, "maxfev": 2000},
    )
    return result.x[0], result.x[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chalm", required=True)
    p.add_argument("--methratio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-obs", type=int, default=50)
    p.add_argument("--threads", type=int, default=-1)
    a = p.parse_args()

    chalm = pd.read_pickle(a.chalm)
    meth = pd.read_pickle(a.methratio)
    chalm = chalm.loc[chalm.isna().sum(axis=1) <= chalm.shape[1] - a.min_obs]
    meth = meth.loc[chalm.index].sort_index()

    info = pd.DataFrame(
        [meth.min(axis=1), meth.max(axis=1), meth.mean(axis=1), meth.std(axis=1)],
        index=["min", "max", "mean", "std"],
    ).T

    ab = Parallel(n_jobs=a.threads, verbose=1, backend="multiprocessing")(
        delayed(beta_binom_fit)(row.dropna().astype(int)) for _, row in (meth * 100).iterrows()
    )
    info["betabinom_a"] = [x[0] for x in ab]
    info["betabinom_b"] = [x[1] for x in ab]
    info.index.name = "loci.hg38"
    info.to_pickle(a.out)


if __name__ == "__main__":
    main()
