"""
 # @ Author: Chaorong Chen (c.chen@uci.edu)
 # @ Create Time: 2024-10-29 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2026-05-15 16:56:09
 # @ Description: A2H v0.4.0
 """

import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
import warnings
import sys
from datetime import datetime
import os
import numpy as np
from scipy.stats import betabinom
from os.path import isdir

warnings.filterwarnings("ignore")
version = "0.4.0"
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    action="store",
    dest="input",
    type=str,
    help="Input methyaltion file with first column as probe ID and first row as sample ID.",
)
parser.add_argument(
    "-o",
    action="store",
    dest="output",
    default=".",
    type=str,
    help="Output directory for A2H transformation.",
)
parser.add_argument(
    "--prefix",
    action="store",
    default="A2H",
    type=str,
    help="Prefix for A2H transformation output.",
)

parser.add_argument(
    "-c",
    action="store",
    dest="coef",
    type=str,
    default=os.path.dirname(os.path.realpath(__file__)) + "/A2H_coef.pkl",
    help="A2H coefficients provided in .pkl format.",
)

parser.add_argument(
    "--cpg-info",
    action="store",
    dest="cpg_info_path",
    default=os.path.dirname(os.path.realpath(__file__)) + "/A2H_cpg_info.pkl",
    type=str,
    help="CpG loci and their infomation in training data.",
)
parser.add_argument("-t", action="store", dest="thread", default=-1, type=int)
parser.add_argument(
    "--ref",
    action="store",
    dest="ref",
    default="hg38",
    choices=["hg38", "hg19"],
    type=str,
)
parser.add_argument(
    "--beadchip",
    action="store",
    default="EPIC",
    choices=["450k", "EPIC", "HM450", "850k"],
)

parser.add_argument(
    "--cl",
    action="store_true",
    dest="cl",
    help="Output confidence level for each CpG.",
)
parser.add_argument(
    "--probe",
    action="store_true",
    dest="probe",
    help="Use probe ID for output index, otherwise use CpG loci.",
)
parser.add_argument(
    "--verbose",
    action="store",
    dest="verbose",
    type=int,
    help="Print A2H transformation progress.",
    default=1,
)

parser.add_argument(
    "--chalm",
    action="store_true",
    dest="chalm",
    help="Output CHALM.",
)

parser.add_argument(
    "--camda",
    action="store_true",
    dest="camda",
    help="Output CAMDA.",
)

parser.add_argument(
    "--filter",
    action="store",
    dest="filter",
    default=None,
    type=float,
    help="The minimum confidence level to keep the transformed values, if assigned. Overwriting --cl.",
)

parser.add_argument(
    "--exclude-constant",
    action="store_true",
    dest="exclude_constant",
    help="Drop the 501 constant-input CpGs (KDE-fallback sites listed in A2H_constant_cpgs.tsv) from all outputs.",
)

parser.add_argument(
    "--replace",
    action="store",
    dest="replace",
    help="If assigned, replace the filtered and missing value with the assigned value. Otherwise, replace the values with NaNs.",
)

print(
    f"\n\n==========A2H v{version}==========\n\n",
)


def chalm_transform(x):
    feature = x[0]
    coef = chalm_coef_by_feature[feature]
    if isinstance(coef, (int, float)):
        return np.array([coef] * (len(x) - 1))
    else:
        return np.polyval(coef, x[1:])

def camda_transform(x):
    feature = x[0]
    coef = camda_coef_by_feature[feature]
    if isinstance(coef, (int, float)):
        return np.array([coef] * (len(x) - 1))
    else:
        return np.polyval(coef, x[1:])


def beta_binom_ci(x, a, b):
    """
    Two-tailed confidence level
    x: 0 <= methylation ratio <=100
    """
    n = 100
    return 1 - 2 * abs(0.5 - betabinom.cdf(x, n, a, b))


def A2H_CI(x):
    info = cpg_info.loc[x[0]]
    return beta_binom_ci(
        [_ if np.isnan(_) else int(_ * 100) for _ in x[1:]],
        info["betabinom_a"],
        info["betabinom_b"],
    )


args = parser.parse_args()

# Parameters check
if not isdir(args.output):
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Error: Output directory does not exist. Please double check the directory path.",
    )
    sys.exit(1)
else:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Output directory assigned: %s" % args.output,
    )


print(
    "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Prefix for A2H transformation output assigned: %s" % args.prefix,
)


if args.filter is not None:
    args.cl = True
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Minimum confidence level assigned: %s" % args.filter,
    )
if args.cl and not args.cpg_info_path:
    print("Error: --cpg-info is required when --cl is enabled.")
    sys.exit(1)
if (args.beadchip.upper() == "EPIC") or (args.beadchip.lower() == "850k"):
    beadchip_version = "EPIC.%s" % args.ref
elif (args.beadchip.upper() == "HM450") or (args.beadchip.lower() == "450k"):
    beadchip_version = "HM450.%s" % args.ref
else:
    print("Error: Beadchip/reference version is not supported.")
    sys.exit(1)

if not args.chalm and not args.camda:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Error: --chalm or --camda is required.",
    )
    sys.exit(1)


print(
    "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Reading coefficient file %s" % args.coef,
)

try:
    allProbe_result = pd.read_pickle(args.coef)
except ModuleNotFoundError as exc:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Error: reading coefficient file failed because Python module '%s' is not installed."
        % exc.name,
    )
    sys.exit(1)

if "featureID" not in allProbe_result.columns:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Error: coefficient file must contain featureID column.",
    )
    sys.exit(1)

allProbe_result = allProbe_result.drop_duplicates(subset="featureID", keep="first")

chalm_coef_column = "chalm_coef" if "chalm_coef" in allProbe_result.columns else "coef"
if chalm_coef_column not in allProbe_result.columns:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Error: coefficient file must contain chalm_coef or coef column for CHALM output.",
    )
    sys.exit(1)

chalm_coef_by_feature = allProbe_result.set_index("featureID")[chalm_coef_column].to_dict()

camda_coef_by_feature = None
if args.camda:
    camda_coef_column = "camda_coef" if "camda_coef" in allProbe_result.columns else None
    if camda_coef_column is None and len(allProbe_result.columns) >= 3:
        camda_coef_column = allProbe_result.columns[2]
    if camda_coef_column not in allProbe_result.columns:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Error: --camda requires CAMDA coefficients, but this coefficient file only contains CHALM coefficients.",
        )
        sys.exit(1)
    camda_coef_by_feature = allProbe_result.set_index("featureID")[camda_coef_column].to_dict()


print(
    "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Reading input %s" % args.input,
)

methylation_input = pd.read_csv(args.input, sep="\t")
cpg_probe_info = (
    pd.read_table(
        os.path.dirname(os.path.realpath(__file__))
        + "/A2H_cpg_probe.%s.tsv" % args.ref,
        usecols=["loci.hg38", beadchip_version],
    )
    .dropna()
    .set_index(beadchip_version)
    .drop_duplicates()
)

methylation_input = methylation_input[
    methylation_input.iloc[:, 0].isin(cpg_probe_info.index)
]
if args.probe:
    methylation_probeid = methylation_input.iloc[:, 0].copy().values

methylation_input.iloc[:, 0] = cpg_probe_info.loc[methylation_input.iloc[:, 0]].values

if args.exclude_constant:
    constant_loci = pd.read_table(
        os.path.dirname(os.path.realpath(__file__)) + "/A2H_constant_cpgs.tsv",
        usecols=["locus"],
    )["locus"]
    keep = ~methylation_input.iloc[:, 0].isin(constant_loci).values
    if args.probe:
        methylation_probeid = methylation_probeid[keep]
    methylation_input = methylation_input[keep]
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Excluded %d constant-input CpGs." % (~keep).sum(),
    )

print(
    "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "A2H transformation starts.",
)
if args.chalm:
    chalm_output = pd.DataFrame(
        Parallel(n_jobs=args.thread, verbose=args.verbose, backend="multiprocessing")(
            delayed(chalm_transform)(_) for _ in methylation_input.values
        ),
        columns=methylation_input.columns[1:],
        index=methylation_probeid if args.probe else methylation_input.iloc[:, 0],
    )
    chalm_output.to_csv(args.output + f"/{args.prefix}.chalm.tsv", sep="\t")
    
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H transformation completed, saved to %s" % args.output
        + f"/{args.prefix}.chalm.tsv",
    )

if args.camda:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CAMDA transformation starts.",
    )
    camda_output = pd.DataFrame(
        Parallel(n_jobs=args.thread, verbose=args.verbose, backend="multiprocessing")(
            delayed(camda_transform)(_) for _ in methylation_input.values
        ),
        columns=methylation_input.columns[1:],
        index=methylation_probeid if args.probe else methylation_input.iloc[:, 0],
    )
    
    camda_output.to_csv(args.output + f"/{args.prefix}.camda.tsv", sep="\t")
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CAMDA transformation completed, saved to %s" % args.output
        + f"/{args.prefix}.camda.tsv",
    )

# output confidence level for each cpg
if args.cl:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H transformation confidence level calculation starts.",
    )
    cpg_info = pd.read_pickle(args.cpg_info_path)
    methylation_input_ci = pd.DataFrame(
        Parallel(n_jobs=-1, verbose=0, backend="multiprocessing")(
            delayed(A2H_CI)(_) for _ in methylation_input.values
        ),
        columns=methylation_input.columns[1:],
        index=methylation_probeid if args.probe else methylation_input.iloc[:, 0],
    ).round(4)
    methylation_input_ci.to_csv(
        args.output + f"/{args.prefix}.confidence.tsv", sep="\t"
    )

    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H confidence level saved to %s" % args.output
        + f"/{args.prefix}.confidence.tsv",
    )

# filter output based on confidence level for each cpg
if args.filter is not None:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filtering output based on confidence level assigned: %s" % args.filter,
    )
    methylation_input_ci_mask = methylation_input_ci >= args.filter
    if args.replace:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Replace filtered and missing values with %s" % args.replace,
        )
        if args.chalm:
            chalm_output.where(methylation_input_ci_mask, args.replace).to_csv(
                args.output + f"/{args.prefix}.chalm.filtered.tsv", sep="\t"
            )
        if args.camda:
            camda_output.where(methylation_input_ci_mask, args.replace).to_csv(
                args.output + f"/{args.prefix}.camda.filtered.tsv", sep="\t"
            )
    else:
        if args.chalm:
            chalm_output.where(methylation_input_ci_mask).to_csv(
                args.output + f"/{args.prefix}.chalm.filtered.tsv", sep="\t"
            )
        if args.camda:
            camda_output.where(methylation_input_ci_mask).to_csv(
                args.output + f"/{args.prefix}.camda.filtered.tsv", sep="\t"
            )
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filtering done, saved to %s" % args.output + f"/{args.prefix}.*.filtered.tsv",
    )
