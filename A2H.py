"""
 # @ Author: Chaorong Chen (c.chen@uci.edu)
 # @ Create Time: 2024-10-29 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2024-11-5 16:03:10
 # @ Description: A2H v0.3.1
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
version = "0.3.1"
model_dir = os.path.dirname(os.path.realpath(__file__))

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
    default=os.path.join(model_dir, "A2H_chalm_coef.pkl"),
    help="CHALM coefficients provided in .pkl format.",
)
parser.add_argument(
    "--camda-coef",
    action="store",
    dest="camda_coef",
    default=None,
    type=str,
    help=(
        "Direct CAMDA coefficients provided in .pkl format. If assigned, "
        "CAMDA is predicted directly from beta instead of calculated as "
        "CHALM minus beta."
    ),
)
parser.add_argument(
    "--chalm-minmax",
    action="store",
    dest="chalm_minmax",
    default=None,
    type=str,
    help="Optional CHALM min/max file used to clip CHALM output by CpG.",
)
parser.add_argument(
    "--camda-minmax",
    action="store",
    dest="camda_minmax",
    default=None,
    type=str,
    help="Optional CAMDA min/max file used to clip CAMDA output by CpG.",
)

parser.add_argument(
    "--cpg-info",
    action="store",
    dest="cpg_info_path",
    default=os.path.join(model_dir, "A2H_cpg_info.pkl"),
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
    help="Print A2H transformation progress.",
    default=1,
)
parser.add_argument(
    "--camda",
    action="store_true",
    dest="camda",
    help="Output CAMDA.",
)
parser.add_argument(
    "--camda-only",
    action="store_true",
    dest="camda_only",
    help=(
        "Output CAMDA only and skip writing CHALM files. If --camda-coef is "
        "assigned, CAMDA is predicted directly from beta."
    ),
)

parser.add_argument(
    "--filter",
    action="store",
    dest="filter",
    default=None,
    type=float,
    help="The minimum confidence level to keep transformed A2H outputs, if assigned. Overwriting --cl.",
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


coef_lookup = None


def load_coef_lookup(path):
    coef_table = pd.read_pickle(path)
    required_columns = {"featureID", "coef"}
    missing_columns = required_columns - set(coef_table.columns)
    if missing_columns:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Error: Coefficient file %s is missing columns: %s"
            % (path, ", ".join(sorted(missing_columns))),
        )
        sys.exit(1)
    return coef_table.drop_duplicates("featureID").set_index("featureID")["coef"]


def load_minmax(path):
    minmax_table = pd.read_table(path)
    minmax_table.columns = [str(column) for column in minmax_table.columns]
    required_columns = {"min", "max"}
    missing_columns = required_columns - set(minmax_table.columns)
    if missing_columns:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Error: Min/max file %s is missing columns: %s"
            % (path, ", ".join(sorted(missing_columns))),
        )
        sys.exit(1)

    feature_columns = [
        column
        for column in ["featureID", "loci.hg38", "loci", "CpG", "cpg"]
        if column in minmax_table.columns
    ]
    if feature_columns:
        feature_column = feature_columns[0]
    else:
        non_value_columns = [
            column for column in minmax_table.columns if column not in required_columns
        ]
        if not non_value_columns:
            print(
                "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Error: Min/max file %s does not contain a CpG feature column." % path,
            )
            sys.exit(1)
        feature_column = non_value_columns[0]

    minmax_table["min"] = pd.to_numeric(minmax_table["min"], errors="coerce")
    minmax_table["max"] = pd.to_numeric(minmax_table["max"], errors="coerce")
    return minmax_table.drop_duplicates(feature_column).set_index(feature_column)[
        ["min", "max"]
    ]


def apply_minmax(output, feature_ids, path, label):
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Reading %s min/max file %s" % (label, path),
    )
    minmax_lookup = load_minmax(path)
    bounds = pd.DataFrame({"featureID": feature_ids})
    bounds["min"] = bounds["featureID"].map(minmax_lookup["min"])
    bounds["max"] = bounds["featureID"].map(minmax_lookup["max"])
    matched = bounds[["min", "max"]].notna().all(axis=1).sum()
    if matched == 0:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Error: %s min/max file %s does not match any input CpG loci."
            % (label, path),
        )
        sys.exit(1)
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Applying %s min/max clipping to %s of %s CpGs."
        % (label, matched, len(bounds)),
    )
    clipped = output.copy()
    matched_mask = bounds[["min", "max"]].notna().all(axis=1).values
    clipped.iloc[matched_mask, :] = clipped.iloc[matched_mask, :].clip(
        lower=bounds.loc[matched_mask, "min"].values,
        upper=bounds.loc[matched_mask, "max"].values,
        axis=0,
    )
    return clipped


def transform(x):
    feature = x[0]
    coef = coef_lookup.loc[feature]
    if np.isscalar(coef):
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
    "Prefix for A2H output assigned: %s" % args.prefix,
)

if args.camda_coef:
    args.camda = True
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Direct CAMDA coefficient file assigned: %s" % args.camda_coef,
    )
if args.camda_only:
    args.camda = True
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CAMDA-only mode enabled. CHALM files will not be written.",
    )
if args.chalm_minmax:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CHALM min/max file assigned: %s" % args.chalm_minmax,
    )
if args.camda_minmax:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CAMDA min/max file assigned: %s" % args.camda_minmax,
    )

if args.filter:
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


print(
    "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Reading input %s" % args.input,
)

methylation_input = pd.read_csv(args.input, sep="\t")
cpg_probe_info = (
    pd.read_table(
        os.path.join(model_dir, "A2H_cpg_probe.%s.tsv" % args.ref),
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

output_columns = methylation_input.columns[1:]
output_index = methylation_probeid if args.probe else methylation_input.iloc[:, 0]
output_features = methylation_input.iloc[:, 0].copy().values
chalm_output = None
camda_output = None
need_chalm_transform = (not args.camda_only) or (args.camda and not args.camda_coef)

if need_chalm_transform:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Reading CHALM coefficient file %s" % args.coef,
    )

    coef_lookup = load_coef_lookup(args.coef)

    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H CHALM transformation starts.",
    )

    chalm_output = pd.DataFrame(
        Parallel(n_jobs=args.thread, verbose=args.verbose, backend="multiprocessing")(
            delayed(transform)(_) for _ in methylation_input.values
        ),
        columns=output_columns,
        index=output_index,
    )

    if args.chalm_minmax:
        chalm_output = apply_minmax(
            chalm_output, output_features, args.chalm_minmax, "CHALM"
        )

    if not args.camda_only:
        chalm_output.to_csv(args.output + f"/{args.prefix}.chalm.tsv", sep="\t")
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "A2H CHALM transformation completed, saved to %s" % args.output
            + f"/{args.prefix}.chalm.tsv",
        )
    else:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "A2H CHALM transformation completed for legacy CAMDA calculation.",
        )

if args.camda:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H CAMDA transformation starts.",
    )
    if args.camda_coef:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Reading direct CAMDA coefficient file %s" % args.camda_coef,
        )
        coef_lookup = load_coef_lookup(args.camda_coef)
        camda_output = pd.DataFrame(
            Parallel(n_jobs=args.thread, verbose=args.verbose, backend="multiprocessing")(
                delayed(transform)(_) for _ in methylation_input.values
            ),
            columns=output_columns,
            index=output_index,
        )
    else:
        camda_output = pd.DataFrame(
            pd.DataFrame(chalm_output).values - methylation_input.iloc[:, 1:].values,
            columns=output_columns,
            index=output_index,
        )
    if args.camda_minmax:
        camda_output = apply_minmax(
            camda_output, output_features, args.camda_minmax, "CAMDA"
        )
    camda_output.to_csv(args.output + f"/{args.prefix}.camda.tsv", sep="\t")
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H CAMDA transformation completed, saved to %s" % args.output
        + f"/{args.prefix}.camda.tsv",
    )

# output confidence level for each cpg
if args.cl:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "A2H confidence level calculation starts.",
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
if args.filter:
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filtering output based on confidence level assigned: %s" % args.filter,
    )
    methylation_input_ci_mask = methylation_input_ci >= args.filter
    filtered_outputs = []
    if args.replace:
        print(
            "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Replace filtered and missing values with %s" % args.replace,
        )
        if chalm_output is not None and not args.camda_only:
            chalm_filtered_path = args.output + f"/{args.prefix}.chalm.filtered.tsv"
            chalm_output.where(methylation_input_ci_mask, args.replace).to_csv(
                chalm_filtered_path, sep="\t"
            )
            filtered_outputs.append(chalm_filtered_path)
        if args.camda:
            camda_filtered_path = args.output + f"/{args.prefix}.camda.filtered.tsv"
            camda_output.where(methylation_input_ci_mask, args.replace).to_csv(
                camda_filtered_path, sep="\t"
            )
            filtered_outputs.append(camda_filtered_path)
    else:
        if chalm_output is not None and not args.camda_only:
            chalm_filtered_path = args.output + f"/{args.prefix}.chalm.filtered.tsv"
            chalm_output.where(methylation_input_ci_mask).to_csv(
                chalm_filtered_path, sep="\t"
            )
            filtered_outputs.append(chalm_filtered_path)
        if args.camda:
            camda_filtered_path = args.output + f"/{args.prefix}.camda.filtered.tsv"
            camda_output.where(methylation_input_ci_mask).to_csv(
                camda_filtered_path, sep="\t"
            )
            filtered_outputs.append(camda_filtered_path)
    print(
        "[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filtering done, saved to %s" % ", ".join(filtered_outputs),
    )
