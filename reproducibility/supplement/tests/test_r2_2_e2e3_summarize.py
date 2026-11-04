import numpy as np
import pandas as pd
import pytest

from src.r2_2_e2e3_summarize import censored_median, epic_calls, paired_metrics


def write_calls(directory, rows):
    directory.mkdir()
    for name, column in (("methRatio.txt", "m"), ("CHALM.txt", "chalm")):
        pd.DataFrame({"chr": [r[0] for r in rows], "pos": [r[1] for r in rows],
                      "ratio": [r[2][column] for r in rows], "CT_count": [r[3] for r in rows]}).to_csv(
            directory / name, sep="\t", index=False)


def test_paired_deltas_use_matched_loci_and_derive_camda(tmp_path):
    universe = pd.Index(["chr1_10", "chr1_20", "chr1_30"])
    write_calls(tmp_path / "native", [("chr1", 10, {"m": 0.5, "chalm": 0.7}, 20),
                                      ("chr1", 20, {"m": 0.2, "chalm": 0.4}, 30),
                                      ("chr2", 99, {"m": 0.9, "chalm": 0.9}, 30)])
    write_calls(tmp_path / "cond", [("chr1", 10, {"m": 0.6, "chalm": 0.6}, 5)])
    native = epic_calls(tmp_path / "native", universe)
    cond = epic_calls(tmp_path / "cond", universe)
    assert list(native.index) == ["chr1_10", "chr1_20"]
    rows = {r["target"]: r for r in paired_metrics(native, cond, len(universe))}
    assert rows["m"]["n_matched"] == 1 and rows["m"]["native_retention"] == 0.5
    assert np.isclose(rows["m"]["mean_signed_delta"], 0.1)
    assert np.isclose(rows["chalm"]["mean_signed_delta"], -0.1)
    assert np.isclose(rows["camda"]["mean_signed_delta"], -0.2)  # (0.6-0.6) - (0.7-0.5)


def test_censored_median_counts_absent_loci_low():
    assert censored_median(pd.Series([10.0, 20.0, 30.0]), 4) == 15.0
    assert np.isnan(censored_median(pd.Series([10.0]), 4))


def test_ct_disagreement_between_callers_fails(tmp_path):
    universe = pd.Index(["chr1_10"])
    write_calls(tmp_path / "d", [("chr1", 10, {"m": 0.5, "chalm": 0.7}, 20)])
    table = pd.read_csv(tmp_path / "d" / "CHALM.txt", sep="\t")
    table["CT_count"] = 21
    table.to_csv(tmp_path / "d" / "CHALM.txt", sep="\t", index=False)
    with pytest.raises(ValueError):
        epic_calls(tmp_path / "d", universe)
