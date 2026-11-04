import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import group_sites, observation_table, paired_site_differences, reference_control_predictions, row_stats, scalar_metrics, score_grouped, spearman_rows, truth_root


class EvaluationTests(unittest.TestCase):
    def test_original_depth_uses_original_namespace(self):
        self.assertEqual(truth_root(Path("/source"), 4), Path("/source/results/paired_validation_v2/wgbs/probe_space"))
        self.assertEqual(truth_root(Path("/source"), 10), Path("/source/results/paired_validation_v2_depth10/wgbs/probe_space"))

    def test_missing_and_constant_rows(self):
        y = np.array([[0.0, 0.5, 1.0, np.nan], [0.2, 0.2, 0.2, np.nan]])
        pred = np.array([[0.0, 0.25, 1.0, 0.0], [0.3, 0.3, 0.3, np.nan]])
        scores = row_stats(y, pred)
        self.assertEqual(scores["n"].tolist(), [3, 3])
        self.assertAlmostEqual(scores["rmse"][0], np.sqrt(0.25**2 / 3))
        self.assertTrue(np.isnan(scores["pearson_r"][1]))
        self.assertAlmostEqual(spearman_rows(y, pred)[0], 1.0)

    def test_constant_prediction_correlation_is_undefined(self):
        y = np.array([[0.1, 0.2, 0.3, 0.4]])
        pred = np.full_like(y, 0.8)
        self.assertTrue(np.isnan(row_stats(y, pred)["pearson_r"][0]))
        self.assertTrue(np.isnan(scalar_metrics(y, pred)["pearson_r"]))
        self.assertTrue(np.isnan(spearman_rows(y, pred)[0]))

    def test_nrmse_uses_observed_range(self):
        result = scalar_metrics(np.array([0.2, 0.4]), np.array([0.2, 0.3]))
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.01 / 2))
        self.assertAlmostEqual(result["nrmse"], result["rmse"] / 0.2)
        constant = scalar_metrics(np.array([0.2, 0.2]), np.array([0.2, 0.3]))
        self.assertTrue(np.isnan(constant["nrmse"]))

    def test_matched_groups_are_disjoint(self):
        beta = np.tile(np.linspace(0.1, 0.2, 10), (20, 1))
        sites = pd.Index([f"cg{i:08d}" for i in range(20)])
        groups = group_sites(beta, sites)
        self.assertEqual(groups.group_id.nunique(), 2)
        self.assertEqual(groups.probe_id.nunique(), 20)
        indexed = pd.Series(np.arange(20), index=sites)
        ordered = groups.sort_values(["group_id", "probe_id"])
        scored = score_grouped(beta, beta, indexed.loc[ordered.probe_id].to_numpy(), ordered.group_id.to_numpy())
        self.assertTrue(scored.all_sites_ge8.all())
        self.assertTrue(np.allclose(scored.rmse, 0))

    def test_grouping_can_use_reference_statistics(self):
        beta = np.tile(np.linspace(0.1, 0.2, 10), (20, 1))
        sites = pd.Index([f"cg{i:08d}" for i in range(20)])
        stats = pd.DataFrame({"probe_id": sites[::-1], "mean_m": [0.25] * 20,
                              "var_m": [0.01] * 20})
        groups = group_sites(beta, sites, stats)
        self.assertEqual(groups.probe_id.nunique(), 20)
        self.assertTrue((groups.mean_m == 0.25).all())

    def test_reference_controls_align_by_locus_and_share_array_inputs(self):
        controls = pd.DataFrame({
            "locus": ["chr1_2", "chr1_1"],
            "reference_mean": [0.8, 0.2],
            "linear_intercept": [0.3, 0.1],
            "linear_slope": [0.4, 0.5],
        })
        beta = np.array([[0.2, 0.4], [0.5, 0.7]])
        loci = pd.Series(["chr1_1", "chr1_2"])
        predictions = reference_control_predictions(beta, loci, controls, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(predictions["reference_mean"], [[0.2, 0.2], [0.8, 0.8]])
        np.testing.assert_allclose(predictions["reference_linear"], [[0.2, 0.3], [0.5, 0.58]])
        np.testing.assert_allclose(predictions["universal_quadratic"], 0.1 + 0.2 * beta + 0.3 * beta**2)

    def test_paired_differences_reject_unequal_eligible_sites(self):
        table = pd.DataFrame({
            "probe_id": ["a", "b", "a", "b"],
            "target": ["CHALM"] * 4,
            "method": ["A2H", "A2H", "reference_linear", "reference_linear"],
            "n": [8, 9, 8, 9],
            "rmse": [0.1, 0.3, 0.2, 0.2],
            "spearman_rho": [0.2, 0.4, 0.1, 0.5],
            "eligible": [True] * 4,
        })
        difference = paired_site_differences(table).iloc[0]
        self.assertAlmostEqual(difference.mean_site_rmse_A2H_minus_control, 0)
        self.assertAlmostEqual(difference.fraction_sites_lower_rmse_A2H, 0.5)
        table.loc[3, "eligible"] = False
        with self.assertRaises(ValueError):
            paired_site_differences(table)

    def test_observation_table_keeps_subject_ids_and_common_mask(self):
        probes = pd.Index(["a", "b"])
        observed = np.array([[0.1, 0.2, 0.3], [0.5, np.nan, 0.7]])
        mask = np.isfinite(observed)
        out = observation_table(probes, ["S1", "S2", "S3"], observed,
                                {"A2H": observed + 0.1, "m": observed}, mask, 2)
        self.assertEqual(list(zip(out.probe_id, out.sample_id)),
                         [("a", "S1"), ("a", "S2"), ("a", "S3"), ("b", "S1"), ("b", "S3")])
        self.assertTrue(np.isfinite(out[["observed", "A2H", "m"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
