import unittest

import numpy as np
import pandas as pd

from src.fit_atlas_fold import fit_sites
from src.tissue_only_transfer import (epic_mapping, members, paired_site_metrics,
                                      predict_coefficients, tissue_membership)


class TissueOnlyTests(unittest.TestCase):
    def samples(self):
        return pd.DataFrame({"sample_id": [f"s{i}" for i in range(205)],
                             "donor_id": [f"d{i}" for i in range(205)],
                             "Group": ["Blood-T"] * 44 + ["Colon-Ep"] * 83 + ["Neuron"] * 78})

    def test_exact_sizes_and_same_donor_other_tissue_retained(self):
        samples = self.samples()
        samples.loc[204, "donor_id"] = samples.loc[0, "donor_id"]
        samples.loc[203, "donor_id"] = samples.loc[44, "donor_id"]
        manifest, summary = tissue_membership(samples)
        self.assertEqual(summary.n_train.tolist(), [161, 122])
        self.assertEqual(summary.n_heldout.tolist(), [44, 83])
        self.assertTrue(summary.n_excluded.eq(0).all())
        for model in summary.model:
            train, test = members(manifest, samples, model)
            self.assertIn("s204", train)
            self.assertIn("s203", train)
            self.assertFalse(set(train) & set(test))

    def test_wrong_policy_or_duplicate_membership_rejected(self):
        samples = self.samples()
        manifest, _ = tissue_membership(samples)
        bad = manifest.copy()
        bad.loc[204, "role"] = "excluded"
        with self.assertRaisesRegex(ValueError, "tissue-only"):
            members(bad, samples, "exclude_blood_immune")
        samples.loc[1, "sample_id"] = "s0"
        with self.assertRaisesRegex(ValueError, "unique"):
            tissue_membership(samples)

    def test_group_rules_preserved(self):
        samples = pd.DataFrame({"sample_id": list("abcde"), "donor_id": list("abcde"),
                                "Group": ["Blood-Mono+Macro", "Pancreas-Duct", "Epid-Kerat", "Breast-Basal-Ep", "Endothelium"]})
        _, counts = tissue_membership(samples)
        self.assertEqual(counts.n_heldout.tolist(), [1, 3])

    def test_explicit_mapping_and_coordinate_convention(self):
        manifest = pd.DataFrame({"probeID": ["cg1", "cg2", "ch3", "cg4"],
                                 "CpG_chrm": ["chr1"]*4, "CpG_beg": [9, 19, 29, np.nan]})
        mapping = pd.DataFrame({"loci.hg38": ["chr1_10", "chr1_20", "chr1_30", "chr1_40"],
                                "EPIC.hg38": ["cg1", "cg2", "ch3", "cg4"],
                                "HM450.hg38": ["cg1", "cg2", "ch3", "cg4"]})
        result = epic_mapping(manifest, mapping)
        self.assertEqual(result.probe_id.tolist(), ["cg1", "cg2"])
        repeated = epic_mapping(manifest, pd.concat([mapping, mapping.iloc[:1]], ignore_index=True))
        pd.testing.assert_frame_equal(result, repeated)
        mapping.loc[0, "loci.hg38"] = "chr1_9"
        with self.assertRaisesRegex(ValueError, "coordinates"):
            epic_mapping(manifest, mapping)
        audit = epic_mapping(manifest, mapping, allow_conflicts=True)
        self.assertFalse(audit.loc[audit.probe_id == "cg1", "eligible_mapping"].iloc[0])
        self.assertTrue(audit.loc[audit.probe_id == "cg2", "eligible_mapping"].iloc[0])

    def test_cross_platform_conflict_is_excluded(self):
        manifest = pd.DataFrame({"probeID": ["cg1"], "CpG_chrm": ["chr1"], "CpG_beg": [9]})
        mapping = pd.DataFrame({"loci.hg38": ["chr1_10", "chr1_11"],
                                "EPIC.hg38": ["cg1", None], "HM450.hg38": [None, "cg1"]})
        with self.assertRaisesRegex(ValueError, "HM450/EPIC"):
            epic_mapping(manifest, mapping)
        result = epic_mapping(manifest, mapping, allow_conflicts=True)
        self.assertFalse(result.eligible_mapping.iloc[0])

    def test_ambiguous_probe_rejected(self):
        manifest = pd.DataFrame({"probeID": ["cg1"], "CpG_chrm": ["chr1"], "CpG_beg": [9]})
        mapping = pd.DataFrame({"loci.hg38": ["chr1_10", "chr1_11"],
                                "EPIC.hg38": ["cg1", "cg1"], "HM450.hg38": ["cg1", "cg1"]})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            epic_mapping(manifest, mapping)

    def test_optional_coefficient_export_does_not_change_predictions(self):
        x = np.tile(np.linspace(0, 1, 60), (4, 1))
        y = .1 + .7*x + .1*x*x
        x[1] = .4
        x[2, :15] = np.nan
        x[3] = np.repeat([.2, .8], 30)
        test = np.tile(np.linspace(0, 1, 10), (4, 1))
        original, expected = fit_sites(x, y, test)
        quality, actual = fit_sites(x, y, test, include_coefficients=True)
        pd.testing.assert_frame_equal(quality[original.columns], original)
        for method in expected:
            np.testing.assert_array_equal(actual[method], expected[method])
        for method, prediction in predict_coefficients(quality, test).items():
            np.testing.assert_array_equal(prediction, actual[method])
        self.assertEqual(quality.fit_status.tolist(), ["quadratic", "constant_kde", "insufficient", "rank_deficient"])

    def test_quadratic_and_linear_against_independent_scalar_fits(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(size=(5, 65))
        y = .15 + .65*x + .12*x*x + rng.normal(0, .01, x.shape)
        x[0, :5] = np.nan
        test = rng.uniform(size=(5, 10))
        q, predicted = fit_sites(x, y, test, include_coefficients=True)
        for i in range(5):
            valid = np.isfinite(x[i]) & np.isfinite(y[i])
            np.testing.assert_allclose(predicted["A2H"][i], np.polyval(np.polyfit(x[i, valid], y[i, valid], 2), test[i]), atol=1e-12, rtol=0)
            np.testing.assert_allclose(predicted["reference_linear"][i], np.polyval(np.polyfit(x[i, valid], y[i, valid], 1), test[i]), atol=1e-12, rtol=0)

    def test_heldout_outcomes_not_used_for_fitting(self):
        x = np.tile(np.linspace(0, 1, 60), (2, 1))
        y = .1 + .7*x + .1*x*x
        q1, _ = fit_sites(x, y, np.full((2, 10), .2), include_coefficients=True)
        q2, _ = fit_sites(x, y, np.full((2, 10), .9), include_coefficients=True)
        pd.testing.assert_frame_equal(q1, q2)

    def test_metrics_use_identical_rows_and_minimum_eight(self):
        truth = np.tile(np.linspace(0, 1, 10), (3, 1))
        predictions = {m: truth.copy() for m in ("A2H", "reference_linear", "m")}
        predictions["reference_linear"][0, :2] = np.nan
        predictions["A2H"][1, :3] = np.nan
        predictions["A2H"][2] += .2
        metrics, valid = paired_site_metrics(["a", "b", "c"], truth, predictions)
        for method in ("A2H", "reference_linear", "m"):
            part = metrics.loc[metrics.method == method]
            self.assertEqual(part.n.tolist(), [8, 7, 10])
            self.assertEqual(part.eligible.tolist(), [True, False, True])
        self.assertAlmostEqual(metrics.loc[(metrics.locus == "c") & (metrics.method == "A2H"), "rmse"].iloc[0], .2)
        self.assertEqual(valid[0].sum(), 8)

    def test_constant_truth_correlation_missing_not_filled(self):
        truth = np.zeros((1, 10))
        predictions = {m: np.ones((1, 10))*.1 for m in ("A2H", "reference_linear", "m")}
        metrics, _ = paired_site_metrics(["a"], truth, predictions)
        self.assertTrue(metrics.pearson_r.isna().all())
        self.assertTrue(np.allclose(metrics.rmse, .1))


if __name__ == "__main__":
    unittest.main()
