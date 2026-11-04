import copy
import unittest

from src.package_r1_6_cv_and_holdouts import FIELDS, combine


class CombinedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.cv = [
            {"endpoint": endpoint, "target": target, "training_depth": ">=4",
             "n_atlas_samples": "205", "n_splits": "100",
             "n_EPIC_candidate_units": "824908" if endpoint == "subject_level_out_of_fold" else "824939",
             "n_eligible_units_min": "824498", "n_eligible_units_max": "824527",
             "rmse_mean": "0.05231246065843752", "rmse_sd": "0.000056773517451733724",
             "metric_definition": endpoint, "feature_definition": "historical EPIC"}
            for endpoint in ("subject_level_out_of_fold", "historical_heldout_fold_mean")
            for target in ("CHALM", "CAMDA")
        ]
        self.holdouts = [
            {"model": model, "target": target, "method": method,
             "n_train": train, "n_heldout": test, "n_sites": "821907",
             "n_observations": "35795063", "mean_site_rmse": "0.05465790230481616"}
            for model, train, test in (("exclude_blood_immune", "161", "44"),
                                       ("exclude_epithelial", "122", "83"))
            for target in ("CHALM", "CAMDA")
            for method in ("A2H", "reference_linear", "m")
        ]

    def combined(self):
        return combine(self.cv, self.holdouts, "cv.tsv", "holdouts.tsv")

    def test_complete_keys_and_numeric_strings_preserved(self):
        before = copy.deepcopy((self.cv, self.holdouts))
        table = self.combined()
        self.assertEqual(len(table), 16)
        self.assertEqual(len({(r["endpoint"], r["reference"], r["target"], r["method"]) for r in table}), 16)
        self.assertTrue(all(set(r) == set(FIELDS) for r in table))
        self.assertEqual(table[0]["rmse"], self.cv[0]["rmse_mean"])
        self.assertEqual(table[0]["rmse_sd"], self.cv[0]["rmse_sd"])
        self.assertEqual(table[4]["rmse"], self.holdouts[0]["mean_site_rmse"])
        self.assertEqual((self.cv, self.holdouts), before)

    def test_unknown_quantities_and_endpoint_units_remain_distinct(self):
        table = self.combined()
        for row in table:
            if row["endpoint"] == "fixed_tissue_holdout":
                self.assertEqual(row["rmse_sd"], "")
                self.assertEqual(row["n_observations"], "35795063")
                self.assertEqual(row["candidate_unique_EPIC_loci"], "824231")
            else:
                self.assertEqual(row["n_observations"], "")
                if row["endpoint"] == "historical_heldout_fold_mean":
                    self.assertEqual(row["candidate_unique_EPIC_loci"], "")
                    self.assertEqual(row["legacy_valid_manifest_rows"], "824939")
                else:
                    self.assertEqual(row["candidate_unique_EPIC_loci"], "824908")
                    self.assertEqual(row["legacy_valid_manifest_rows"], "")
        plotted = [r for r in table if r["method"] == "A2H" and r["endpoint"] != "historical_heldout_fold_mean"]
        self.assertEqual(len(plotted), 6)

    def test_missing_or_duplicate_cv_key_rejected(self):
        for values in (self.cv[:-1], self.cv[:-1] + [self.cv[0]]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "keys"):
                combine(values, self.holdouts, "cv.tsv", "holdouts.tsv")

    def test_missing_or_duplicate_holdout_key_rejected(self):
        for values in (self.holdouts[:-1], self.holdouts[:-1] + [self.holdouts[0]]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "keys"):
                combine(self.cv, values, "cv.tsv", "holdouts.tsv")

    def test_unexpected_cv_definitions_rejected(self):
        for field, value in (("training_depth", ">=10"), ("n_atlas_samples", "204"), ("n_splits", "99")):
            values = copy.deepcopy(self.cv)
            values[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CV definition"):
                combine(values, self.holdouts, "cv.tsv", "holdouts.tsv")

    def test_wrong_tissue_partitions_rejected(self):
        for model in ("exclude_blood_immune", "exclude_epithelial"):
            values = copy.deepcopy(self.holdouts)
            next(r for r in values if r["model"] == model)["n_train"] = "160"
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "partition"):
                combine(self.cv, values, "cv.tsv", "holdouts.tsv")


if __name__ == "__main__":
    unittest.main()
