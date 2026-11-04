import unittest

import pandas as pd

from src.run_atlas_cv_split import TARGETS
from src.summarize_atlas_cv import METHODS
from src.summarize_cv_rmse import aggregate


class CvRmseSummaryTests(unittest.TestCase):
    def fixture(self):
        shards = pd.DataFrame({"shard_index": [0, 1], "n_loci": [2, 8]})
        rows = []
        for target in TARGETS:
            for shard, size, rmse in ((0, 2, 0.1), (1, 8, 0.3)):
                for method in METHODS:
                    rows.append({"target": target, "repeat": 0, "shard_index": shard,
                                 "method": method, "n_loci": size,
                                 "n_sites_two_or_more": size,
                                 "n_site_subject_observations": 10 * size,
                                 "mean_site_rmse": rmse + (0.01 if method == "reference_linear" else 0)})
        return pd.DataFrame(rows), shards, {"split_count": 10}

    def test_weights_shards_by_eligible_cpg_count(self):
        frame, shards, completion = self.fixture()
        summary, differences = aggregate(frame, shards, completion)
        row = summary.loc[(summary.target == "CHALM") & (summary.method == "A2H")].iloc[0]
        self.assertAlmostEqual(row.mean_site_rmse, 0.26)
        paired = differences.loc[(differences.target == "CHALM") &
                                 (differences.control == "reference_linear")].iloc[0]
        self.assertAlmostEqual(paired.mean_site_rmse_A2H_minus_control, -0.01)

    def test_rejects_incomplete_method_grid(self):
        frame, shards, completion = self.fixture()
        with self.assertRaisesRegex(ValueError, "complete target-repeat-shard-method grid"):
            aggregate(frame.iloc[:-1], shards, completion)

    def test_rejects_different_method_masks(self):
        frame, shards, completion = self.fixture()
        frame.loc[0, "n_sites_two_or_more"] = 1
        with self.assertRaisesRegex(ValueError, "different CV observation masks"):
            aggregate(frame, shards, completion)


if __name__ == "__main__":
    unittest.main()
