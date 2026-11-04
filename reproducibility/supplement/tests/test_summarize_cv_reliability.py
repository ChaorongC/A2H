import unittest

import numpy as np
import pandas as pd

from src.summarize_cv_reliability import strata, summarize


class ReliabilityTests(unittest.TestCase):
    def test_partition_and_observed_denominator(self):
        frame = pd.DataFrame({
            "min_train_joint": [48, 75, 201],
            "min_unique_input": [1, 2, 3],
            "n_quadratic_folds": [100, 99, 100],
            "n_training_folds": [100, 100, 100],
        })
        labels = strata(frame)
        self.assertEqual(labels["min_train_joint"].tolist(), ["<50", "50-99", ">=200"])
        self.assertEqual(labels["fit_status"].tolist(),
                         ["quadratic_all_folds", "nonquadratic_any_fold", "quadratic_all_folds"])
        self.assertEqual(labels["min_unique_input"].tolist(), ["<=1", "2", ">=3"])
        row = summarize({"rmse": [np.array([0.1, np.nan, 0.3])],
                         "observed": [np.array([True, False, True])],
                         "defined": [np.array([True, False, False])]}, 3)
        self.assertEqual(row["n_with_heldout_truth"], 2)
        self.assertEqual(row["n_with_defined_rank"], 1)
        self.assertAlmostEqual(row["median_site_rmse"], 0.2)
        self.assertAlmostEqual(row["fraction_undefined_rank_among_observed"], 0.5)


if __name__ == "__main__":
    unittest.main()
