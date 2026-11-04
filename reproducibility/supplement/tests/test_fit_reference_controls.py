import unittest

import numpy as np
import pandas as pd

from src.fit_reference_controls import fit_chunk


class ReferenceControlTests(unittest.TestCase):
    def test_fit_is_reference_only_and_equal_locus_weight(self):
        x = np.array([[0.0, 0.5, 1.0, np.nan], [0.0, 0.5, 1.0, 0.5], [0.3, 0.3, 0.3, np.nan]])
        y = 0.2 + 0.4 * np.nan_to_num(x) + 0.3 * np.nan_to_num(x)**2
        y[np.isnan(x)] = np.nan
        table, gram, rhs = fit_chunk(x, y, pd.Index(["a", "b", "c"]), 3)
        self.assertEqual(table.n_train_joint.tolist(), [3, 4, 3])
        self.assertAlmostEqual(table.loc[2, "reference_mean"], y[2, :3].mean())
        self.assertTrue(table.loc[2, "linear_constant_fallback"])
        self.assertAlmostEqual(table.loc[2, "linear_slope"], 0)
        np.testing.assert_allclose(np.linalg.solve(gram, rhs), [0.2, 0.4, 0.3], atol=1e-12)

    def test_joint_missingness_and_minimum_training_count(self):
        x = np.array([[0.0, 0.5, 1.0], [0.0, np.nan, 1.0]])
        y = np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])
        table, _, _ = fit_chunk(x, y, pd.Index(["a", "b"]), 3)
        self.assertEqual(table.n_train_joint.tolist(), [2, 2])
        self.assertTrue(table.reference_mean.isna().all())


if __name__ == "__main__":
    unittest.main()
