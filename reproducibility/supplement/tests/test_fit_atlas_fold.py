import unittest

import numpy as np

from src.fit_atlas_fold import fit_sites


class FoldFitTests(unittest.TestCase):
    def test_quadratic_and_constant_kde_fallback(self):
        x = np.array([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.5] * 6])
        y = np.array([0.1 + 0.3 * x[0] + 0.2 * x[0]**2, [0.19] * 6])
        quality, pred = fit_sites(x, y, np.array([[0.3], [0.5]]), min_train=5)
        self.assertEqual(quality.fit_status.tolist(), ["quadratic", "constant_kde"])
        self.assertAlmostEqual(pred["A2H"][0, 0], 0.1 + 0.3 * 0.3 + 0.2 * 0.3**2)
        self.assertAlmostEqual(pred["A2H"][1, 0], 2 / 9)

    def test_joint_count_rank_and_training_only_fit(self):
        x = np.array([[0.0, 0.5, 1.0, np.nan, 0.5], [0.0, 1.0, 0.0, 1.0, 0.0]])
        y = np.array([[0.1, 0.2, 0.3, 0.4, np.nan], [0.1, 0.2, 0.1, 0.2, 0.1]])
        quality, pred = fit_sites(x, y, np.array([[0.7], [0.7]]), min_train=4)
        self.assertEqual(quality.n_train_joint.tolist(), [3, 5])
        self.assertEqual(quality.fit_status.tolist(), ["insufficient", "rank_deficient"])
        self.assertTrue(np.isnan(pred["A2H"]).all())
        self.assertTrue(np.isfinite(pred["reference_linear"][1, 0]))


if __name__ == "__main__":
    unittest.main()
