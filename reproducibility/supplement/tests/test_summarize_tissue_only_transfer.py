import unittest
import numpy as np
import pandas as pd

from src.summarize_tissue_only_transfer import (ALL_MODELS, align_external, bootstrap,
                                              external_metrics, reconstruct_sites)


class SummaryTests(unittest.TestCase):
    def frames(self):
        frame = pd.DataFrame({"probe_id": np.repeat(["cg1", "cg2"], 10),
            "sample_id": np.tile([f"s{i}" for i in range(10)], 2),
            "observed": np.tile(np.arange(10)/10, 2)})
        frame["m"] = frame.observed+.05
        frame["A2H"] = frame.observed+.1
        frame["reference_linear"] = frame.observed+.2
        frames = {m: frame.copy() for m in ALL_MODELS}
        frames[ALL_MODELS[1]]["A2H"] += .03
        return frames

    def test_common_mask_and_eight_subjects(self):
        frames = self.frames()
        frames[ALL_MODELS[1]].loc[[0, 1, 10, 11, 12], "A2H"] = np.nan
        common, audit, _ = align_external(frames, {"cg1", "cg2"})
        self.assertEqual(len(common), 8)
        self.assertEqual(set(common.probe_id), {"cg1"})
        self.assertEqual(audit.n_common_observations.tolist(), [8, 7])
        self.assertEqual(set(external_metrics(common).n), {8})

    def test_wrong_keys_or_truth_rejected(self):
        frames = self.frames()
        frames[ALL_MODELS[1]].loc[0, "observed"] = .8
        with self.assertRaises(AssertionError):
            align_external(frames, {"cg1", "cg2"})
        frames = self.frames()
        frames[ALL_MODELS[1]] = frames[ALL_MODELS[1]].iloc[1:]
        with self.assertRaisesRegex(ValueError, "universe"):
            align_external(frames, {"cg1", "cg2"})

    def test_bootstrap_paired_mask_and_constant_difference(self):
        frames = self.frames()
        frames[ALL_MODELS[1]].loc[[0, 1], "A2H"] = np.nan
        common, _, _ = align_external(frames, {"cg1", "cg2"})
        site = external_metrics(common)
        result, values, draws = bootstrap(common, [f"s{i}" for i in range(10)], site, replicates=20)
        row = result.loc[(result.model == ALL_MODELS[1]) & (result.method == "A2H")].iloc[0]
        self.assertAlmostEqual(row.mean_site_rmse, .13)
        for name in ("delta_vs_full", "delta_ci_low", "delta_ci_high"):
            self.assertAlmostEqual(row[name], .03)
        again = bootstrap(common, [f"s{i}" for i in range(10)], site, replicates=20)
        pd.testing.assert_frame_equal(values, again[1])
        pd.testing.assert_frame_equal(draws, again[2])

    def test_independent_heldout_reconstruction(self):
        f = self.frames()["full_reference"].rename(columns={"probe_id": "locus"})
        f.loc[:2, "A2H"] = np.nan
        f["evaluation_valid"] = np.isfinite(f[["A2H", "observed", "m", "reference_linear"]]).all(axis=1)
        s = reconstruct_sites(f)
        self.assertEqual(s.loc[s.locus == "cg1", "n"].tolist(), [7]*3)
        self.assertFalse(s.loc[s.locus == "cg1", "eligible"].any())
        self.assertTrue(np.allclose(s.loc[s.method == "A2H", "rmse"], .1))


if __name__ == "__main__":
    unittest.main()
