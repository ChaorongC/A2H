import unittest

import numpy as np
import pandas as pd

from src.run_transfer_shard import external_arrays, external_scores, model_members


class TransferShardTests(unittest.TestCase):
    def test_membership_rejects_shared_donor(self):
        manifest = pd.DataFrame({
            "model": ["holdout", "holdout", "holdout"],
            "sample_id": ["a", "b", "c"],
            "donor_id": ["d1", "d1", "d2"],
            "role": ["train", "heldout", "excluded"],
        })
        with self.assertRaisesRegex(ValueError, "Donor overlap"):
            model_members(manifest, "holdout", ["a", "b", "c"])

    def test_external_scores_preserve_probe_and_sample_pairs(self):
        x = np.tile(np.linspace(0, 1, 60), (2, 1))
        y = 0.1 + 0.7 * x + 0.1 * x**2
        loci = pd.Index(["chr1_1", "chr1_2"])
        obs = pd.DataFrame({
            "probe_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "locus": ["chr1_1", "chr1_1", "chr1_2", "chr1_2", "chr1_2", "chr1_2"],
            "sample_id": ["s2", "s1"] * 3,
            "m": [0.8, 0.2, 0.6, 0.4, 0.1, 0.9],
            "observed": [0.724, 0.244, 0.556, 0.396, 0.171, 0.811],
            "A2H": [0.7, 0.2, 0.5, 0.4, 0.1, 0.8],
        })
        sites, predictions = external_scores(obs, loci, x, y, ["s1", "s2"], 50)
        transfer = sites.loc[sites.method == "A2H"]
        self.assertEqual(len(transfer), 3)
        self.assertTrue(np.allclose(transfer.rmse, 0, atol=1e-12))
        self.assertEqual(len(predictions), len(obs))
        match = predictions.set_index(["probe_id", "sample_id"]).loc[("p1", "s2")]
        self.assertAlmostEqual(match.transfer_A2H, 0.724)
        self.assertAlmostEqual(match.historical_A2H, 0.7)

    def test_external_arrays_reject_ambiguous_probe(self):
        obs = pd.DataFrame({
            "probe_id": ["p1", "p1"], "sample_id": ["s1", "s2"],
            "locus": ["chr1_1", "chr1_2"], "m": [0.1, 0.2],
            "observed": [0.1, 0.2], "A2H": [0.1, 0.2],
        })
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            external_arrays(obs, pd.Index(["chr1_1", "chr1_2"]), ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
