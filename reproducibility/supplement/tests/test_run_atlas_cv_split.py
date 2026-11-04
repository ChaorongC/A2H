import unittest

import numpy as np
import pandas as pd

from src.run_atlas_cv_split import score_shard, split_members


class SplitRunnerTests(unittest.TestCase):
    def test_split_members_rejects_donor_leakage(self):
        manifest = pd.DataFrame({
            "branch": ["donor_grouped"] * 3, "split_index": [0] * 3,
            "repeat": [0] * 3, "fold": [0] * 3,
            "role": ["train", "train", "test"],
            "sample_id": ["s1", "s2", "s3"], "donor_id": ["a", "b", "a"],
        })
        with self.assertRaises(ValueError):
            split_members(manifest, "donor_grouped", 0, ["s1", "s2", "s3"])
        manifest.loc[2, "sample_id"] = "unexpected"
        with self.assertRaises(ValueError):
            split_members(manifest, "donor_grouped", 0, ["s1", "s2", "s3"])

    def test_held_out_target_cannot_change_predictions(self):
        train = [f"s{i}" for i in range(6)]
        test = ["heldout"]
        frame = pd.DataFrame({"locus": ["a"]})
        for i, sample in enumerate(train):
            frame[f"m__{sample}"] = [i / 5]
            frame[f"CHALM__{sample}"] = [0.1 + 0.5 * i / 5]
        frame["m__heldout"] = [0.7]
        frame["CHALM__heldout"] = [0.4]
        q1, p1 = score_shard(frame, "CHALM", train, test, np.array([0.1, 0.5, 0.0]), 5, 0, 0, 0)
        frame["CHALM__heldout"] = [0.9]
        q2, p2 = score_shard(frame, "CHALM", train, test, np.array([0.1, 0.5, 0.0]), 5, 0, 0, 0)
        self.assertTrue(q1.equals(q2))
        self.assertEqual(p1.A2H.iloc[0], p2.A2H.iloc[0])
        self.assertNotEqual(p1.observed.iloc[0], p2.observed.iloc[0])


if __name__ == "__main__":
    unittest.main()
