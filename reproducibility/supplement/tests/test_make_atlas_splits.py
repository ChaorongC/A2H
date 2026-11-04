import unittest

import pandas as pd

from src.make_atlas_splits import make_splits


class SplitTests(unittest.TestCase):
    def test_grouped_split_excludes_shared_donors_and_covers_each_repeat(self):
        samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(12)],
                                "donor_id": ["d0", "d0"] + [f"d{i}" for i in range(1, 11)]})
        manifest, audit = make_splits(samples, "donor_grouped", n_splits=5, n_repeats=2)
        self.assertEqual(len(audit), 10)
        self.assertTrue((audit.shared_donors == 0).all())
        self.assertTrue((manifest.query("role == 'test'").groupby("repeat").sample_id.nunique() == 12).all())

    def test_historical_branch_preserves_sample_order(self):
        samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(10)],
                                "donor_id": [f"d{i}" for i in range(10)]})
        manifest, audit = make_splits(samples, "historical_sample", n_splits=5, n_repeats=2)
        self.assertEqual(len(audit), 10)
        self.assertEqual(manifest.query("role == 'test'").groupby("repeat").size().tolist(), [10, 10])


if __name__ == "__main__":
    unittest.main()
