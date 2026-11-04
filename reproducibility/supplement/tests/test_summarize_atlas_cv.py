import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.summarize_atlas_cv import METHODS, summarize_repeat


class SummarizeTests(unittest.TestCase):
    def test_repeat_uses_each_heldout_subject_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_path = root / "shard.parquet"
            pd.DataFrame({"locus": ["a", "b"]}).to_parquet(shard_path, index=False)
            shards = pd.DataFrame({"shard_index": [0], "path": [str(shard_path)]})
            completions = {}
            for split, subject in enumerate(["s1", "s2", "s3"]):
                frame = pd.DataFrame({
                    "locus": ["a", "b"], "sample_id": [subject] * 2,
                    "target": ["CHALM"] * 2, "repeat": [0] * 2,
                    "split_index": [split] * 2, "observed": [0.1 * (split + 1), 0.4],
                    "fit_status": ["quadratic"] * 2,
                })
                for method in METHODS:
                    frame[method] = frame.observed + 0.1
                path = root / f"split{split}.parquet"
                frame.to_parquet(path, index=False)
                completions[split] = {"repeat": 0, "test_ids": [subject],
                                      "outputs": [{"target": "CHALM", "shard_index": 0,
                                                   "prediction_path": str(path)}]}
            summary = summarize_repeat(shards, "CHALM", 0, completions, ["s1", "s2", "s3"], root)
            self.assertEqual(len(summary), len(METHODS))
            self.assertTrue((summary.n_site_subject_observations == 6).all())
            completions[2]["test_ids"] = ["s2"]
            with self.assertRaises(ValueError):
                summarize_repeat(shards, "CHALM", 0, completions, ["s1", "s2", "s3"], root)


if __name__ == "__main__":
    unittest.main()
