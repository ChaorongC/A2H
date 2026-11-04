import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.summarize_transfer import heldout_summary, load_completions, method_differences, paired_external_metrics


class TransferSummaryTests(unittest.TestCase):
    def test_load_completions_verifies_heldout_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "holdout" / "shard_0000"
            folder.mkdir(parents=True)
            heldout = folder / "heldout.parquet"
            heldout.write_bytes(b"test")
            detail = {"heldout_site_metrics": str(heldout), "heldout_sha256": file_hash(heldout)}
            record = {"model": "holdout", "shard_index": 0,
                      "source_sha256": {key: "same" for key in ("atlas_shards", "external_shards", "membership", "samples")},
                      "targets": {"CHALM": detail, "CAMDA": detail}}
            (folder / "completion.json").write_text(json.dumps(record))
            self.assertEqual(len(load_completions(root, "holdout", 1)), 1)
            heldout.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_completions(root, "holdout", 1)

    def test_paired_external_metrics_keep_only_common_sites(self):
        frame = pd.DataFrame({
            "probe_id": ["p1", "p1", "p2"], "sample_id": ["s1", "s2", "s1"],
            "observed": [0.1, 0.9, 0.5], "transfer_A2H": [0.1, 0.9, 0.5],
            "historical_A2H": [0.2, 0.8, 0.4],
            "transfer_reference_linear": [0.2, 0.8, 0.4], "m": [0.2, 0.8, 0.4],
        })
        paired, site = paired_external_metrics(frame, 2)
        self.assertEqual(set(paired.probe_id), {"p1"})
        self.assertEqual(set(site.method), {"A2H", "historical_A2H", "reference_linear", "m"})
        differences = method_differences(site, "CHALM")
        self.assertEqual(set(differences.control), {"historical_A2H", "reference_linear", "m"})
        self.assertTrue((differences.mean_site_rmse_A2H_minus_control < 0).all())

    def test_heldout_summary_rejects_different_masks(self):
        site = pd.DataFrame({
            "locus": ["chr1_1"] * 3,
            "method": ["A2H", "reference_linear", "m"],
            "n": [8, 7, 8], "rmse": [0.1, 0.2, 0.2],
        })
        with self.assertRaisesRegex(ValueError, "unequal subject counts"):
            heldout_summary(site, "CHALM", "holdout", 8)


if __name__ == "__main__":
    unittest.main()
