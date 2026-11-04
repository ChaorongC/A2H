import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.summarize_atlas_site_quality import run


class SiteQualityTests(unittest.TestCase):
    def test_rejects_incomplete_repeat_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repeat = root / "repeat"
            repeat.mkdir()
            (repeat / "completion.json").write_text(json.dumps({
                "branch": "donor_grouped", "split_count": 10, "aggregation_complete": False,
            }))
            with self.assertRaises(ValueError):
                run(root, root, repeat, root / "out", "donor_grouped", 10)


if __name__ == "__main__":
    unittest.main()
