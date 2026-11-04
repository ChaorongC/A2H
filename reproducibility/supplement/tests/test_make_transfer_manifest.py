import unittest

import pandas as pd

from src.make_transfer_manifest import build, class_masks, donor_subset


class TransferManifestTests(unittest.TestCase):
    def test_classification_and_donor_safe_subset(self):
        samples = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "donor_id": ["d1", "d1", "d2", "d3", "d4", "d5"],
            "Group": ["Blood-T", "Colon-Ep", "Endothelium", "Pancreas-Duct", "Epid-Kerat", "Neuron"],
        })
        masks = class_masks(samples)
        self.assertEqual(masks["blood_immune"].sum(), 1)
        self.assertEqual(masks["epithelial"].sum(), 3)
        selected = donor_subset(samples, 4, 1)
        self.assertEqual(samples.donor_id.isin(selected).sum(), 4)
        manifest, summary = build(samples, n_controls=1, seed=1)
        blood = manifest.query("model == 'exclude_blood_immune'")
        self.assertEqual(set(blood.query("role == 'train'").donor_id) &
                         set(blood.query("role == 'heldout'").donor_id), set())
        self.assertEqual(summary.set_index("model").loc["exclude_blood_immune", "n_train"], 4)


if __name__ == "__main__":
    unittest.main()
