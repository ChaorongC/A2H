import unittest

import numpy as np
import pandas as pd

from src.audit_model import compare_coefficients


class ModelAuditTests(unittest.TestCase):
    def test_coefficient_comparison_aligns_by_locus(self):
        atlas = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], index=["chr1_1", "chr1_2"])
        deployed = pd.DataFrame({
            "featureID": ["chr1_2", "chr1_1"],
            "chalm_coef": [np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0])],
        })
        self.assertTrue(compare_coefficients(deployed, atlas, "chalm_coef")["exact_match"])
        deployed.at[0, "chalm_coef"] = np.array([4.0, 5.0, 7.0])
        self.assertEqual(compare_coefficients(deployed, atlas, "chalm_coef")["coefficient_triplet_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
