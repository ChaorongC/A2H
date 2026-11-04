import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "chalm_array_from_methratio.py"
BUILD_ORIGINAL_SCRIPT = REPO_ROOT / "scripts" / "build_original_chalm_camda_matrices.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "chalm_array_from_methratio", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_build_original_module():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "build_original_chalm_camda_matrices", BUILD_ORIGINAL_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildProbeMatrixTest(unittest.TestCase):
    def test_build_original_script_resolves_a2h_probe_map_name(self):
        module = load_build_original_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2h_probe_map = root / "A2H_cpg_probe.hg38.tsv"
            a2h_probe_map.write_text("loci.hg38\tEPIC.hg38\n")

            resolved = module.resolve_probe_map_path(root, "hg38")

        self.assertEqual(resolved, a2h_probe_map)

    def test_build_probe_matrix_maps_loci_to_epic_probes(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sample_a = tmpdir / "sample_a_methRatio.txt"
            sample_b = tmpdir / "sample_b_methRatio.txt"
            probe_map = tmpdir / "probe_map.tsv"

            sample_a.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.10",
                        "chr1\t200\t+\tCG\t0.20",
                        "chr1\t300\t+\tCHH\t0.99",
                    ]
                )
                + "\n"
            )
            sample_b.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.15",
                        "chr1\t250\t+\tCG\t0.25",
                    ]
                )
                + "\n"
            )
            probe_map.write_text(
                "\n".join(
                    [
                        "loci.hg38\tEPIC.hg38",
                        "chr1_100\tcg000001",
                        "chr1_200\tcg000002",
                        "chr1_200\tcg000003",
                        "chr1_250\tcg000004",
                        "chr1_999\tcg000005",
                    ]
                )
                + "\n"
            )

            matrix = module.build_probe_matrix(
                {
                    "GSM_A": sample_a,
                    "GSM_B": sample_b,
                },
                probe_map,
                reference="hg38",
                beadchip="EPIC",
            )

            self.assertListEqual(list(matrix.columns), ["Probe_ID", "GSM_A", "GSM_B"])
            self.assertEqual(set(matrix["Probe_ID"]), {"cg000001", "cg000002", "cg000003", "cg000004"})

            matrix = matrix.set_index("Probe_ID")
            self.assertAlmostEqual(matrix.loc["cg000001", "GSM_A"], 0.10)
            self.assertAlmostEqual(matrix.loc["cg000001", "GSM_B"], 0.15)
            self.assertAlmostEqual(matrix.loc["cg000002", "GSM_A"], 0.20)
            self.assertAlmostEqual(matrix.loc["cg000003", "GSM_A"], 0.20)
            self.assertTrue(pd.isna(matrix.loc["cg000002", "GSM_B"]))
            self.assertAlmostEqual(matrix.loc["cg000004", "GSM_B"], 0.25)
            self.assertTrue(pd.isna(matrix.loc["cg000004", "GSM_A"]))

    def test_build_original_locus_and_camda_matrices(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meth_a = tmpdir / "GSM_A_methRatio.txt"
            meth_b = tmpdir / "GSM_B_methRatio.txt"
            chalm_a = tmpdir / "GSM_A_CHALM.txt"
            chalm_b = tmpdir / "GSM_B_CHALM.txt"
            probe_map = tmpdir / "probe_map.tsv"

            meth_a.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.10",
                        "chr1\t200\t+\tCG\t0.20",
                    ]
                )
                + "\n"
            )
            meth_b.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.15",
                        "chr1\t250\t+\tCG\t0.25",
                    ]
                )
                + "\n"
            )
            chalm_a.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.30",
                        "chr1\t200\t+\tCG\t0.40",
                    ]
                )
                + "\n"
            )
            chalm_b.write_text(
                "\n".join(
                    [
                        "chr\tpos\tstrand\tcontext\tratio",
                        "chr1\t100\t+\tCG\t0.35",
                        "chr1\t250\t+\tCG\t0.55",
                    ]
                )
                + "\n"
            )
            probe_map.write_text(
                "\n".join(
                    [
                        "loci.hg38\tEPIC.hg38",
                        "chr1_100\tcg000001",
                        "chr1_200\tcg000002",
                        "chr1_250\tcg000003",
                        "chr1_999\tcg000004",
                    ]
                )
                + "\n"
            )

            meth_matrix = module.build_original_ratio_matrix(
                {"GSM_A": meth_a, "GSM_B": meth_b},
                probe_map,
                reference="hg38",
                beadchip="EPIC",
            )
            chalm_matrix = module.build_original_ratio_matrix(
                {"GSM_A": chalm_a, "GSM_B": chalm_b},
                probe_map,
                reference="hg38",
                beadchip="EPIC",
            )
            camda_matrix = module.build_original_camda_matrix(meth_matrix, chalm_matrix)

            self.assertListEqual(list(meth_matrix.columns), ["Probe_ID", "GSM_A", "GSM_B"])
            self.assertListEqual(list(chalm_matrix.columns), ["Probe_ID", "GSM_A", "GSM_B"])
            self.assertListEqual(list(camda_matrix.columns), ["Probe_ID", "GSM_A", "GSM_B"])

            meth_matrix = meth_matrix.set_index("Probe_ID")
            chalm_matrix = chalm_matrix.set_index("Probe_ID")
            camda_matrix = camda_matrix.set_index("Probe_ID")

            self.assertAlmostEqual(camda_matrix.loc["chr1_100", "GSM_A"], 0.20)
            self.assertAlmostEqual(camda_matrix.loc["chr1_100", "GSM_B"], 0.20)
            self.assertAlmostEqual(camda_matrix.loc["chr1_200", "GSM_A"], 0.20)
            self.assertTrue(pd.isna(camda_matrix.loc["chr1_200", "GSM_B"]))
            self.assertAlmostEqual(camda_matrix.loc["chr1_250", "GSM_B"], 0.30)
            self.assertTrue(pd.isna(camda_matrix.loc["chr1_250", "GSM_A"]))

    def test_filter_matrix_by_confidence_masks_low_confidence_values(self):
        module = load_module()

        matrix = pd.DataFrame(
            {
                "Probe_ID": ["chr1_100", "chr1_200"],
                "GSM_A": [0.2, 0.4],
                "GSM_B": [0.3, 0.5],
            }
        )
        confidence = pd.DataFrame(
            {
                "Probe_ID": ["chr1_100", "chr1_200"],
                "GSM_A": [0.6, 0.4],
                "GSM_B": [0.9, 0.7],
            }
        )

        filtered = module.filter_matrix_by_confidence(matrix, confidence, threshold=0.5)
        filtered = filtered.set_index("Probe_ID")

        self.assertAlmostEqual(filtered.loc["chr1_100", "GSM_A"], 0.2)
        self.assertAlmostEqual(filtered.loc["chr1_100", "GSM_B"], 0.3)
        self.assertTrue(pd.isna(filtered.loc["chr1_200", "GSM_A"]))
        self.assertAlmostEqual(filtered.loc["chr1_200", "GSM_B"], 0.5)


if __name__ == "__main__":
    unittest.main()
