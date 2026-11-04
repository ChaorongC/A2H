import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FILTER_SCRIPT = REPO_ROOT / "scripts" / "filter_wgbs_depth_outputs.py"
A2H_SCRIPT = REPO_ROOT / "scripts" / "generate_paired_validation_a2h_outputs.py"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_wgbs_depth_threshold_outputs.sh"


HEADER = (
    "chr\tpos\tstrand\tcontext\tratio\teff_CT_count\tC_count\tCT_count\t"
    "rev_G_count\trev_GA_count\tCI_lower\tCI_upper\n"
)


def ratio_rows() -> str:
    return (
        HEADER
        + "chr1\t101\t+\tCG\t0.250\t4.00\t1\t4\tNA\tNA\t0.089\t0.532\n"
        + "chr1\t102\t+\tCG\t0.571\t7.00\t4\t7\tNA\tNA\t0.250\t0.842\n"
        + "chr1\t103\t+\tCG\t0.875\t8.00\t7\t8\tNA\tNA\t0.529\t0.978\n"
        + "chr1\t104\t+\tCG\t0.900\t10.00\t9\t10\tNA\tNA\t0.596\t0.982\n"
    )


class WgbsDepthFilteringTest(unittest.TestCase):
    def test_filter_script_retains_rows_at_or_above_ct_count_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_trad = root / "input" / "traditional"
            input_chalm = root / "input" / "chalm"
            output_trad = root / "output" / "traditional"
            output_chalm = root / "output" / "chalm"
            input_trad.mkdir(parents=True)
            input_chalm.mkdir(parents=True)
            (input_trad / "S1_methRatio.txt").write_text(ratio_rows())
            (input_chalm / "S1_CHALM.txt").write_text(ratio_rows())
            summary = root / "output" / "coverage_filter_summary.tsv"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(FILTER_SCRIPT),
                    "--input-traditional-dir",
                    str(input_trad),
                    "--input-chalm-dir",
                    str(input_chalm),
                    "--output-traditional-dir",
                    str(output_trad),
                    "--output-chalm-dir",
                    str(output_chalm),
                    "--min-depth",
                    "8",
                    "--sample-id",
                    "S1",
                    "--summary",
                    str(summary),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            filtered = pd.read_csv(output_trad / "S1_methRatio.txt", sep="\t")
            self.assertEqual(filtered["pos"].tolist(), [103, 104])
            self.assertTrue((filtered["CT_count"] >= 8).all())

            filtered_chalm = pd.read_csv(output_chalm / "S1_CHALM.txt", sep="\t")
            pd.testing.assert_frame_equal(filtered_chalm, filtered)

            summary_df = pd.read_csv(summary, sep="\t")
            self.assertEqual(set(summary_df["file_type"]), {"methRatio", "CHALM"})
            self.assertEqual(summary_df["input_rows"].tolist(), [4, 4])
            self.assertEqual(summary_df["retained_rows"].tolist(), [2, 2])
            self.assertEqual(summary_df["min_retained_ct_count"].tolist(), [8.0, 8.0])

    def test_paired_a2h_generator_exposes_wgbs_input_output_overrides(self):
        completed = subprocess.run(
            [sys.executable, str(A2H_SCRIPT), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--wgbs-input-dir", completed.stdout)
        self.assertIn("--a2h-wgbs-dir", completed.stdout)

    def test_depth_runner_documents_supported_cohorts_and_threshold_argument(self):
        completed = subprocess.run(
            ["bash", str(RUNNER_SCRIPT), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--cohort gse46644|pgpuk|pidsley|gse240482_prostate_epicv1", completed.stdout)
        self.assertIn("--min-depth N", completed.stdout)

    def test_depth_runner_supports_gse240482_prostate_epicv1_alias(self):
        completed = subprocess.run(
            ["bash", str(RUNNER_SCRIPT), "--cohort", "gse240482_prostate_epicv1", "--min-depth", "8", "--dry-run"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("source_study=epicv2_eval", completed.stdout)
        self.assertIn("output_prefix=gse240482_prostate_epicv1_EPIC", completed.stdout)
        self.assertIn("platform=EPIC", completed.stdout)

    def test_paired_a2h_generator_exposes_output_prefix_override(self):
        completed = subprocess.run(
            [sys.executable, str(A2H_SCRIPT), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-prefix", completed.stdout)


if __name__ == "__main__":
    unittest.main()
