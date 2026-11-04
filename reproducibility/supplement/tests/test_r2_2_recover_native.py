import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/r2_2_recover_native.py'
module = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('recovery', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module, 'source-preserving recovery implementation is missing')

    def test_prefer_valid_mapped_bam(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'mapped.bam').touch()
            (root / 'unsorted.bam').touch()
            mode, source = module.choose_input(root, lambda p: True)
            self.assertEqual((mode, source), ('mapped', root / 'mapped.bam'))

    def test_reject_partial_mapped_use_unsorted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'mapped.bam').touch()
            (root / 'unsorted.bam').touch()
            mode, source = module.choose_input(root, lambda p: p.name == 'unsorted.bam')
            self.assertEqual((mode, source), ('unsorted', root / 'unsorted.bam'))

    def test_raw_restart_for_missing_or_truncated_bams(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(module.choose_input(Path(d), lambda p: False), ('raw', None))

    def test_recovery_output_must_not_equal_or_nest_in_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for out in [root, root / 'recovery']:
                with self.assertRaises(ValueError):
                    module.validate_destination(root, out)

    def test_existing_recovery_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'existing'
            out.mkdir()
            with self.assertRaises(FileExistsError):
                module.validate_destination(root / 'source', out)

    def test_historical_caller_flags(self):
        cmd = module.caller_command('python', 'CHALM.py', 'ref.fa', 'samdir', 'a.bam', 'b.txt', False)
        self.assertEqual(cmd, ['python', 'CHALM.py', 'trad', '-d', 'ref.fa', '-x', 'CG', '-i', 'no-action', '-m', '4', '-p', '-r', '-u', '-g', '-q', '-s', 'samdir', '-o', 'b.txt', 'a.bam'])
        chalm = module.caller_command('python', 'CHALM.py', 'ref.fa', 'samdir', 'a.bam', 'b.txt', True)
        self.assertEqual(chalm[-3:], ['-l', '1', 'a.bam'])

    def test_trimmed_restart_requires_both_original_mates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trim = root / 'trim'
            trim.mkdir()
            (trim / 'S_R1_val_1.fq.gz').touch()
            with self.assertRaises(ValueError):
                module.trimmed_mates(root, 'S')
            (trim / 'S_R2_val_2.fq.gz').touch()
            self.assertEqual(module.trimmed_mates(root, 'S'),
                             [trim / 'S_R1_val_1.fq.gz', trim / 'S_R2_val_2.fq.gz'])

    def test_threads_respect_allocation_and_samtools_main_thread(self):
        self.assertEqual(module.thread_budget(40, 40), (40, 39))
        self.assertEqual(module.thread_budget(40, 32), (32, 31))
        self.assertEqual(module.thread_budget(40, 64), (40, 39))
        self.assertEqual(module.thread_budget(1, 1), (1, 0))

    def test_invalid_thread_budget_rejected(self):
        for requested, allocated in [(0, 40), (41, 40), (40, 0)]:
            with self.assertRaises(ValueError):
                module.thread_budget(requested, allocated)


if __name__ == '__main__':
    unittest.main()
