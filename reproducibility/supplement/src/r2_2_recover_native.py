"""Resume failed native atlas jobs without modifying their original work trees."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REVIEW = Path(__file__).resolve().parents[1]
MANIFEST = REVIEW / 'results/r2_2_sensitivity_20261011/E2_E3_access/selection_39_groups/E2_E3_sample_selection.tsv'
REFERENCE = 'reference/hg38.fa'
CHALM = 'tools/CHALM/src/CHALM.py'
BSMAP = 'bsmap'
PYTHON = 'python'
OLD_SAM = 'tools/samtools-0.1.19'


def choose_input(root, quickcheck):
    for mode, name in [('mapped', 'mapped.bam'), ('unsorted', 'unsorted.bam')]:
        source = root / name
        if source.is_file() and quickcheck(source):
            return mode, source
    return 'raw', None


def validate_destination(source_root, out):
    source_root, out = source_root.resolve(), out.resolve()
    if out == source_root or source_root in out.parents or out in source_root.parents:
        raise ValueError('Recovery must use a separate work tree')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')


def caller_command(python, chalm, reference, samdir, bam, out, read_level):
    command = [str(python), str(chalm), 'trad', '-d', str(reference), '-x', 'CG',
               '-i', 'no-action', '-m', '4', '-p', '-r', '-u', '-g', '-q',
               '-s', str(samdir), '-o', str(out)]
    if read_level:
        command += ['-l', '1']
    return command + [str(bam)]


def trimmed_mates(original_dir, sample_id):
    mates = [original_dir / 'trim' / f'{sample_id}_R{m}_val_{m}.fq.gz' for m in [1, 2]]
    if not all(path.is_file() for path in mates):
        raise ValueError(f'Both completed trimmed mates are required for {sample_id}')
    return mates


def thread_budget(requested, allocated):
    if not 1 <= requested <= 40 or allocated < 1:
        raise ValueError('Request 1-40 threads and a positive CPU allocation')
    cores = min(requested, allocated)
    return cores, cores - 1


def fingerprint(path):
    s = path.stat()
    return {'path': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def recover(args):
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', str(args.threads)))
    alignment_threads, samtools_threads = thread_budget(args.threads, allocated)
    with args.manifest.open(newline='') as handle:
        selected = list(csv.DictReader(handle, delimiter='\t'))
    if len(selected) != 39 or len({r['sample_id'] for r in selected}) != 39:
        raise ValueError('Expected frozen 39-sample native universe')
    if len([r for r in selected if r['sample_id'] == args.sample_id and r['e2_candidate'] == 'yes']) != 1:
        raise ValueError('Recovery sample is not in the frozen E2 selection')
    validate_destination(args.original_dir, args.out_dir)
    if args.mode in ('mapped', 'unsorted'):
        if args.source_bam is None or args.source_bam.parent.resolve() != args.original_dir.resolve():
            raise ValueError('BAM source must remain inside the original sample directory')
        before = fingerprint(args.source_bam)
        if args.expected_bytes is not None and before['bytes'] != args.expected_bytes:
            raise ValueError('Source size changed since frozen recovery manifest')
        if args.expected_mtime_ns is not None and before['mtime_ns'] != args.expected_mtime_ns:
            raise ValueError('Source timestamp changed since frozen recovery manifest')
    else:
        before = None
    args.out_dir.mkdir(parents=True)
    metadata = {'sample_id': args.sample_id, 'mode': args.mode, 'source': before,
                'original_dir': str(args.original_dir.resolve()), 'out_dir': str(args.out_dir.resolve()),
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'code_sha256': digest(Path(__file__)), 'manifest_sha256': digest(args.manifest),
                'min_depth': 4, 'requested_threads': args.threads, 'allocated_cpus': allocated,
                'alignment_threads': alignment_threads, 'samtools_additional_threads': samtools_threads,
                'commands': [], 'state': 'running', 'source_tree_modified': False}
    meta = args.out_dir / 'recovery_metadata.json'

    def save():
        meta.write_text(json.dumps(metadata, indent=2) + '\n')

    def run(stage, command, stdout=None, capture=False):
        metadata['stage'] = stage
        metadata['commands'].append({'stage': stage, 'argv': [str(c) for c in command],
                                     'started_utc': datetime.now(timezone.utc).isoformat()})
        save()
        print(f'{metadata["commands"][-1]["started_utc"]} {stage}: {command}', flush=True)
        result = subprocess.run([str(c) for c in command], check=True, cwd=args.out_dir,
                                stdout=subprocess.PIPE if capture else stdout, text=capture)
        metadata['commands'][-1]['exit_code'] = result.returncode
        save()
        return result.stdout.strip() if capture else None

    def count(stage, path, filtered=False):
        command = [args.samtools, 'view', '-@', str(samtools_threads), '-c']
        if filtered:
            command += ['-F', '268']
        n = int(run(stage, command + [path], capture=True))
        if n <= 0:
            raise ValueError(f'No valid reads at {stage}')
        return n

    try:
        if args.mode == 'raw':
            reads = args.out_dir / 'reads'
            run('prepare_original_lanes', [args.python, REVIEW / 'src/r2_2_prepare_native_reads.py',
                '--sample-id', args.sample_id, '--manifest', args.manifest, '--out', reads])
            raw = [reads / f'{args.sample_id}_R{m}.fastq.gz' for m in [1, 2]]
            run('gzip_integrity', ['gzip', '-t', *raw])
            trim = args.out_dir / 'trim'
            trim.mkdir()
            run('historical_trimming', ['trim_galore', '--paired', '--2colour', '20',
                '--clip_R1', '10', '--clip_R2', '20', '--three_prime_clip_R1', '10',
                '--three_prime_clip_R2', '10', '--cores', '32', '--output_dir', trim, *raw])
            mates = []
            for mate in [1, 2]:
                paths = list(trim.glob(f'*_val_{mate}.fq.gz'))
                if len(paths) != 1:
                    raise ValueError(f'Ambiguous trimmed mate {mate}')
                mates.append(paths[0])
        elif args.mode == 'trimmed':
            mates = trimmed_mates(args.original_dir, args.sample_id)
            metadata['trimmed_sources'] = [fingerprint(path) for path in mates]
            save()
            run('original_trimmed_gzip_integrity', ['gzip', '-t', *mates])
        if args.mode in ('raw', 'trimmed'):
            source = args.out_dir / 'unsorted.bam'
            run('historical_bsmap', [args.bsmap, '-a', mates[0], '-b', mates[1],
                '-d', args.reference, '-R', '-o', source, '-p', str(alignment_threads)])
            if args.mode == 'trimmed' and [fingerprint(path) for path in mates] != metadata['trimmed_sources']:
                raise ValueError('Original trimmed mates changed while recovering')
        else:
            source = args.source_bam
        run('input_quickcheck', [args.samtools, 'quickcheck', '-v', source])
        expected = count('full_source_decode_count', source, args.mode != 'mapped')
        metadata['expected_filtered_records'] = expected
        if args.mode == 'mapped':
            mapped = source
        else:
            mapped = args.out_dir / 'mapped.bam'
            with mapped.open('xb') as output:
                run('historical_filter', [args.samtools, 'view', '-@', str(samtools_threads),
                    '-F', '268', '-b', source], stdout=output)
            run('filtered_quickcheck', [args.samtools, 'quickcheck', '-v', mapped])
        sorted_bam = args.out_dir / 'sorted.bam'
        # --sort-memory only sizes the in-memory batches (fewer temp files to merge);
        # samtools sort is stable, so record order is unchanged.
        sort_memory = ['-m', args.sort_memory] if args.sort_memory else []
        # --sort-tmp-dir puts temp files on node-local disk: merging hundreds of temp files
        # on a busy BeeGFS is latency-bound (measured 0.06-0.4 MB/s on DFS3b, 2026-10-15).
        sort_tmp = ['-T', args.sort_tmp_dir / 'sorted.tmp'] if args.sort_tmp_dir else []
        run('historical_sort', [args.samtools, 'sort', '-@', str(samtools_threads), *sort_memory, *sort_tmp,
            '-o', sorted_bam, mapped])
        run('sorted_quickcheck', [args.samtools, 'quickcheck', '-v', sorted_bam])
        actual = count('full_sorted_decode_count', sorted_bam)
        if actual != expected:
            raise ValueError(f'Record-count mismatch: expected {expected}, got {actual}')
        run('historical_index', [args.samtools, 'index', '-@', str(samtools_threads), sorted_bam])
        with (args.out_dir / 'flagstat.txt').open('x') as output:
            run('flagstat', [args.samtools, 'flagstat', '-@', str(samtools_threads), sorted_bam], stdout=output)
        if before is not None and fingerprint(source) != before:
            raise ValueError('Original BAM changed while recovering')
        metadata['validated_sorted_records'] = actual
        if not args.bam_only and args.call_processes > 1:
            # Same CHALM.py trad commands, one process per contig, joined in CHALM.py's
            # sorted() contig order (src/r2_2_epic_window_call.py).
            sys.path.insert(0, str(REVIEW))
            from src.r2_2_epic_window_call import call_by_contig
            metadata['stage'] = 'calls_per_contig'
            metadata['commands'].append({'stage': 'calls_per_contig', 'processes': args.call_processes,
                                         'contig_fasta_dir': str(args.contig_fasta_dir),
                                         'started_utc': datetime.now(timezone.utc).isoformat()})
            save()
            calls = call_by_contig(sorted_bam, args.out_dir, args.contig_fasta_dir, Path(args.chalm),
                                   Path(args.old_samtools), args.python, 4, args.samtools, args.call_processes)
            metadata['commands'][-1]['call_metadata'] = calls
            metadata['commands'][-1]['exit_code'] = 0
            metadata['call_sha256'] = {t: digest(args.out_dir / f'{t}.txt') for t in ['methRatio', 'CHALM']}
        elif not args.bam_only:
            for target, read_level in [('methRatio', False), ('CHALM', True)]:
                out = args.out_dir / f'{target}.txt'
                run(target, caller_command(args.python, args.chalm, args.reference,
                    args.old_samtools, sorted_bam, out, read_level))
                if not out.exists() or out.stat().st_size == 0:
                    raise ValueError(f'Missing caller output: {out}')
            metadata['call_sha256'] = {t: digest(args.out_dir / f'{t}.txt') for t in ['methRatio', 'CHALM']}
        metadata['state'] = 'bam_validated' if args.bam_only else 'native_calls_complete_unverified_archive'
        save()
    except Exception as exc:
        metadata['state'] = 'failed'
        metadata['error'] = repr(exc)
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-id')
    p.add_argument('--mode', choices=['mapped', 'unsorted', 'trimmed', 'raw'])
    p.add_argument('--original-dir', type=Path)
    p.add_argument('--source-bam', type=Path)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--task-manifest', type=Path)
    p.add_argument('--task-index', type=int)
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    p.add_argument('--expected-bytes', type=int)
    p.add_argument('--expected-mtime-ns', type=int)
    p.add_argument('--samtools', default='samtools')
    p.add_argument('--threads', type=int, default=40, help='CPU budget for BSMAP and samtools, capped at the Slurm allocation')
    p.add_argument('--reference', default=REFERENCE)
    p.add_argument('--chalm', default=CHALM)
    p.add_argument('--bsmap', default=BSMAP)
    p.add_argument('--python', default=PYTHON)
    p.add_argument('--old-samtools', default=OLD_SAM)
    p.add_argument('--bam-only', action='store_true', help='Validate BAM mechanics only; no native-call completion claim')
    p.add_argument('--sort-memory', help='samtools sort -m per thread (e.g. 8G); default keeps samtools default')
    p.add_argument('--sort-tmp-dir', type=Path, help='samtools sort -T directory (node-local); default: next to the output')
    p.add_argument('--call-processes', type=int, default=1,
                   help='>1: run the two historical calls per contig in parallel (needs --contig-fasta-dir)')
    p.add_argument('--contig-fasta-dir', type=Path, help='One-contig FASTA files from r2_2_epic_window_call split-reference')
    args = p.parse_args()
    if args.task_manifest:
        with args.task_manifest.open(newline='') as handle:
            tasks = list(csv.DictReader(handle, delimiter='\t'))
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            p.error('Recovery task index out of range')
        row = tasks[args.task_index]
        args.sample_id, args.mode = row['sample_id'], row['mode']
        args.original_dir, args.out_dir = Path(row['original_dir']), Path(row['out_dir'])
        args.source_bam = Path(row['source_bam']) if row['source_bam'] else None
        args.expected_bytes = int(row['source_bytes']) if row['source_bytes'] else None
        args.expected_mtime_ns = int(row['source_mtime_ns']) if row['source_mtime_ns'] else None
    if not all([args.sample_id, args.mode, args.original_dir, args.out_dir]):
        p.error('Specify one frozen task or explicit sample/mode/original-dir/out-dir')
    recover(args)


if __name__ == '__main__':
    main()
