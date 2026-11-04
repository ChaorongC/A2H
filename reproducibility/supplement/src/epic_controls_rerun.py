"""Rerun the frozen external evaluation of an EPIC cohort with comparators covering all EPIC loci.

The source branch (prepared inputs, WGBS truth and deployed-A2H outputs) is reused read-only through
symlinks; only the comparator source changes, from results/reference_controls_original (HM450
loci) to results/reference_controls_epic_20261017 (same values on those loci, plus EPIC-only loci;
locked universal curve). Computation is the unchanged src/gse128734_external.evaluate, followed by
the A2H-from-WGBS-beta comparison with the definitions of results/external_beta_decomposition_20261015.
Run from R/: python -m src.epic_controls_rerun {GSE240482_EPICv1|GSE128734_EPIC} --output-dir <new branch>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import src.gse128734_external as g128
import src.gse240482_external as g240
from src.evaluate_pgpuk import file_hash, reference_control_predictions, scalar_metrics

HIST = g128.HIST
CONTROLS = HIST / 'results/reference_controls_epic_20261017'
SETS = {
    'GSE240482_EPICv1': dict(source=HIST / 'results/gse240482_external_20261017', cohort='GSE240482', platform='EPICv1',
                             report=g240.report),
    'GSE128734_EPIC': dict(source=HIST / 'results/gse128734_external_20261014/EPIC', cohort='GSE128734', platform='EPIC',
                           report=None),
}
LINKED = ('primary_pairs.tsv', 'all_array_pairs.tsv', 'excluded_pairs.tsv', 'mapping.tsv', 'mapping_audit.tsv',
          'array_beta.tsv', 'truth', 'transformation')


def controls_epic(mapping: pd.DataFrame, beta: np.ndarray, target: str) -> dict:
    prov = json.loads((CONTROLS / 'provenance.json').read_text())
    path = CONTROLS / f'{target}_controls.parquet'
    if file_hash(path) != prov[target]['output_sha256'] or not prov[target]['locked_rows_reproduced_exactly']:
        raise ValueError('Extended EPIC controls changed or did not reproduce the locked rows')
    locked = json.loads((HIST / 'results/reference_controls_original/provenance.json').read_text())
    coefficients = prov[target]['universal_coefficients_intercept_linear_quadratic']
    if coefficients != locked[target]['universal_coefficients_intercept_linear_quadratic']:
        raise ValueError('Universal curve differs from the locked coefficients')
    return reference_control_predictions(beta, mapping.locus, pd.read_parquet(path), coefficients)


def setup(source: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f'Refusing to reuse {out}')
    if not (source / 'analysis_completion.json').exists():
        raise ValueError('Source branch has no completed evaluation')
    out.mkdir(parents=True)
    shutil.copy2(source / 'preparation.json', out / 'preparation.json')
    for name in LINKED:
        os.symlink((source / name).resolve(), out / name)


def wgbs_input(out: Path, set_name: str, cohort: str, platform: str) -> None:
    """Deployed A2H on WGBS beta (same CLI flags), scored on this branch's depth>=10 observation set."""
    prep = g128.prepared(out)
    root = out / 'a2h_wgbs_input'
    root.mkdir(exist_ok=False)
    pairs = pd.read_csv(out / 'primary_pairs.tsv', sep='\t')
    mapping = pd.read_csv(out / 'mapping.tsv', sep='\t')
    columns = {}
    for row in pairs.itertuples(index=False):
        frame = pd.read_parquet(out / 'truth' / f'{row.array_id}.parquet')
        if not frame.probe_id.equals(mapping.probe_id):
            raise ValueError('Truth probe order mismatch')
        columns[row.array_id] = frame.beta.to_numpy()
    matrix = pd.DataFrame(columns, index=pd.Index(mapping.probe_id, name='Probe_ID')).dropna(how='all')
    input_path = root / 'wgbs_beta_input.tsv'
    matrix.to_csv(input_path, sep='\t')
    prefix = f'{prep["prefix"]}_wgbs'
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    command = [g128.PYTHON, str(g128.MODEL / 'A2H.py'), '-i', str(input_path), '-o', str(root), '--prefix', prefix,
               '--ref', 'hg38', '--beadchip', 'EPIC', '-t', str(threads), '--probe', '--chalm', '--camda', '--filter', '0.05',
               '--cpg-info', str(g128.MODEL / 'A2H_cpg_info.pkl'), '-c', str(g128.MODEL / 'A2H_coef.pkl'), '--verbose', '0']
    subprocess.run(command, check=True)
    support = g240.read_matrix(root / f'{prefix}.confidence.tsv', 'support_wgbs')
    coef = pd.read_pickle(g128.MODEL / 'A2H_coef.pkl').set_index('featureID')
    locus = mapping.set_index('probe_id').locus
    summary, per_sample, checks = [], [], {}
    for target in g128.TARGETS:
        obs = pd.read_parquet(out / f'{target}_observations.parquet')
        pred = g240.read_matrix(root / f'{prefix}.{target.lower()}.tsv', 'A2H_wgbs')
        key = pd.MultiIndex.from_frame(obs[['probe_id', 'sample_id']])
        obs['A2H_wgbs'] = pred.reindex(key).to_numpy()
        obs['support_wgbs'] = support.reindex(key).to_numpy()
        n_all = len(obs)
        obs = obs[np.isfinite(obs[['WGBS_beta', 'A2H_wgbs']]).all(axis=1)].copy()
        sample = obs.iloc[np.linspace(0, len(obs) - 1, min(500, len(obs)), dtype=int)]
        column = 'chalm_coef' if target == 'CHALM' else 'camda_coef'
        for r in sample.itertuples(index=False):
            c = coef.at[locus[r.probe_id], column]
            expected = float(c) if np.isscalar(c) else float(np.polyval(c, r.WGBS_beta))
            if not np.isclose(r.A2H_wgbs, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f'A2H CLI output differs from polyval at {r.probe_id}/{r.sample_id}')
        checks[target] = {'depth10_observations': n_all, 'with_wgbs_input_prediction': len(obs), 'polyval_rows_checked': len(sample)}
        comparisons = {'beta_array_vs_wgbs': ('WGBS_beta', 'm', obs), 'A2H_array_vs_truth': ('observed', 'A2H', obs),
                       'A2H_wgbs_vs_truth': ('observed', 'A2H_wgbs', obs),
                       'A2H_wgbs_vs_truth_wgbs_support_ge_0.05': ('observed', 'A2H_wgbs', obs[obs.support_wgbs >= .05])}
        for comparison, (y, p, sub) in comparisons.items():
            summary.append({'set': set_name, 'cohort': cohort, 'platform': platform, 'target': target, 'comparison': comparison,
                            'mask': 'native A2H observations on all EPIC loci with comparators (depth>=10, array support>=0.05) '
                                    'with finite WGBS beta' + ('; WGBS-input support >= 0.05' if 'support' in comparison else ''),
                            **g240.metric_row(sub, y, p)})
            for s, part in sub.groupby('sample_id', sort=True):
                m = scalar_metrics(part[y].to_numpy(), part[p].to_numpy())
                per_sample.append({'set': set_name, 'target': target, 'comparison': comparison, 'sample_id': s,
                                   'n': m['n'], 'pearson_r': m['pearson_r'], 'rmse': m['rmse']})
    table = pd.DataFrame(summary)
    table.to_csv(out / 'decomposition_summary.tsv', sep='\t', index=False)
    pd.DataFrame(per_sample).to_csv(out / 'decomposition_per_sample.tsv', sep='\t', index=False)
    lines = [f'# {set_name}: A2H error from array beta vs WGBS beta input (all EPIC loci)', '',
             'Depth >=10 observation set of this branch (array-input support >=0.05), restricted to finite WGBS beta. '
             'Definitions match results/external_beta_decomposition_20261015. WGBS beta and the WGBS targets share reads, '
             'so A2H-from-WGBS-beta is an optimistic reference for the mapping error.', '',
             '| Target | Comparison | n | CpGs | Mean per-CpG RMSE | Pooled RMSE | Pooled r | Mean signed error |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table.itertuples(index=False):
        lines.append(f'| {r.target} | {r.comparison} | {r.n:,} | {r.n_sites:,} | {r.mean_site_rmse:.4f} | {r.rmse:.4f} | '
                     f'{r.pearson_r:.3f} | {r.mean_signed_error:+.4f} |')
    (out / 'DECOMPOSITION.md').write_text('\n'.join(lines) + '\n')
    g128.save(root / 'completion.json', {'state': 'complete', 'command': command, 'checks': checks,
                                          'input': g128.record(input_path),
                                          'outputs': {p.name: g128.record(p) for p in sorted(root.glob(f'{prefix}.*.tsv'))},
                                          'summary': g128.record(out / 'decomposition_summary.tsv'), 'finished_utc': g128.utc()})


def run(set_name: str, out: Path) -> None:
    spec = SETS[set_name]
    setup(spec['source'], out)
    g128.controls = controls_epic
    if spec['report'] is not None:
        g128.report = spec['report']
    g128.evaluate(out)
    wgbs_input(out, set_name, spec['cohort'], spec['platform'])
    g128.save(out / 'epic_controls_rerun.json', {
        'state': 'complete', 'source_branch': str(spec['source']), 'controls': str(CONTROLS),
        'controls_provenance_sha256': file_hash(CONTROLS / 'provenance.json'),
        'linked_read_only': list(LINKED), 'code_sha256': file_hash(Path(__file__)), 'finished_utc': g128.utc()})
    print('Rerun complete', set_name, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('set_name', choices=sorted(SETS))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.set_name, args.output_dir)


if __name__ == '__main__':
    main()
