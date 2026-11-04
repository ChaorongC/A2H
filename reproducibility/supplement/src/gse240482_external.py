"""GSE240482 (six primary prostate tumours, EPIC v1 array + WGBS) on the frozen external pipeline.

Array beta and WGBS truth come from the same paired_validation_v2 processing that produced the
PGP-UK truth (Trim Galore clip 10/20 + 3' 10/10, BSMAP -R, CHALM-fast trad -m 4). The
evaluation reuses src/gse128734_external.py unchanged (transform, evaluate: A2H vs m, m^2, m^3,
reference mean/linear and universal-quadratic controls, R1-5 tissue-exclusion models, complete-
sample disjoint groups, per-sample panels, site errors). Only cohort-specific input selection,
the report text, and an extra step are new:

  wgbs-input  deployed A2H applied to the WGBS beta of the same sample (same CLI flags), scored
              against WGBS CHALM/CAMDA on the depth>=10 observation set, with the definitions of
              results/external_beta_decomposition_20261015 (array beta vs WGBS beta, A2H from
              array beta, A2H from WGBS beta). WGBS beta and truth share reads, so the WGBS-input
              error is an optimistic reference, not an independent-platform result.

Run from R/: python -m src.gse240482_external {prepare|truth --index i|transform|evaluate|wgbs-input}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import src.gse128734_external as g128
from src.evaluate_pgpuk import file_hash, scalar_metrics
from src.tissue_only_transfer import epic_mapping

HIST, MODEL, EPIC, CONTROL, TRANSFER, PYTHON = g128.HIST, g128.MODEL, g128.EPIC, g128.CONTROL, g128.TRANSFER, g128.PYTHON
TARGETS = g128.TARGETS
SOURCE = g128.SOURCE
V2 = SOURCE / 'results/paired_validation_v2'
META = SOURCE / 'metadata/paired_validation_v2'
OUT = HIST / 'results/gse240482_external_20261017'
PREFIX = 'gse240482_EPICv1'
STUDY = 'epicv2_eval'
BETA = V2 / 'array/gse240482_prostate_epicv1_EPIC.beta.tsv'
QC = V2 / 'array/gse240482_prostate_epicv1_EPIC.sample_qc.tsv'
MAY_A2H = V2 / 'a2h_array'
MIN_SITE_SAMPLES = 5  # per-CpG summary: >=5 of 6 samples, proportional to PGP-UK's >=8 of 10


def wgbs_paths(wgbs_id: str) -> dict[str, Path]:
    return {'beta': V2 / f'wgbs/traditional/{wgbs_id}_methRatio.txt', 'CHALM': V2 / f'wgbs/chalm/{wgbs_id}_CHALM.txt'}


def array_id(column: str) -> str:
    return column.split('_')[0]


def prepare(out: Path) -> None:
    if (out / 'preparation.json').exists():
        raise FileExistsError('Analysis preparation already exists')
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(META / 'paired_samples.tsv', sep='\t').query('study == @STUDY').copy()
    if len(pairs) != 6 or pairs.wgbs_id.duplicated().any() or pairs.array_id.duplicated().any():
        raise ValueError('Expected six one-to-one GSE240482 pairs')
    if not pairs.tissue.eq('prostate_tumor').all() or not pairs.platform.eq('EPIC').all():
        raise ValueError('Unexpected tissue or platform')
    runs = pd.read_csv(META / 'paired_wgbs_run_manifest.tsv', sep='\t').query('study == @STUDY')
    if set(runs.wgbs_id) != set(pairs.wgbs_id) or not runs.process_status.eq('complete').all():
        raise ValueError('WGBS runs incomplete for GSE240482')
    qc = pd.read_csv(QC, sep='\t')
    if not qc.kept.astype(str).str.upper().eq('TRUE').all() or set(qc.array_id.map(array_id)) != set(pairs.array_id):
        raise ValueError('Array sample QC does not cover all six kept arrays')
    beta = pd.read_csv(BETA, sep='\t', index_col=0)
    beta.columns = [array_id(c) for c in beta.columns]
    if set(beta.columns) != set(pairs.array_id) or beta.index.has_duplicates:
        raise ValueError('Array beta columns do not match the six pairs')
    if not np.isfinite(beta.to_numpy()).all() or not ((beta >= 0) & (beta <= 1)).all().all():
        raise ValueError('Invalid array beta values')
    mapping = epic_mapping(pd.read_csv(EPIC, sep='\t', usecols=['probeID', 'CpG_chrm', 'CpG_beg']),
                           pd.read_csv(MODEL / 'A2H_cpg_probe.hg38.tsv', sep='\t',
                                       usecols=['loci.hg38', 'EPIC.hg38', 'HM450.hg38']),
                           allow_conflicts=True)
    coef = pd.read_pickle(MODEL / 'A2H_coef.pkl')
    info = pd.read_pickle(MODEL / 'A2H_cpg_info.pkl')
    if coef.featureID.duplicated().any() or info.index.has_duplicates:
        raise ValueError('Ambiguous deployed locus coefficients/info')
    mapping['in_array'] = mapping.probe_id.isin(beta.index)
    mapping['in_450k'] = mapping.HM450_locus.notna()
    mapping['has_model'] = mapping.locus.isin(coef.featureID) & mapping.locus.isin(info.index)
    mapping['eligible'] = mapping.eligible_mapping & mapping.in_array & mapping.has_model
    mapping.to_csv(out / 'mapping_audit.tsv', sep='\t', index=False)
    eligible = mapping.loc[mapping.eligible, ['probe_id', 'locus']].sort_values('probe_id').reset_index(drop=True)
    if eligible.probe_id.duplicated().any() or len(eligible) < 100:
        raise ValueError('Invalid or empty compatible mapping')
    eligible.to_csv(out / 'mapping.tsv', sep='\t', index=False)
    primary = pd.DataFrame({'array_id': pairs.array_id, 'wgbs_id': pairs.wgbs_id, 'sample_id': pairs.subject_or_cell,
                            'participant_id': pairs.subject_or_cell, 'tissue': pairs.tissue,
                            'tissue_label': 'prostate tumour', 'platform': 'EPIC v1', 'replicate': 1, 'primary': True,
                            'metadata_status': 'one EPIC v1 array per WGBS sample'}).reset_index(drop=True)
    primary.to_csv(out / 'primary_pairs.tsv', sep='\t', index=False)
    primary.to_csv(out / 'all_array_pairs.tsv', sep='\t', index=False)
    pd.DataFrame(columns=list(primary.columns) + ['reason']).to_csv(out / 'excluded_pairs.tsv', sep='\t', index=False)
    beta = beta.loc[eligible.probe_id, primary.array_id].astype(np.float64)
    beta.index.name = 'Probe_ID'
    beta.to_csv(out / 'array_beta.tsv', sep='\t')
    wgbs = {row.wgbs_id: {k: g128.record(p) for k, p in wgbs_paths(row.wgbs_id).items()} for row in primary.itertuples()}
    inputs = {'pairs': META / 'paired_samples.tsv', 'runs': META / 'paired_wgbs_run_manifest.tsv',
              'protocol_flags': META / 'bsmap_protocol_flags.tsv', 'beta': BETA, 'array_qc': QC,
              'wgbs_processing_script': SOURCE / 'scripts/paired_validation_v2/process_wgbs_sample.sh',
              'model_script': MODEL / 'A2H.py', 'coefficients': MODEL / 'A2H_coef.pkl',
              'model_info': MODEL / 'A2H_cpg_info.pkl', 'probe_map': MODEL / 'A2H_cpg_probe.hg38.tsv', 'epic': EPIC,
              'control_provenance': CONTROL / 'provenance.json', 'transfer_preparation': TRANSFER / 'preparation.json',
              'analysis_script': Path(__file__), 'shared_pipeline': Path(g128.__file__),
              'legacy_metrics': HIST / 'src/evaluate_pgpuk.py', 'legacy_grouping': HIST / 'src/run_pgpuk_disjoint_complete10.py',
              'legacy_mapping': HIST / 'src/tissue_only_transfer.py'}
    for name in ('primary_pairs.tsv', 'mapping.tsv', 'array_beta.tsv', 'all_array_pairs.tsv', 'excluded_pairs.tsv'):
        inputs[name] = out / name
    pgp = json.loads((HIST / 'results/pgpuk_depth10_calibrated/provenance.json').read_text())
    for target in TARGETS:
        p = CONTROL / f'{target}_controls.parquet'
        if file_hash(p) != pgp['file_sha256'][f'control_{target}']:
            raise ValueError('Reference controls differ from the verified PGP-UK inputs')
        inputs[f'{target}_controls'] = p
    g128.save(out / 'preparation.json', {
        'created_utc': g128.utc(), 'state': 'prepared', 'frozen_inputs': {k: g128.record(p) for k, p in inputs.items()},
        'wgbs_inputs': wgbs, 'compatible_probes': len(eligible),
        'coordinate_conflicts_in_array': int((mapping.in_array & ~mapping.eligible_mapping).sum()),
        'platform': 'EPIC', 'prefix': PREFIX, 'primary_samples': len(primary),
        'cohort': 'GSE240482 (SuperSeries; arrays GSE240412, WGBS GSE240481); Peters et al. BMC Genomics 2024;25:251',
        'biological_units': 'six primary prostate tumours (one array and one WGBS library each)',
        'wgbs_library': 'CEGX TrueMethyl Whole-Genome v3.1, HiSeq X Ten (GEO); processed with the PGP-UK v2 protocol, no Swift clip',
        'training_depth': 4, 'external_primary_depth': 10, 'support_cutoff': .05,
        'grouping_complete_samples': len(primary), 'group_size': 10, 'mean_block_size': 100, 'variance_ddof': 0,
        'site_summary_min_samples': MIN_SITE_SAMPLES, 'new_support_refilter': False, 'no_external_training': True})
    print('Prepared', len(eligible), 'compatible probes,', len(primary), 'pairs', flush=True)


def truth(out: Path, index: int) -> None:
    prep = g128.prepared(out)
    pairs = pd.read_csv(out / 'primary_pairs.tsv', sep='\t')
    if not 0 <= index < len(pairs):
        raise ValueError('Invalid explicit sample task')
    row = pairs.iloc[index]
    dest = out / 'truth' / f'{row.array_id}.parquet'
    if dest.exists():
        raise FileExistsError('Truth output already exists')
    mapping = pd.read_csv(out / 'mapping.tsv', sep='\t')
    tables, inputs = {}, {}
    for key, path in wgbs_paths(row.wgbs_id).items():
        expected = prep['wgbs_inputs'][row.wgbs_id][key]
        if file_hash(path) != expected['sha256']:
            raise ValueError(f'WGBS table changed since preparation: {path}')
        inputs[key] = expected
        tables[key] = g128.read_truth(path, mapping).reindex(mapping.locus)
    result = mapping.copy()
    for key in ('beta', 'CHALM'):
        result[key] = tables[key].ratio.to_numpy(dtype=np.float64)
        result[key + '_depth'] = tables[key].CT_count.to_numpy(dtype=np.float64)
    result['CAMDA'] = result.CHALM - result.beta
    if result.loc[result.CAMDA.notna(), 'CAMDA'].lt(-.0011).any():
        raise ValueError('Direct CHALM-beta identity/range requires review')
    dest.parent.mkdir(exist_ok=True)
    result.to_parquet(dest, index=False)
    g128.save(dest.with_suffix('.json'), {
        'state': 'complete', 'wgbs_id': row.wgbs_id, 'array': row.array_id, 'inputs': inputs,
        'output': g128.record(dest), 'finished_utc': g128.utc(),
        'depth10_joint_sites': int((result.beta_depth.ge(10) & result.CHALM_depth.ge(10)).sum()),
        'CAMDA_definition': 'aligned CHALM ratio minus traditional beta; both count thresholds required'})
    print('Truth complete', row.array_id, flush=True)


def transform(out: Path) -> None:
    """Deployed A2H on the array beta (shared step), then a known-answer check against the May run."""
    g128.transform(out)
    root = out / 'transformation'
    beta = pd.read_csv(out / 'array_beta.tsv', sep='\t', index_col=0)
    checks = {}
    for kind in ('chalm', 'camda', 'confidence', 'chalm.filtered', 'camda.filtered'):
        new = pd.read_csv(root / f'{PREFIX}.{kind}.tsv', sep='\t', index_col=0)
        may_name = kind.replace('.filtered', '.filtered_005')
        may = pd.read_csv(MAY_A2H / f'gse240482_prostate_epicv1_EPIC.{may_name}.tsv', sep='\t', index_col=0)
        may.columns = [array_id(c) for c in may.columns]
        shared = new.index.intersection(may.index)
        a, b = new.loc[shared, beta.columns].to_numpy(float), may.loc[shared, beta.columns].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        checks[kind] = {'shared_probes': len(shared), 'new_probes': len(new), 'finite_pairs': int(both.sum()),
                        'nan_pattern_equal': bool((np.isfinite(a) == np.isfinite(b)).all()),
                        'max_abs_diff': float(np.max(np.abs(a[both] - b[both]))) if both.any() else None}
        if not checks[kind]['nan_pattern_equal'] or checks[kind]['max_abs_diff'] > 1e-9:
            raise ValueError(f'Rerun A2H differs from the May outputs for {kind}: {checks[kind]}')
    g128.save(root / 'may_reproduction.json', {'state': 'reproduced', 'checks': checks, 'finished_utc': g128.utc()})
    print('Transform reproduces May A2H outputs on shared probes', flush=True)


def report(out, metrics, transfer, groups, quality, prep):
    """Cohort-specific report for g128.evaluate (same tables as the GSE128734 report)."""
    primary = metrics.loc[metrics.depth.eq(10) & metrics.scope.eq('pooled')]
    site = pd.read_csv(out / 'site_metrics.tsv', sep='\t').query('depth == 10')
    lines = ['# GSE240482 (prostate tumour, EPIC v1) external A2H validation', '',
             f'Generated {g128.utc()}. Six primary prostate tumours (Peters et al., BMC Genomics 2024;25:251), each with one '
             'EPIC v1 array and one WGBS library (CEGX TrueMethyl v3.1). WGBS and array processing are the same '
             'paired_validation_v2 protocol used for the PGP-UK truth; no Swift clip applies.',
             f'Historical model training depth >=4; primary external truth >=10. {prep["compatible_probes"]:,} '
             'coordinate-compatible EPIC probes before truth/support missingness. Deployed A2H, support >=0.05 applied once.', '',
             '## R1-1: Comparators (pooled over CpG-sample pairs, depth >=10)', '',
             'Identical eligible observations within each target. Pooled RMSE differs from the mean per-CpG RMSE below.', '',
             '| Target | Method | Observations | Pearson r | RMSE | NRMSE |', '|---|---|---:|---:|---:|---:|']
    for row in primary.itertuples(index=False):
        lines.append(f'| {row.target} | {row.method} | {row.n:,} | {row.pearson_r:.5f} | {row.rmse:.5f} | {row.nrmse:.5f} |')
    lines += ['', f'## Mean per-CpG RMSE (depth >=10; CpGs with >={MIN_SITE_SAMPLES} of 6 samples)', '',
              f'The >={MIN_SITE_SAMPLES}-of-6 rule is the proportional analogue of the PGP-UK >=8-of-10 rule; each CpG has equal weight.', '',
              '| Target | Method | CpGs | Mean per-CpG RMSE | Median per-CpG RMSE |', '|---|---|---:|---:|---:|']
    for (target, method), frame in site.loc[site.n >= MIN_SITE_SAMPLES].groupby(['target', 'method'], sort=False):
        lines.append(f'| {target} | {method} | {len(frame):,} | {frame.rmse.mean():.5f} | {frame.rmse.median():.5f} |')
    lines += ['', '## R1-4: Disjoint matched mean/variance groups', '',
              f'{len(quality):,} disjoint groups of 10 CpGs complete across all six samples (60 observations per group/method); '
              'array-beta mean blocks of 100, then variance-sorted groups of 10.', '',
              '| Target | Method | Defined correlations | Median group r | Median group RMSE |', '|---|---|---:|---:|---:|']
    for (target, method), frame in groups.groupby(['target', 'method']):
        lines.append(f'| {target} | {method} | {frame.pearson_r.notna().sum():,} | {frame.pearson_r.median():.5f} | {frame.rmse.median():.5f} |')
    lines += ['', '## R1-5 and R2-4: Reference composition', '',
              'Frozen tissue-only exclusion models (161 samples without blood/immune; 122 without epithelial) on a common mask.', '',
              '| Target | Reference | Observations | Pearson r | RMSE |', '|---|---|---:|---:|---:|']
    for row in transfer.loc[transfer.scope.eq('pooled')].itertuples(index=False):
        lines.append(f'| {row.target} | {row.model} | {row.n:,} | {row.pearson_r:.5f} | {row.rmse:.5f} |')
    lines += ['', '## Scope', '',
              'Six tumours from one study; tumour purity and clonal composition are not modelled. These are disease-tissue '
              'transformation diagnostics, not evidence for SLE/AD tissues or a calibrated new-subject interval. Pooled and '
              'grouped correlations are descriptive; no within-CpG ranking claim. figures/ has one panel per tumour; '
              'site_metrics.tsv has per-CpG RMSE/bias; beta_array_wgbs_agreement.tsv has array-vs-WGBS beta agreement. '
              'The A2H-from-WGBS-beta comparison is in DECOMPOSITION.md.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def evaluate(out: Path) -> None:
    g128.report = report  # cohort-specific text; all computations are the shared GSE128734 evaluate()
    g128.evaluate(out)


def metric_row(frame: pd.DataFrame, y: str, pred: str) -> dict:
    """Same definitions as results/external_beta_decomposition_20261015 (metric_row)."""
    sub = frame[np.isfinite(frame[y]) & np.isfinite(frame[pred])]
    yy, pp = sub[y].to_numpy(), sub[pred].to_numpy()
    result = scalar_metrics(yy, pp)
    result.pop('nrmse', None)
    result['mae'] = float(np.mean(np.abs(yy - pp))) if len(yy) else np.nan
    site_rmse = np.sqrt(((sub[pred] - sub[y]) ** 2).groupby(sub['probe_id']).mean())
    result.update(n_sites=int(sub['probe_id'].nunique()), n_samples=int(sub['sample_id'].nunique()),
                  mean_site_rmse=float(site_rmse.mean()), median_site_rmse=float(site_rmse.median()),
                  mean_signed_error=float(np.mean(pp - yy)) if len(yy) else np.nan)
    return result


def read_matrix(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path, sep='\t', index_col=0)
    frame.index.name = 'probe_id'
    return frame.stack(future_stack=True).rename(name).rename_axis(['probe_id', 'sample_id'])


def wgbs_input(out: Path) -> None:
    prep = g128.prepared(out)
    if not (out / 'analysis_completion.json').exists():
        raise ValueError('Run evaluate first')
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
    prefix = f'{PREFIX}_wgbs'
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    command = [PYTHON, str(MODEL / 'A2H.py'), '-i', str(input_path), '-o', str(root), '--prefix', prefix, '--ref', 'hg38',
               '--beadchip', 'EPIC', '-t', str(threads), '--probe', '--chalm', '--camda', '--filter', '0.05',
               '--cpg-info', str(MODEL / 'A2H_cpg_info.pkl'), '-c', str(MODEL / 'A2H_coef.pkl'), '--verbose', '0']
    subprocess.run(command, check=True)
    support = read_matrix(root / f'{prefix}.confidence.tsv', 'support_wgbs')
    coef = pd.read_pickle(MODEL / 'A2H_coef.pkl').set_index('featureID')
    locus = mapping.set_index('probe_id').locus
    summary, per_sample, checks = [], [], {}
    for target in TARGETS:
        obs = pd.read_parquet(out / f'{target}_observations.parquet')
        pred = read_matrix(root / f'{prefix}.{target.lower()}.tsv', 'A2H_wgbs')
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
            summary.append({'set': 'GSE240482_EPICv1', 'cohort': 'GSE240482', 'platform': 'EPICv1', 'target': target,
                            'comparison': comparison,
                            'mask': 'rerun native A2H observations (depth>=10, array support>=0.05) with finite WGBS beta'
                                    + ('; WGBS-input support >= 0.05' if 'support' in comparison else ''),
                            **metric_row(sub, y, p)})
            for s, part in sub.groupby('sample_id', sort=True):
                m = scalar_metrics(part[y].to_numpy(), part[p].to_numpy())
                per_sample.append({'set': 'GSE240482_EPICv1', 'target': target, 'comparison': comparison, 'sample_id': s,
                                   'n': m['n'], 'pearson_r': m['pearson_r'], 'rmse': m['rmse']})
    pd.DataFrame(summary).to_csv(out / 'decomposition_summary.tsv', sep='\t', index=False)
    pd.DataFrame(per_sample).to_csv(out / 'decomposition_per_sample.tsv', sep='\t', index=False)
    table = pd.DataFrame(summary)
    lines = ['# GSE240482: A2H error from array beta vs WGBS beta input', '',
             'Depth >=10 observation set of the rerun (array-input support >=0.05), restricted to finite WGBS beta. '
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
                                          'summary': g128.record(out / 'decomposition_summary.tsv'),
                                          'finished_utc': g128.utc()})
    print('WGBS-input decomposition complete', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'truth', 'transform', 'evaluate', 'wgbs-input'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'truth':
        truth(args.output, args.index)
    elif args.action == 'transform':
        transform(args.output)
    elif args.action == 'evaluate':
        evaluate(args.output)
    else:
        wgbs_input(args.output)


if __name__ == '__main__':
    main()
