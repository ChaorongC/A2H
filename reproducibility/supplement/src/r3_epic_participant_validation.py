"""R3-S4/S5: remask saved external predictions to audited EPIC loci."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / 'results'
SOURCE = ROOT / 'pgpuk_depth10_calibrated'
E1 = ROOT / 'r2_2_sensitivity_20261011/E1'
PUBLIC = HERE.parent / 'PGP_UK_blood_array_WGBS_mapping.tsv'
LINKS = ROOT / 'pgpuk_pairing_validation_20260930/existing_manifest_comparison.tsv'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def participant_metrics(observed, predicted):
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require identical finite participant observations')
    lower, upper = float(y.min()), float(y.max())
    span = upper - lower
    error = p - y
    rmse = float(np.sqrt(np.mean(error ** 2)))
    varying = len(y) > 1 and span > 0 and np.ptp(p) > 0
    return {'n': len(y), 'pearson_r': float(np.corrcoef(y, p)[0, 1]) if varying else np.nan,
            'rmse': rmse, 'mae': float(np.mean(np.abs(error))),
            'bias_pred_minus_obs': float(error.mean()), 'observed_min': lower,
            'observed_max': upper, 'observed_range': span,
            'nrmse': rmse / span if span > 0 else np.nan,
            'nrmse_status': 'defined' if span > 0 else 'zero_observed_range',
            'prediction_min': float(p.min()), 'prediction_max': float(p.max())}


def metadata():
    public = pd.read_csv(PUBLIC, sep='\t')
    links = pd.read_csv(LINKS, sep='\t')
    if len(public) != 10 or len(links) != 10 or not links.status.eq('MATCH').all():
        raise ValueError('Expected the ten verified published blood links')
    for key in ['pgp_id', 'array_id', 'wgbs_experiment', 'wgbs_run']:
        if public[key].isna().any() or public[key].duplicated().any():
            raise ValueError('Missing or duplicate public metadata key')
    if not public.tissue.eq('blood').all() or links.local_array_id.duplicated().any():
        raise ValueError('Unexpected tissue or duplicate local IDs')
    joined = public.merge(links, on='pgp_id', how='left', validate='one_to_one', sort=False)
    if (joined.pgp_id.tolist() != public.pgp_id.tolist() or joined.local_array_id.isna().any()
            or not joined.array_id.eq(joined.authoritative_array_id).all()
            or not joined.wgbs_run.eq(joined.manifest_wgbs_run).all()):
        raise ValueError('Public and local metadata disagree')
    return joined.rename(columns={'local_array_id': 'sample_id'})


def render_one(pair, frames, metrics, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout='constrained')
    for ax, target in zip(axes, ['CHALM', 'CAMDA']):
        part = frames[target].loc[frames[target].sample_id == pair.sample_id]
        row = metrics.loc[(metrics.sample_id == pair.sample_id) & (metrics.target == target)].iloc[0]
        y, p = part.observed.to_numpy(), part.A2H.to_numpy()
        lo = min(-.02, float(p.min()) - .02)
        hi = max(1.02, float(p.max()) + .02)
        ax.hexbin(y, p, gridsize=65, mincnt=1, bins='log', cmap='viridis',
                  extent=(lo, hi, lo, hi), rasterized=True)
        ax.plot([lo, hi], [lo, hi], ls='--', color='#aa4444', lw=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect='equal',
               xlabel=f'Observed WGBS {target}', ylabel=f'A2H-estimated {target}', title=target)
        ax.text(.03, .97, f'n = {int(row.n):,}\nr = {row.pearson_r:.4f}\nRMSE = {row.rmse:.5f}\nNRMSE = {row.nrmse:.5f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .95, 'pad': 4})
    fig.suptitle(f'{pair.pgp_id} | blood | {pair.array_id} | {pair.wgbs_run}', fontsize=10)
    prefix = output / f'{pair.sample_id}_{pair.pgp_id}'
    fig.savefig(prefix.with_suffix('.png'), dpi=200)
    fig.savefig(prefix.with_suffix('.pdf'))
    plt.close(fig)


def run(output, pilot=False):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    pairs = metadata()
    ids = pairs.sample_id.tolist()
    prep = json.loads((E1/'preparation.json').read_text())
    mapping = pd.read_csv(E1/'epic_mapping.tsv', sep='\t')
    if len(mapping) != 824209 or mapping.probe_id.duplicated().any() or mapping.locus.duplicated().any():
        raise ValueError('Unexpected one-to-one audited EPIC mapping')
    frames, flow, excluded, rows = {}, [], [], []
    inputs = {str(p.resolve()): sha256(p) for p in [PUBLIC, LINKS, E1/'epic_mapping.tsv', E1/'preparation.json']}
    report_pairs = pairs.iloc[:1] if pilot else pairs
    for target in ['CHALM', 'CAMDA']:
        path = SOURCE/f'{target}_observations.parquet'
        h = sha256(path)
        if h != prep['input'][f'external_{target}']['sha256']:
            raise ValueError('Saved source observations changed since E1 preparation')
        inputs[str(path.resolve())] = h
        original = pd.read_parquet(path, columns=['probe_id', 'sample_id', 'm', 'observed', 'A2H'])
        if original.duplicated(['probe_id', 'sample_id']).any() or set(original.sample_id) != set(ids):
            raise ValueError('Duplicated or unexpected source participant/probe keys')
        not_mapped = original.loc[~original.probe_id.isin(mapping.probe_id), ['probe_id']].drop_duplicates()
        not_mapped['target'] = target
        not_mapped['reason'] = 'absent_from_audited_one_to_one_EPIC_map'
        excluded.append(not_mapped)
        f = original.merge(mapping[['probe_id', 'locus']], on='probe_id', how='inner', validate='many_to_one')
        if not np.isfinite(f[['m', 'observed', 'A2H']]).all().all():
            raise ValueError('Nonfinite saved inputs; no new imputation or filtering allowed')
        if not f.observed.between(0, 1).all() or not f.m.between(0, 1).all():
            raise ValueError('Observed fraction outside domain')
        expected = prep['flow']['targets'][target]
        if len(f) != expected['n_pairs'] or f.probe_id.nunique() != expected['n_probes']:
            raise ValueError('Remasked keys differ from completed E1 external universe')
        frames[target] = f
        flow.append({'target': target, 'source_loci': original.probe_id.nunique(),
            'source_observations': len(original), 'retained_epic_loci': f.probe_id.nunique(),
            'retained_observations': len(f), 'excluded_loci': len(not_mapped),
            'additional_support_filters': 0, 'complete_ten_filter': False})
        for pair in report_pairs.itertuples(index=False):
            local = f.loc[f.sample_id == pair.sample_id]
            metrics = participant_metrics(local.observed, local.A2H)
            rows.append({'target': target, 'method': 'A2H', 'sample_id': pair.sample_id,
                'pgp_id': pair.pgp_id, 'tissue': 'blood', 'array_id': pair.array_id,
                'wgbs_experiment': pair.wgbs_experiment, 'wgbs_run': pair.wgbs_run, **metrics})
    ch = frames['CHALM'].set_index(['probe_id', 'sample_id']).sort_index()
    ca = frames['CAMDA'].set_index(['probe_id', 'sample_id']).sort_index()
    if not ch.index.equals(ca.index):
        raise ValueError('CHALM/CAMDA masks differ; no implicit reordering')
    np.testing.assert_array_equal(ch.m, ca.m)
    metric = pd.DataFrame(rows)
    output.mkdir(parents=True)
    pair_columns = ['pgp_id', 'tissue', 'array_id', 'wgbs_experiment', 'wgbs_run', 'sample_id']
    pairs[pair_columns].to_csv(output/'pairing_key.tsv', sep='\t', index=False)
    metric.to_csv(output/'sample_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(flow).to_csv(output/'eligibility_flow.tsv', sep='\t', index=False)
    pd.concat(excluded).to_csv(output/'excluded_probes.tsv', sep='\t', index=False)
    for pair in report_pairs.itertuples(index=False):
        render_one(pair, frames, metric, output)
    summary = metric.groupby('target').agg(n_participants=('sample_id', 'nunique'),
        n_observations=('n', 'sum'), min_sites_per_participant=('n', 'min'),
        max_sites_per_participant=('n', 'max'), min_pearson=('pearson_r', 'min'),
        max_pearson=('pearson_r', 'max'), min_rmse=('rmse', 'min'), max_rmse=('rmse', 'max'),
        min_observed_range=('observed_range', 'min'), max_observed_range=('observed_range', 'max'),
        min_nrmse=('nrmse', 'min'), max_nrmse=('nrmse', 'max')).reset_index()
    summary.to_csv(output/'target_summary.tsv', sep='\t', index=False)
    report = ('# Reviewer 3 EPIC Participant Validation\n\n'
        f"{'Pilot' if pilot else 'Full validation'}: {len(report_pairs)} reported blood participants. "
        'Saved original-depth A2H predictions and depth10 WGBS truth are unchanged; only the '
        'audited EPIC mapping restriction is applied. No refitting, added support cutoff or '
        'complete-ten restriction. Pairing uses explicit metadata keys, not accession order.\n\n'
        'Per-participant RMSE is pooled over that participant\'s eligible CpGs. NRMSE divides '
        'that RMSE by maximum minus minimum observed WGBS target within the same participant '
        'and target. Zero observed range gives undefined NRMSE. Pearson is across CpGs in one '
        'participant, not across participants at a CpG. Prediction tails are not clipped.\n\n'
        'The 450K arrays are restricted to their audited EPIC overlap. The denominator is '
        'therefore distinct from historical broader plots, complete-ten grouped validation '
        'and the common-three-model reference-composition comparison.\n')
    (output/'report.md').write_text(report)
    completion = {'pilot': pilot, 'n_reported_participants': len(report_pairs), 'flow': flow,
        'source_training_depth': 4, 'external_truth_min_depth': 10,
        'nrmse_definition': 'participant RMSE / participant observed target range',
        'metadata_pairing': 'published pgp_id to validated local sample ID; arrays and ERR runs cross-checked',
        'input_sha256': inputs, 'code_sha256': sha256(Path(__file__)),
        'output_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    (output/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    run(args.output_dir, pilot=args.pilot)
