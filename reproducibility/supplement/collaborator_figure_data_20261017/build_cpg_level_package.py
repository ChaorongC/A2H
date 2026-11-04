"""CpG x sample tables behind the PGP-UK participant and R2-2 summaries: a package separate from package_v4.

Parts, each its own folder and zip under cpg_level_20261018/:
  A2H_cpg_level_20261018                            open data: PGP-UK participant CpG table (copy of the v4 file),
                                                    E1 per-CpG RMSE by subset, E3-C per CpG x participant
  A2H_addon_E1_predictions_20261018                 open data, large: every E1 subset's prediction per CpG x participant
  A2H_addon_R2-2_per_sample_caller_tables_20261018  large; atlas files are EGA controlled-access (user decision
                                                    2026-10-18: send under the user's DAA). Per-sample beta/CHALM/depth
Each part re-derives the published summary values it underlies (CHECKS.tsv) before it is zipped.
Run from R/: python collaborator_figure_data_20261017/build_cpg_level_package.py {main|addon-e1|gse240482|addon-callers}
(addon-callers copies and indexes; then verify-callers-part as a 20-task array, then finish-callers).
After building, only the zip FILES were renamed (2026-10-19, user request) to name response, comment and dataset:
  A2H_cpg_level_20261018.zip -> A2H_CpG-level_PGPUK_Resp2.3-R2-2_Resp3.11-3.12-R3-4-R3-5_20261018.zip
  A2H_cpg_level_GSE240482_20261018.zip -> A2H_CpG-level_GSE240482-prostate-tumour_Resp2.5-R2-4_20261018.zip
  A2H_addon_E1_predictions_20261018.zip -> A2H_addon_PGPUK_Resp2.3-R2-2_E1-reference-size-predictions_20261018.zip
  A2H_addon_R2-2_per_sample_caller_tables_20261018.zip -> A2H_addon_Atlas-EGA_PGPUK_Resp2.3-R2-2_E2-E3-depth-read-length-per-sample_20261018.zip
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
import src.r2_2_e2e3_external as e2e3  # noqa: E402
from src.r2_2_common import site_metrics  # noqa: E402
from src.r2_2_e2e3_summarize import paired_metrics  # noqa: E402

RES = R / 'results'
HERE = Path(__file__).resolve().parent
ROOT = HERE / 'cpg_level_20261018'
V4 = HERE / 'package_v4/A2H_figure_data_20261017'
E1 = RES / 'r2_2_sensitivity_20261011/E1'
PAIRS = RES / 'pgpuk_depth10_calibrated/pairs.tsv'
EPIC_DIR = Path('work/r2_2_e2e3_20261015/summary_20261016/epic')
SUMMARY = RES / 'r2_2_e2e3_20261015/summary_20261016'
TARGETS = ('CHALM', 'CAMDA')
CONDITIONS = ('native', 'raw125', 'raw100', 'raw75')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check(rows, part, item, expected, observed, tol=1e-12):
    ok = bool(np.isclose(expected, observed, rtol=0, atol=tol)) if isinstance(expected, float) else expected == observed
    rows.append({'part': part, 'item': item, 'published': expected, 'from_package_file': observed, 'pass': ok})


def finish(out: Path, checks: list) -> None:
    table = pd.DataFrame(checks)
    table.to_csv(out / 'CHECKS.tsv', sep='\t', index=False)
    rows = [{'file': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'MANIFEST.tsv']
    pd.DataFrame(rows).to_csv(out / 'MANIFEST.tsv', sep='\t', index=False)
    print(f'{out.name}: {len(table)} checks, {int((~table["pass"]).sum())} failed', flush=True)
    if not table['pass'].all():
        raise SystemExit(f'Checks failed; see {out / "CHECKS.tsv"}')


def ext_table(target: str) -> tuple[pd.DataFrame, list[str]]:
    return e2e3.external_table(E1, target)


def runs() -> list[tuple[Path, int, int]]:
    out = []
    for run_dir in sorted((E1 / 'runs').glob('N*_rep*')):
        out.append((run_dir, int(run_dir.name[1:].split('_')[0]), int(run_dir.name.split('rep')[1])))
    if len(out) != 60:
        raise ValueError(f'Expected 60 E1 runs, found {len(out)}')
    return out


def published_e1() -> pd.DataFrame:
    return pd.read_csv(E1 / 'summary/E1_reference_size_metrics.tsv', sep='\t')


# ------------------------------------------------------------------------------------------- main part
def build_main() -> None:
    out = ROOT / 'A2H_cpg_level_20261018'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    checks = []
    # 1. PGP-UK participants (Resp 3.11/3.12 and the |dbeta| <= 0.05 per-sample table): identical copy of the v4 table.
    d = out / 'PGPUK_participants'
    d.mkdir(parents=True)
    src = V4 / 'Resp3.11_3.12_R3-4_R3-5_PGPUK_participants/per_participant_cpg_observations.tsv.gz'
    dest = d / src.name
    shutil.copy2(src, dest)
    v4_manifest = pd.read_csv(V4 / 'MANIFEST.tsv', sep='\t').set_index('file')
    check(checks, 'participants', 'copy identical to package_v4 file (sha256)',
          v4_manifest.at['Resp3.11_3.12_R3-4_R3-5_PGPUK_participants/per_participant_cpg_observations.tsv.gz', 'sha256'],
          sha(dest))
    obs = pd.read_csv(dest, sep='\t')
    metrics = pd.read_csv(V4 / 'Resp3.11_3.12_R3-4_R3-5_PGPUK_participants/participant_metrics.tsv', sep='\t')
    filtered = pd.read_csv(V4 / 'Secondary_dbeta_filter_PGPUK/PGPUK_per_sample.tsv', sep='\t')
    for target in TARGETS:
        for version, sub in (('unfiltered', obs[obs.target == target]),
                             ('dbeta_le_0.05', obs[(obs.target == target) & obs['dbeta_le_0.05']])):
            rmse = np.sqrt(((sub.a2h_predicted - sub.wgbs_observed) ** 2).groupby(sub.sample_id).mean())
            ref = filtered[(filtered.target == target) & (filtered.version == version)].set_index('sample_id')
            check(checks, 'participants', f'{target} {version}: max |participant RMSE - PGPUK_per_sample.tsv|', 0.0,
                  float((rmse - ref.rmse).abs().max()), 1e-5)
            if version == 'unfiltered':
                ref = metrics[(metrics.target == target) & (metrics.method == 'A2H')].set_index('sample_id')
                check(checks, 'participants', f'{target}: max |participant RMSE - participant_metrics.tsv|', 0.0,
                      float((rmse - ref.rmse).abs().max()), 1e-5)
    # 2. E1: per-CpG RMSE for every subset (and the full 205-sample model on the same pairs).
    d = out / 'R2-2_E1_reference_size'
    d.mkdir()
    parts = []
    cols = ['probe_id', 'all_n', 'all_rmse', 'all_sse', 'common_n', 'common_rmse', 'common_sse',
            'baseline_common_rmse', 'baseline_common_sse']
    for target in TARGETS:
        ext, samples = ext_table(target)
        obs_m = ext[[f'observed__{s}' for s in samples]].to_numpy(dtype=float)
        base = ext[[f'baseline__{s}' for s in samples]].to_numpy(dtype=float)
        fixed = site_metrics(obs_m, base)
        parts.append(pd.DataFrame({'target': target, 'n_reference': 205, 'replicate': -1, 'probe_id': ext.probe_id,
                                   'all_n': fixed.n, 'all_rmse': fixed.rmse, 'all_sse': fixed.sse}))
        for run_dir, size, rep in runs():
            site = pd.read_parquet(run_dir / f'{target}_site_metrics.parquet', columns=cols)
            parts.append(site.assign(target=target, n_reference=size, replicate=rep))
    table = pd.concat(parts, ignore_index=True)
    order = ['target', 'n_reference', 'replicate'] + cols
    table = table[order]
    e1_path = d / 'E1_per_cpg_rmse_by_subset.parquet'
    table.to_parquet(e1_path, index=False, compression='zstd')
    back = pd.read_parquet(e1_path)
    pub = published_e1()
    for (target, size, rep), part in back[back.replicate >= 0].groupby(['target', 'n_reference', 'replicate']):
        for case, sse, n in (('all', 'all_sse', 'all_n'), ('common', 'common_sse', 'common_n'),
                             ('baseline_common', 'baseline_common_sse', 'common_n')):
            ref = pub[(pub.target == target) & (pub.n_reference == size) & (pub.replicate == rep) & (pub.case == case)].iloc[0]
            check(checks, 'E1', f'{target} N{size} rep{rep} {case} pooled RMSE', float(ref.pooled_rmse),
                  float(np.sqrt(part[sse].sum() / part[n].sum())))
    for target in TARGETS:
        part = back[(back.target == target) & (back.replicate == -1)]
        ref = pub[(pub.target == target) & (pub.case == 'baseline_fixed')].iloc[0]
        check(checks, 'E1', f'{target} full 205 pooled RMSE', float(ref.pooled_rmse),
              float(np.sqrt(part.all_sse.sum() / part.all_n.sum())))
    # 3. E3-C: PGP-UK truth recomputed from shorter reads, per CpG x participant x condition.
    d = out / 'R2-2_E3C_pgpuk_read_length'
    d.mkdir()
    pairs = pd.read_csv(PAIRS, sep='\t').set_index('array_id')
    parts = []
    for target in TARGETS:
        ext, samples = ext_table(target)
        wgbs_ids = pairs.loc[samples, 'wgbs_id'].tolist()
        locked = ext[[f'observed__{s}' for s in samples]].to_numpy(dtype=float)
        deployed = ext[[f'baseline__{s}' for s in samples]].to_numpy(dtype=float)
        beta = ext[[f'm__{s}' for s in samples]].to_numpy(dtype=float)
        rows = np.isfinite(locked)
        for condition in CONDITIONS:
            truth = e2e3.capped_truth(EPIC_DIR, condition, ext, wgbs_ids, target, 10)
            depth = np.column_stack([pd.read_parquet(EPIC_DIR / f'pgpuk_c__{w}__{condition}.parquet')
                                     .set_index('locus').reindex(ext.locus)['ct'].to_numpy(dtype=float) for w in wgbs_ids])
            i, j = np.nonzero(rows)
            parts.append(pd.DataFrame({'target': target, 'condition': condition, 'sample_id': np.array(samples)[j],
                                       'wgbs_id': np.array(wgbs_ids)[j], 'probe_id': ext.probe_id.to_numpy()[i],
                                       'locus': ext.locus.to_numpy()[i], 'array_beta': beta[i, j],
                                       'a2h_deployed': deployed[i, j], 'truth_native_locked': locked[i, j],
                                       'truth_this_condition': truth[i, j], 'depth_this_condition': depth[i, j]}))
    table = pd.concat(parts, ignore_index=True)
    e3_path = d / 'E3C_pgpuk_per_cpg_per_participant.parquet'
    table.to_parquet(e3_path, index=False, compression='zstd')
    back = pd.read_parquet(e3_path)
    pub = pd.read_csv(RES / 'r2_2_e2e3_pgpuk_c_20261016/pgpuk_c.tsv', sep='\t').set_index(['target', 'condition'])
    for target in TARGETS:
        native = back[(back.target == target) & (back.condition == 'native')]
        both = native.truth_this_condition.notna()
        check(checks, 'E3C', f'{target} native recomputed truth vs locked truth, max |diff|', 0.0,
              float((native.truth_this_condition - native.truth_native_locked)[both].abs().max()), 1e-12)
        for condition in CONDITIONS[1:]:
            sub = back[(back.target == target) & (back.condition == condition) & back.truth_this_condition.notna()
                       & back.a2h_deployed.notna()]
            a = np.sqrt(((sub.a2h_deployed - sub.truth_this_condition) ** 2).groupby(sub.probe_id).mean()).mean()
            b = np.sqrt(((sub.a2h_deployed - sub.truth_native_locked) ** 2).groupby(sub.probe_id).mean()).mean()
            ref = pub.loc[(target, condition)]
            check(checks, 'E3C', f'{target} {condition} n_pairs', int(ref.n_pairs), len(sub))
            check(checks, 'E3C', f'{target} {condition} mean per-CpG RMSE vs shortened-read truth', float(ref.mean_site_rmse_a), float(a))
            check(checks, 'E3C', f'{target} {condition} mean per-CpG RMSE vs native truth', float(ref.mean_site_rmse_b), float(b))
    shutil.copy2(HERE / 'README_cpg_level.md', out / 'README.md')
    finish(out, checks)


# ------------------------------------------------------------------------------------------- add-on: E1 predictions
def build_addon_e1() -> None:
    out = ROOT / 'A2H_addon_E1_predictions_20261018'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    checks = []
    pub = published_e1()
    for target in TARGETS:
        ext, samples = ext_table(target)
        long = {'probe_id': np.repeat(ext.probe_id.to_numpy(), len(samples)), 'sample_id': np.tile(samples, len(ext))}
        for name, prefix in (('array_beta', 'm__'), ('wgbs_observed', 'observed__'), ('a2h_full205', 'baseline__')):
            long[name] = ext[[f'{prefix}{s}' for s in samples]].to_numpy(dtype=float).ravel()
        for run_dir, size, rep in runs():
            pred = pd.read_parquet(run_dir / f'{target}_predictions.parquet').set_index('probe_id').reindex(ext.probe_id)
            long[f'pred_N{size}_rep{rep:02d}'] = pred[[f'pred__{s}' for s in samples]].to_numpy(dtype=float).ravel()
        frame = pd.DataFrame(long)
        pred_path = out / f'E1_{target}_predictions_by_subset.parquet'
        frame.to_parquet(pred_path, index=False, compression='zstd')
        back = pd.read_parquet(pred_path)
        y, base = back.wgbs_observed.to_numpy(), back.a2h_full205.to_numpy()
        for run_dir, size, rep in runs():
            p = back[f'pred_N{size}_rep{rep:02d}'].to_numpy()
            common = np.isfinite(y) & np.isfinite(p) & np.isfinite(base)
            for case, pred in (('common', p), ('baseline_common', base)):
                value = float(np.sqrt(np.mean((pred[common] - y[common]) ** 2)))
                ref = pub[(pub.target == target) & (pub.n_reference == size) & (pub.replicate == rep) & (pub.case == case)].iloc[0]
                check(checks, 'E1 predictions', f'{target} N{size} rep{rep} {case} pooled RMSE', float(ref.pooled_rmse), value)
    shutil.copy2(HERE / 'README_addon_E1.md', out / 'README.md')
    finish(out, checks)


# ------------------------------------------------------------------------------------------- add-on: caller tables
def build_addon_callers() -> None:
    out = ROOT / 'A2H_addon_R2-2_per_sample_caller_tables_20261018'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    atlas_dir, pgpuk_dir = out / 'atlas_EGA_controlled_access', out / 'pgpuk_open_data'
    atlas_dir.mkdir(parents=True)
    pgpuk_dir.mkdir()
    groups = pd.read_csv(RES / 'tissue_only_r1_5_r1_6_20261008/atlas_samples.tsv', sep='\t').set_index('sample_id').Group
    pairs = pd.read_csv(PAIRS, sep='\t').set_index('wgbs_id')
    pgp = pd.read_csv(V4 / 'Resp3.11_3.12_R3-4_R3-5_PGPUK_participants/pairing_key.tsv', sep='\t').set_index('sample_id').pgp_id
    index = []
    for path in sorted(EPIC_DIR.glob('*.parquet')):
        experiment, sample, condition = path.stem.split('__')
        atlas = experiment != 'pgpuk_c'
        dest = (atlas_dir if atlas else pgpuk_dir) / path.name
        shutil.copyfile(path.resolve(), dest)
        index.append({'file': str(dest.relative_to(out)), 'data_class': 'EGA atlas, controlled access' if atlas else 'PGP-UK, open',
                      'experiment': experiment, 'sample_id': sample, 'condition': condition,
                      'atlas_cell_type_group': groups.get(sample, '') if atlas else '',
                      'pgpuk_array_sample_id': '' if atlas else pairs.at[sample, 'array_id'],
                      'pgp_id': '' if atlas else pgp.get(pairs.at[sample, 'array_id'], ''),
                      'n_loci': len(pd.read_parquet(dest, columns=['locus'])), 'sha256': sha(dest)})
    pd.DataFrame(index).to_csv(out / 'index.tsv', sep='\t', index=False)
    # Verification is parallel: verify-callers-part (array) then finish-callers.


# ------------------------------------------------------------------------------------------- GSE240482 tumours
G240 = RES / 'gse240482_epicv1_epic_controls_20261017'
METHODS = {'A2H': 'a2h_predicted', 'A2H_wgbs': 'a2h_from_wgbs_beta', 'm': 'array_beta_as_prediction',
           'm2': 'beta_squared', 'm3': 'beta_cubed', 'reference_mean': 'reference_mean',
           'reference_linear': 'reference_linear', 'universal_quadratic': 'universal_quadratic'}


def build_gse240482() -> None:
    """Six prostate tumours (GEO GSE240482): CpG x tumour predictions with the |dbeta| <= 0.05 flag, all EPIC loci."""
    out = ROOT / 'A2H_cpg_level_GSE240482_20261018'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    pairs = pd.read_csv(G240 / 'primary_pairs.tsv', sep='\t')
    key = pairs[['sample_id', 'array_id', 'wgbs_id', 'tissue_label']].rename(columns={'sample_id': 'tumour_id'})
    key.assign(array_series='GSE240412', wgbs_series='GSE240481', superseries='GSE240482',
               publication='Peters et al., BMC Genomics 2024;25:251 (PMID 38448820)').to_csv(
        out / 'pairing_key.tsv', sep='\t', index=False)
    parts = []
    for target in TARGETS:
        obs = pd.read_parquet(G240 / f'{target}_observations.parquet')
        wgbs_pred = pd.read_csv(G240 / f'a2h_wgbs_input/gse240482_EPICv1_wgbs.{target.lower()}.tsv', sep='\t', index_col=0)
        wgbs_pred = wgbs_pred.stack(future_stack=True)
        frame = pd.DataFrame({
            'target': target, 'tumour_id': obs.biological_sample_id, 'array_id': obs.sample_id,
            'probe_id': obs.probe_id, 'locus': obs.locus, 'array_beta': obs.m, 'wgbs_beta': obs.WGBS_beta,
            'dbeta_le_0.05': (obs.m - obs.WGBS_beta).abs() <= 0.05,  # full precision, as in the filter analyses
            'wgbs_observed': obs.observed, 'a2h_predicted': obs.A2H,
            'a2h_from_wgbs_beta': wgbs_pred.reindex(pd.MultiIndex.from_frame(obs[['probe_id', 'sample_id']])).to_numpy(),
            'beta_squared': obs.m2, 'beta_cubed': obs.m3, 'reference_mean': obs.reference_mean,
            'reference_linear': obs.reference_linear, 'universal_quadratic': obs.universal_quadratic})
        if frame.a2h_from_wgbs_beta.isna().any():
            raise ValueError('Missing A2H-from-WGBS-beta prediction')
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    path = out / 'per_tumour_cpg_observations.tsv.gz'
    table.to_csv(path, sep='\t', index=False, float_format='%.6g')
    back = pd.read_csv(path, sep='\t')
    checks, rows = [], []
    back['array_beta_as_prediction'] = back.array_beta
    for target in TARGETS:
        for version in ('unfiltered', 'dbeta_le_0.05'):
            sub = back[back.target == target]
            if version != 'unfiltered':
                sub = sub[sub['dbeta_le_0.05']]
            scopes = [('pooled', 'all', sub)] + [('tumour', t, part) for t, part in sub.groupby('tumour_id')]
            for method, column in METHODS.items():
                for scope, sample, part in scopes:
                    err = part[column] - part.wgbs_observed
                    rmse = float(np.sqrt((err ** 2).mean()))
                    rng = float(part.wgbs_observed.max() - part.wgbs_observed.min())
                    row = {'target': target, 'version': version, 'method': method, 'scope': scope, 'tumour_id': sample,
                           'n_pairs': len(part), 'n_cpgs': part.probe_id.nunique(),
                           'pearson_r': float(np.corrcoef(part.wgbs_observed, part[column])[0, 1]), 'rmse': rmse,
                           'nrmse': rmse / rng if rng > 0 else np.nan, 'mean_signed_error': float(err.mean())}
                    if scope == 'pooled':
                        row['mean_per_cpg_rmse'] = float(np.sqrt((err ** 2).groupby(part.probe_id).mean()).mean())
                    rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'per_tumour_metrics.tsv', sep='\t', index=False)
    published = pd.read_csv(G240 / 'metrics.tsv', sep='\t')
    published = published[(published.depth == 10) & (published.scope == 'sample')].set_index(['target', 'method', 'sample_id'])
    filt = pd.read_csv(RES / 'external_delta_beta_filter_20261017/per_sample.tsv', sep='\t')
    filt = filt[filt.set == 'GSE240482_EPICv1'].merge(pairs[['array_id', 'sample_id']].rename(
        columns={'sample_id': 'tumour_id', 'array_id': 'sample_id'}), on='sample_id').set_index(['target', 'version', 'tumour_id'])
    for row in metrics[(metrics.scope == 'tumour')].itertuples(index=False):
        if row.version == 'unfiltered' and row.method in ('A2H', 'm', 'm2', 'm3', 'reference_mean', 'reference_linear',
                                                          'universal_quadratic'):
            ref = published.loc[(row.target, row.method, row.tumour_id)]
            check(checks, 'GSE240482', f'{row.target} {row.method} {row.tumour_id} n', int(ref.n), int(row.n_pairs))
            check(checks, 'GSE240482', f'{row.target} {row.method} {row.tumour_id} RMSE', float(ref.rmse), row.rmse, 1e-5)
            check(checks, 'GSE240482', f'{row.target} {row.method} {row.tumour_id} r', float(ref.pearson_r), row.pearson_r, 1e-5)
        if row.method == 'A2H':
            ref = filt.loc[(row.target, row.version, row.tumour_id)]
            check(checks, 'GSE240482', f'{row.target} {row.version} A2H {row.tumour_id} n vs filter analysis',
                  int(ref.n), int(row.n_pairs))
            check(checks, 'GSE240482', f'{row.target} {row.version} A2H {row.tumour_id} RMSE vs filter analysis',
                  float(ref.rmse), row.rmse, 1e-5)
    shutil.copy2(HERE / 'README_GSE240482.md', out / 'README.md')
    finish(out, checks)


CALLERS = ROOT / 'A2H_addon_R2-2_per_sample_caller_tables_20261018'
VERIFY = ROOT / '_verify_callers'  # scratch for the parallel verification; not part of any package


def verify_callers_part() -> None:
    """Array task: recompute a 1/N share of the published per-sample comparisons from the copied files."""
    task, ntasks = int(sys.argv[2]), int(sys.argv[3])
    index = pd.read_csv(CALLERS / 'index.tsv', sep='\t')
    published = pd.read_csv(SUMMARY / 'paired_epic_deltas.tsv', sep='\t').set_index(['experiment', 'sample_id', 'condition', 'target'])
    work = index[index.condition != 'native'].sort_values('file').reset_index(drop=True)
    work = work.iloc[task::ntasks]
    load = lambda f: pd.read_parquet(CALLERS / f).set_index('locus')
    checks = []
    for (experiment, sample), part in work.groupby(['experiment', 'sample_id']):
        native = load(index.loc[(index.experiment == experiment) & (index.sample_id == sample) & (index.condition == 'native'),
                                'file'].iloc[0])
        for row in part.itertuples(index=False):
            for metric in paired_metrics(native, load(row.file), 824209):
                ref = published.loc[(experiment, sample, row.condition, metric['target'])]
                label = f'{experiment} {sample} {row.condition} {metric["target"]}'
                check(checks, 'callers', f'{label} n_matched', int(ref.n_matched), int(metric['n_matched']))
                check(checks, 'callers', f'{label} mean_signed_delta', float(ref.mean_signed_delta), float(metric['mean_signed_delta']))
                check(checks, 'callers', f'{label} rmse', float(ref.rmse), float(metric['rmse']))
    VERIFY.mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(VERIFY / f'part_{task:02d}.tsv', sep='\t', index=False)


def finish_callers() -> None:
    index = pd.read_csv(CALLERS / 'index.tsv', sep='\t')
    checks = []
    check(checks, 'callers', 'files copied', 586, len(index))
    check(checks, 'callers', 'atlas samples', 39, index.loc[index.experiment != 'pgpuk_c', 'sample_id'].nunique())
    check(checks, 'callers', 'copied file sha256 matches index', len(index),
          int(sum(sha(CALLERS / f) == h for f, h in zip(index.file, index.sha256))))
    parts = pd.concat([pd.read_csv(p, sep='\t') for p in sorted(VERIFY.glob('part_*.tsv'))], ignore_index=True)
    n_expected = 3 * 3 * int((index.condition != 'native').sum())  # 3 targets x 3 statistics per comparison
    check(checks, 'callers', 'published per-sample statistics recomputed', n_expected, len(parts))
    shutil.copy2(HERE / 'README_addon_callers.md', CALLERS / 'README.md')
    finish(CALLERS, checks + parts.to_dict('records'))


if __name__ == '__main__':
    {'main': build_main, 'addon-e1': build_addon_e1, 'addon-callers': build_addon_callers,
     'gse240482': build_gse240482, 'verify-callers-part': verify_callers_part,
     'finish-callers': finish_callers}[sys.argv[1]]()
