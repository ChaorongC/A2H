"""Collect the source data behind the eight A2H response figures into one package for collaborators.

Reads finished result branches only (read-only), writes one folder per team-response number, checks every
exported table against the numbers quoted in the team response, and writes MANIFEST.tsv and CHECKS.tsv.
No analysis is rerun. Run from R/:
python collaborator_figure_data_20261017/build_package.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
RES = R / 'results'
S = RES / 'r2_2_sensitivity_20261011'
NAME = 'A2H_figure_data_20261017'
# package/ = first pass (E1 n_pairs printed in float format); package_v2 = data only;
# package_v3 adds WGBS beta for PGP-UK, the |array beta - WGBS beta| <= 0.05 PGP-UK tables, figures and code.
# package_v3 failed its checks: the Delta-beta filter recomputed from 6-digit values flipped one boundary pair.
OUT = Path(__file__).resolve().parent / 'package_v4' / NAME
V2_D10 = Path('external_validation/results/paired_validation_v2_depth10/wgbs/probe_space')
DBETA = RES / 'external_delta_beta_filter_20261017'
TARGETS = ('CHALM', 'CAMDA')
CHECKS, SOURCES = [], {}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def src(path: Path) -> Path:
    SOURCES[str(path)] = sha(path)
    return path


def check(resp: str, item: str, reported: float, reproduced: float, tol: float) -> None:
    ok = bool(np.isfinite(reproduced) and abs(reported - reproduced) <= tol)
    CHECKS.append({'response': resp, 'item': item, 'reported': reported, 'reproduced_from_package': reproduced,
                   'tolerance': tol, 'pass': ok})


def write(frame: pd.DataFrame, folder: str, name: str) -> Path:
    path = OUT / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, float_format='%.6g')
    return path


def read_back(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')


def copy(source: Path, folder: str, name: str | None = None) -> Path:
    path = OUT / folder / (name or source.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src(source), path)
    return path


def resp_1_2() -> None:
    """R1-1: per-CpG held-out atlas CV error, A2H vs universal quadratic, averaged over the 10 CV repeats."""
    folder = 'Resp1.2_R1-1_CpG_specific_vs_universal'
    base = RES / 'r1_1_locus_specific_calibration'
    epic = pd.read_csv(src(base / 'coefficient_table.tsv.gz'), sep='\t', usecols=['locus', 'is_epic'])
    epic = set(epic.loc[epic.is_epic, 'locus'])
    ranges = pd.read_csv(src(base / 'atlas_truth_ranges.tsv.gz'), sep='\t')
    summary_dir = RES / 'atlas_cv_full/original/historical_sample_summary'
    reported = {('CHALM', 'A2H'): 0.05231, ('CAMDA', 'A2H'): 0.05228,
                ('CHALM', 'universal_quadratic'): 0.09184, ('CAMDA', 'universal_quadratic'): 0.09184}
    parts = []
    for target in TARGETS:
        frames = {}
        for method in ('A2H', 'universal_quadratic'):
            paths = sorted(summary_dir.glob(f'{target}_repeat*_shard*_{method}.parquet'))
            if len(paths) != 520:
                raise ValueError(f'{target} {method}: expected 520 files, found {len(paths)}')
            frame = pd.concat([pd.read_parquet(p, columns=['locus', 'repeat', 'n', 'pearson_r', 'rmse']) for p in paths],
                              ignore_index=True)
            frame = frame[frame.locus.isin(epic)]
            # Published summary: per repeat, mean site RMSE over EPIC sites with n >= 2, then mean over repeats.
            per_repeat = frame[frame.n >= 2].groupby('repeat').rmse.mean()
            check('1.2', f'{target} {method} mean CpG RMSE (mean of 10 repeat means, as reported)',
                  reported[(target, method)], float(per_repeat.mean()), 5e-6)
            frames[method] = frame
        pair = frames['A2H'].merge(frames['universal_quadratic'], on=['locus', 'repeat'], suffixes=('_A2H', '_universal'),
                                   validate='one_to_one')
        pair = pair[(pair.n_A2H >= 2) & np.isfinite(pair.rmse_A2H) & np.isfinite(pair.rmse_universal)]
        rng = ranges[ranges.target == target].set_index('locus').truth_range
        pair['range'] = pair.locus.map(rng)
        for m in ('A2H', 'universal'):
            pair[f'nrmse_{m}'] = np.where(pair.range > 0, pair[f'rmse_{m}'] / pair.range, np.nan)
        agg = pair.groupby('locus').agg(
            n_repeats=('repeat', 'size'), mean_heldout_n=('n_A2H', 'mean'), atlas_truth_range=('range', 'first'),
            A2H_rmse=('rmse_A2H', 'mean'), universal_rmse=('rmse_universal', 'mean'),
            A2H_nrmse=('nrmse_A2H', 'mean'), universal_nrmse=('nrmse_universal', 'mean'),
            A2H_pearson_r=('pearson_r_A2H', 'mean'), universal_pearson_r=('pearson_r_universal', 'mean')).reset_index()
        agg.insert(0, 'target', target)
        parts.append(agg)
    path = write(pd.concat(parts, ignore_index=True), folder, 'per_cpg_cv_A2H_vs_universal.tsv.gz')
    back = read_back(path)
    for target in TARGETS:
        sub = back[back.target == target]
        check('1.2', f'{target} CpGs in per-CpG file (EPIC, paired, n>=2 in >=1 repeat)', 824526, len(sub), 1000)
        for m, key in (('A2H', 'A2H'), ('universal', 'universal_quadratic')):
            check('1.2', f'{target} {m} mean of per-CpG repeat-averaged RMSE (file)', reported[(target, key)],
                  float(sub[f'{m}_rmse'].mean()), 2e-4)
    across = pd.read_csv(src(base / 'cv_metrics_across_repeats.tsv'), sep='\t')
    write(across[across.universe == 'EPIC_BeadChip'], folder, 'summary_across_repeats.tsv')
    copy(base / 'cv_matched_A2H_universal_by_repeat.tsv', folder, 'paired_differences_by_repeat.tsv')


def resp_1_5() -> None:
    folder = 'Resp1.5_R1-4_matched_CpG_groups'
    base = RES / 'pgpuk_depth10_beta_block100_groups10_complete10'
    path = copy(base / 'group_metrics.tsv', folder)
    for name in ('groups.tsv', 'group_quality.tsv', 'summary.tsv', 'flow.tsv'):
        copy(base / name, folder)
    g = read_back(path)
    check('1.5', 'groups per target/method', 4070, float(g.groupby(['target', 'method']).size().min()), 0)
    reported = {('CHALM', 'A2H'): (0.5384, 0.08811), ('CHALM', 'm2'): (0.0833, 0.23589), ('CHALM', 'm3'): (0.0826, 0.32262),
                ('CAMDA', 'A2H'): (0.5267, 0.07596), ('CAMDA', 'm2'): (-0.0233, 0.53535), ('CAMDA', 'm3'): (-0.0241, 0.40671)}
    for (target, method), (r, rmse) in reported.items():
        sub = g[(g.target == target) & (g.method == method)]
        check('1.5', f'{target} {method} median group r', r, float(sub.pearson_r.median()), 5e-5)
        check('1.5', f'{target} {method} median group RMSE', rmse, float(sub.rmse.median()), 5e-6)


def resp_1_6() -> None:
    folder = 'Resp1.6_R1-5_reference_composition'
    base = RES / 'tissue_only_r1_5_r1_6_20261008/summary'
    parts = []
    for target in TARGETS:
        d = pd.read_parquet(src(base / f'R1-5_{target}_site_metrics.parquet'))
        wide = d.pivot_table(index='probe_id', columns=['model', 'method'], values='rmse')
        wide.columns = [f'{method}_{model}' for model, method in wide.columns]
        wide['n_participants'] = d[(d.model == 'full_reference') & (d.method == 'A2H')].set_index('probe_id').n
        wide = wide.reset_index()
        wide.insert(0, 'target', target)
        parts.append(wide)
    out = pd.concat(parts, ignore_index=True)
    order = ['target', 'probe_id', 'n_participants'] + [f'{m}_{r}' for m in ('A2H', 'reference_linear', 'm')
                                                         for r in ('full_reference', 'exclude_blood_immune', 'exclude_epithelial')]
    path = write(out[order], folder, 'per_cpg_rmse_by_reference.tsv.gz')
    copy(base / 'R1-5_summary.tsv', folder, 'summary_with_participant_bootstrap.tsv')
    back = read_back(path)
    reported = {'CHALM': (0.11445, 0.11509, 0.11500), 'CAMDA': (0.08493, 0.08856, 0.08502)}
    for target, values in reported.items():
        sub = back[back.target == target]
        check('1.6', f'{target} CpGs', 134051, len(sub), 0)
        for model, value in zip(('full_reference', 'exclude_blood_immune', 'exclude_epithelial'), values):
            check('1.6', f'{target} A2H {model} mean per-CpG RMSE', value, float(sub[f'A2H_{model}'].mean()), 5e-6)


def resp_1_7() -> None:
    folder = 'Resp1.7_R1-6_celltype_holdouts'
    base = RES / 'tissue_only_r1_5_r1_6_20261008'
    d = pd.read_parquet(src(base / 'summary/R1-6_site_metrics.parquet'))
    d = d[d.eligible & d.method.isin(['A2H', 'reference_linear'])]
    wide = d.pivot_table(index=['model', 'target', 'locus'], columns='method', values='rmse').reset_index()
    n = d[d.method == 'A2H'].set_index(['model', 'target', 'locus']).n
    wide['n_heldout_obs'] = n.reindex(pd.MultiIndex.from_frame(wide[['model', 'target', 'locus']])).to_numpy()
    wide = wide.rename(columns={'model': 'heldout_arm', 'A2H': 'A2H_rmse', 'reference_linear': 'linear_rmse'})
    path = write(wide[['heldout_arm', 'target', 'locus', 'n_heldout_obs', 'A2H_rmse', 'linear_rmse']], folder,
                 'per_cpg_heldout_A2H_vs_linear.tsv.gz')
    copy(base / 'summary/R1-6_summary.tsv', folder, 'heldout_summary.tsv')
    copy(base / 'r1_6_with_whole_atlas_cv/R1-6_combined_summary.tsv', folder, 'heldout_and_whole_atlas_cv_summary.tsv')
    back = read_back(path)
    reported = {('exclude_blood_immune', 'CHALM'): (0.05466, 0.06010), ('exclude_blood_immune', 'CAMDA'): (0.05465, 0.06009),
                ('exclude_epithelial', 'CHALM'): (0.05743, 0.06188), ('exclude_epithelial', 'CAMDA'): (0.05743, 0.06188)}
    for (arm, target), (a, lin) in reported.items():
        sub = back[(back.heldout_arm == arm) & (back.target == target)]
        check('1.7', f'{arm} {target} A2H mean per-CpG RMSE', a, float(sub.A2H_rmse.mean()), 5e-6)
        check('1.7', f'{arm} {target} linear mean per-CpG RMSE', lin, float(sub.linear_rmse.mean()), 5e-6)


def resp_2_3() -> None:
    folder = 'Resp2.3_R2-2_reference_dependence'
    e = pd.read_csv(src(S / 'E1/summary/E1_reference_size_metrics.tsv'), sep='\t')
    p = e[e.case.isin(['common', 'baseline_common'])].pivot_table(
        index=['target', 'n_reference', 'replicate', 'seed'], columns='case', values=['pooled_rmse', 'n_valid_pairs'])
    p.columns = [f'{a}_{b}' for a, b in p.columns]
    p = p.reset_index().rename(columns={'pooled_rmse_common': 'pooled_rmse_subset', 'pooled_rmse_baseline_common': 'pooled_rmse_full205',
                                        'n_valid_pairs_common': 'n_pairs'}).drop(columns='n_valid_pairs_baseline_common')
    p['n_pairs'] = p.n_pairs.astype(int)
    p['rmse_increase_subset_minus_full'] = p.pooled_rmse_subset - p.pooled_rmse_full205
    path = write(p, folder, 'E1_reference_size_20_subsets.tsv')
    back = read_back(path)
    reported = {('CHALM', 75): 0.000802, ('CHALM', 100): 0.000766, ('CHALM', 150): 0.000295,
                ('CAMDA', 75): 0.001458, ('CAMDA', 100): 0.000972, ('CAMDA', 150): 0.000378}
    for (target, size), value in reported.items():
        sub = back[(back.target == target) & (back.n_reference == size)]
        check('2.3', f'{target} N={size} subsets', 20, len(sub), 0)
        check('2.3', f'{target} N={size} median RMSE increase', value, float(sub.rmse_increase_subset_minus_full.median()), 5e-7)
    e2e3 = RES / 'r2_2_e2e3_20261015/summary_20261016'
    copy(e2e3 / 'condition_summary.tsv', folder, 'E2_E3_condition_summary.tsv')
    copy(e2e3 / 'figure/R2_2_depth_readlength_source_data.tsv', folder, 'E2_E3_figure_source_data.tsv')
    copy(e2e3 / 'figure/R2_2_depth_readlength_per_sample.tsv', folder, 'E2_E3_per_sample.tsv')
    for ext in ('pdf', 'png', 'svg'):
        copy(e2e3 / f'figure/R2_2_depth_readlength.{ext}', folder + '/figure')
    copy(RES / 'r2_2_e2e3_pgpuk_c_20261016/pgpuk_c.tsv', folder, 'E3_PGPUK_readlength_validation.tsv')
    copy(RES / 'r2_2_e2e3_external_20261016/compare_e2.tsv', folder, 'refit39_depth_vs_native.tsv')
    copy(RES / 'r2_2_e2e3_external_20261016/compare_e3.tsv', folder, 'refit39_readlength_vs_native.tsv')
    cond = read_back(OUT / folder / 'E2_E3_condition_summary.tsv')
    shifts = cond[(cond.experiment == 'e3') & (cond.target == 'chalm')].set_index('level').mean_shift
    for level, value in (('raw125', -0.011), ('raw100', -0.025), ('raw75', -0.041)):
        check('2.3', f'E3 CHALM mean shift {level}', value, float(shifts[level]), 5e-4)


def resp_2_4() -> None:
    folder = 'Resp2.4_R2-3_site_level_error'
    e4 = pd.read_parquet(src(S / 'E4/summary/E4_per_site_oof_metrics.parquet'),
                         columns=['locus', 'target', 'n', 'rmse', 'mae', 'bias_pred_minus_obs', 'pearson_r', 'n_reference_joint'])
    e4 = e4[np.isfinite(e4.rmse)]
    path = write(e4, folder, 'per_cpg_atlas_cv_error.tsv.gz')
    copy(S / 'E4/summary/E4_by_reference_support.tsv', folder, 'error_by_reference_support.tsv')
    copy(S / 'E4/summary/E4_site_distribution.tsv', folder, 'error_distribution_summary.tsv')
    back = read_back(path)
    for target, q3 in (('CHALM', 0.07263), ('CAMDA', 0.07261)):
        v = back.loc[back.target == target, 'rmse']
        check('2.4', f'{target} CpGs with defined RMSE', 824020, len(v), 0)
        check('2.4', f'{target} median CpG RMSE', 0.04259, float(v.median()), 5e-6)
        check('2.4', f'{target} Q1', 0.02286, float(v.quantile(.25)), 5e-6)
        check('2.4', f'{target} Q3', q3, float(v.quantile(.75)), 5e-6)
    parts = []
    for target in TARGETS:
        for size in (75, 100, 150):
            d = pd.read_parquet(src(S / f'E1/summary/E1_site_fit_stability_{target}_N{size}.parquet'),
                                columns=['probe_id', 'locus', 'target', 'n_reference', 'n_external_inputs_with_2_refits',
                                         'median_prediction_sd', 'max_prediction_sd'])
            parts.append(d)
    path = write(pd.concat(parts, ignore_index=True), folder, 'per_cpg_prediction_sd_reference_subsets.tsv.gz')
    copy(S / 'E1/summary/E1_fit_stability_summary.tsv', folder, 'prediction_sd_summary.tsv')
    back = read_back(path)
    for size, value in ((75, 0.007542), (100, 0.005995), (150, 0.003543)):
        for target in TARGETS:
            v = back[(back.target == target) & (back.n_reference == size)].median_prediction_sd
            check('2.4', f'{target} N={size} median per-CpG prediction SD', value, float(v.median()), 5e-7)


def resp_2_7() -> None:
    folder = 'Resp2.7_R2-6_binomial_shape'
    path = copy(RES / 'r2_6_atlas_reference_20261012/full/binned_curves.tsv', folder, 'binned_curves_50bins.tsv')
    b = read_back(path)
    reported = {('CHALM', 12): 0.47011, ('CHALM', 25): 0.69592, ('CHALM', 37): 0.86695,
                ('CAMDA', 12): 0.22060, ('CAMDA', 25): 0.19124, ('CAMDA', 37): 0.11713}
    for (target, bin_), value in reported.items():
        row = b[(b.target == target) & (b.bin == bin_)].iloc[0]
        check('2.7', f'{target} bin {row.bin_start:.2f}-{row.bin_end:.2f} mean target (equal CpG weight)', value,
              float(row.mean_target_equal_cpg), 5e-6)
    check('2.7', 'bins per target', 50, float(b.groupby('target').size().min()), 0)


def resp_3_11() -> None:
    folder = 'Resp3.11_3.12_R3-4_R3-5_PGPUK_participants'
    base = RES / 'r3_epic_participant_validation_20261012/full'
    excluded = pd.read_csv(src(base / 'excluded_probes.tsv'), sep='\t')
    pairing = pd.read_csv(src(base / 'pairing_key.tsv'), sep='\t')
    parts = []
    for target in TARGETS:
        obs = pd.read_parquet(src(RES / f'pgpuk_depth10_calibrated/{target}_observations.parquet'),
                              columns=['probe_id', 'sample_id', 'observed', 'A2H', 'm'])
        drop = set(excluded.loc[excluded.target == target, 'probe_id'])
        obs = obs[~obs.probe_id.isin(drop)]
        obs = obs.rename(columns={'observed': 'wgbs_observed', 'A2H': 'a2h_predicted', 'm': 'array_beta'})
        obs.insert(0, 'target', target)
        parts.append(obs)
    out = pd.concat(parts, ignore_index=True).merge(pairing[['sample_id', 'pgp_id']], on='sample_id', how='left',
                                                    validate='many_to_one')
    wgbs = pd.read_csv(src(V2_D10 / 'pgpuk_450K.traditional_probe_matrix.tsv'), sep='\t', index_col=0)
    wgbs = wgbs.stack(future_stack=True).rename('wgbs_beta').rename_axis(['probe_id', 'sample_id'])
    out['wgbs_beta'] = wgbs.reindex(pd.MultiIndex.from_frame(out[['probe_id', 'sample_id']])).to_numpy()
    # Flag computed at full precision; values are written with 6 significant digits, which can flip boundary pairs.
    out['dbeta_le_0.05'] = (out.array_beta - out.wgbs_beta).abs() <= 0.05
    out = out[['target', 'sample_id', 'pgp_id', 'probe_id', 'array_beta', 'wgbs_beta', 'dbeta_le_0.05', 'a2h_predicted',
               'wgbs_observed']]
    path = write(out, folder, 'per_participant_cpg_observations.tsv.gz')
    copy(base / 'sample_metrics.tsv', folder, 'participant_metrics.tsv')
    copy(base / 'pairing_key.tsv', folder)
    copy(base / 'target_summary.tsv', folder)
    back = read_back(path)
    metrics = pd.read_csv(base / 'sample_metrics.tsv', sep='\t')
    for target, lo, hi in (('CHALM', 0.93328, 0.94807), ('CAMDA', 0.68590, 0.74561)):
        sub = back[back.target == target]
        check('3.11', f'{target} observations', 1214068, len(sub), 0)
        check('3.11', f'{target} CpGs', 134055, sub.probe_id.nunique(), 0)
        r = sub.groupby('sample_id').apply(lambda g: g.a2h_predicted.corr(g.wgbs_observed))
        check('3.11', f'{target} min participant r', lo, float(r.min()), 5e-5)
        check('3.11', f'{target} max participant r', hi, float(r.max()), 5e-5)
        ref = metrics[(metrics.target == target) & (metrics.method == 'A2H')].set_index('sample_id')
        e = sub.a2h_predicted - sub.wgbs_observed
        rmse = np.sqrt((e ** 2).groupby(sub.sample_id).mean())
        rng = sub.groupby('sample_id').wgbs_observed.agg(lambda v: v.max() - v.min())
        check('3.12', f'{target} max |participant RMSE file - table|', 0.0, float((rmse - ref.rmse).abs().max()), 1e-5)
        check('3.12', f'{target} max |participant NRMSE file - table|', 0.0, float((rmse / rng - ref.nrmse).abs().max()), 1e-5)
        check('3.11', f'{target} pairs without WGBS beta', 0, int(sub.wgbs_beta.isna().sum()), 0)
        # Secondary: the submitted manuscript's concordance filter, reproduced from this file.
        kept = sub[sub['dbeta_le_0.05']]
        e = kept.a2h_predicted - kept.wgbs_observed
        rmse = np.sqrt((e ** 2).groupby(kept.sample_id).mean())
        ref = pd.read_csv(DBETA / 'per_sample.tsv', sep='\t')
        ref = ref[(ref.set == 'PGP-UK_450K_EPIC') & (ref.target == target) & (ref.version == 'dbeta_le_0.05')].set_index('sample_id')
        check('3.11', f'{target} |dbeta|<=0.05 pairs (file vs filter branch)', float(ref.n.sum()), len(kept), 0)
        check('3.11', f'{target} |dbeta|<=0.05 max |participant RMSE file - filter branch|', 0.0,
              float((rmse - ref.rmse).abs().max()), 1e-5)


def dbeta_pgpuk() -> None:
    """PGP-UK rows of the |array beta - WGBS beta| <= 0.05 analysis, unfiltered and filtered side by side."""
    folder = 'Secondary_dbeta_filter_PGPUK'
    for name in ('methods.tsv', 'per_sample.tsv', 'composition.tsv', 'selection_diagnostics.tsv'):
        table = pd.read_csv(src(DBETA / name), sep='\t')
        table = table[table.set == 'PGP-UK_450K_EPIC']
        write(table, folder, f'PGPUK_{name}')
    m = read_back(OUT / folder / 'PGPUK_methods.tsv').set_index(['target', 'version', 'method'])
    for target, unf, filt in (('CHALM', 0.11445, 0.0598), ('CAMDA', 0.08493, 0.0560)):
        check('dbeta', f'{target} A2H mean per-CpG RMSE unfiltered', unf, float(m.at[(target, 'unfiltered', 'A2H'), 'mean_site_rmse']), 5e-6)
        check('dbeta', f'{target} A2H mean per-CpG RMSE |dbeta|<=0.05', filt,
              float(m.at[(target, 'dbeta_le_0.05', 'A2H'), 'mean_site_rmse']), 5e-5)


def code() -> None:
    folder = OUT / 'code'
    folder.mkdir()
    shutil.copy2(Path(__file__).with_name('make_figures.py'), folder / 'make_figures.py')
    shutil.copy2(Path(__file__), folder / 'build_package_provenance_only.py')


def manifest() -> None:
    rows = []
    for path in sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'MANIFEST.tsv'):
        rows.append({'file': str(path.relative_to(OUT)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    pd.DataFrame(rows).to_csv(OUT / 'MANIFEST.tsv', sep='\t', index=False)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest-only':  # after code/make_figures.py has drawn figures/
        manifest()
        return
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    shutil.copy2(Path(__file__).with_name('README_package.md'), OUT / 'README.md')
    for step in (resp_1_2, resp_1_5, resp_1_6, resp_1_7, resp_2_3, resp_2_4, resp_2_7, resp_3_11, dbeta_pgpuk, code):
        print('running', step.__name__, flush=True)
        step()
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(OUT / 'CHECKS.tsv', sep='\t', index=False)
    manifest()
    (OUT.parent / 'provenance.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'builder_sha256': sha(Path(__file__)),
        'sources_sha256': SOURCES, 'n_checks': len(checks), 'n_failed': int((~checks['pass']).sum())}, indent=2) + '\n')
    print(checks.to_string(), flush=True)
    if not checks['pass'].all():
        raise SystemExit('Some checks failed; see CHECKS.tsv')


if __name__ == '__main__':
    main()
