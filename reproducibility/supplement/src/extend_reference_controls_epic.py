"""Extend the locked reference controls (results/reference_controls_original) to EPIC-mapped loci.

The per-locus controls (reference mean and per-CpG linear fit) are fitted locus by locus with the
unchanged src/fit_reference_controls.fit_chunk on the same 205-sample atlas matrices, so adding
loci cannot change existing ones. All loci (locked HM450 set plus EPIC) are recomputed and the
locked rows must be reproduced exactly before anything is written. The universal quadratic is a
single curve; its locked coefficients are copied, not refitted. No external data are used.
Run from R/: python -m src.extend_reference_controls_epic --output-dir results/reference_controls_epic_<date>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_model import ATLAS, MODEL
from src.evaluate_pgpuk import file_hash
from src.fit_reference_controls import METH_FILE, TARGET_FILES, fit_chunk

LOCKED = Path(__file__).resolve().parents[1] / 'results/reference_controls_original'


def extend(output: Path, chunk_size: int) -> None:
    if output.exists():
        raise FileExistsError(f'Refusing to reuse {output}')
    locked_prov = json.loads((LOCKED / 'provenance.json').read_text())
    min_train = locked_prov['min_joint_training_observations']
    probe_map = Path(locked_prov['probe_map'])
    data_root = Path(locked_prov['atlas_root']) / 'data'
    paths = {'probe_map': probe_map, 'methylation': data_root / METH_FILE}
    paths.update({t: data_root / f for t, f in TARGET_FILES.items()})
    for name, path in paths.items():
        if file_hash(path) != locked_prov['file_sha256'][name]:
            raise ValueError(f'Input differs from the locked controls: {name}')
    table = pd.read_csv(probe_map, sep='\t', usecols=['loci.hg38', 'EPIC.hg38', 'HM450.hg38'])
    epic = table[['loci.hg38', 'EPIC.hg38']].dropna().drop_duplicates()
    if epic['EPIC.hg38'].duplicated().any():
        raise ValueError('One EPIC probe maps to multiple loci')
    methylation = pd.read_pickle(paths['methylation'])
    if methylation.index.has_duplicates or list(methylation.columns) != locked_prov['reference_samples']:
        raise ValueError('Reference methylation differs from the locked sample set')
    locked_loci = {t: pd.read_parquet(LOCKED / f'{t}_controls.parquet') for t in TARGET_FILES}
    hm450 = pd.Index(locked_loci['CHALM'].locus)
    epic_loci = pd.Index(epic['loci.hg38'].drop_duplicates())
    absent = epic_loci[~epic_loci.isin(methylation.index)]
    loci = hm450.append(epic_loci[epic_loci.isin(methylation.index) & ~epic_loci.isin(hm450)])
    loci.name = 'locus'
    output.mkdir(parents=True)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'locked_branch': str(LOCKED),
                  'locked_provenance_sha256': file_hash(LOCKED / 'provenance.json'),
                  'atlas_root': locked_prov['atlas_root'], 'probe_map': str(probe_map),
                  'reference_samples': locked_prov['reference_samples'], 'min_joint_training_observations': min_train,
                  'locked_hm450_loci': len(hm450), 'epic_mapped_loci': len(epic_loci),
                  'epic_loci_absent_from_atlas': len(absent), 'added_epic_only_loci': len(loci) - len(hm450),
                  'reference_loci': len(loci), 'file_sha256': {k: file_hash(p) for k, p in paths.items()},
                  'universal_scope': 'locked coefficients copied unchanged from reference_controls_original',
                  'code_sha256': {'extend': file_hash(Path(__file__)),
                                  'fit_reference_controls': file_hash(Path(__file__).with_name('fit_reference_controls.py'))}}
    for target in TARGET_FILES:
        response = pd.read_pickle(paths[target])
        if not methylation.index.equals(response.index) or not methylation.columns.equals(response.columns):
            raise ValueError(f'{target} reference matrix is not aligned with methylation')
        parts = []
        for start in range(0, len(loci), chunk_size):
            part = loci[start:start + chunk_size]
            frame, _, _ = fit_chunk(methylation.loc[part].to_numpy(dtype=float), response.loc[part].to_numpy(dtype=float),
                                    part, min_train)
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True)
        recomputed = frame.iloc[:len(hm450)].reset_index(drop=True)
        try:  # exact values, dtype-insensitive (pandas string vs object for the locus column)
            pd.testing.assert_frame_equal(recomputed, locked_loci[target].reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
        except AssertionError as error:
            raise ValueError(f'{target}: recomputed locked loci differ from reference_controls_original') from error
        frame.to_parquet(output / f'{target}_controls.parquet', index=False)
        added = frame.iloc[len(hm450):]
        provenance[target] = {
            'locked_rows_reproduced_exactly': True,
            'joint_eligible_loci': int(frame.reference_mean.notna().sum()),
            'added_eligible_loci': int(added.reference_mean.notna().sum()),
            'linear_constant_fallback_loci': int(frame.linear_constant_fallback.sum()),
            'universal_coefficients_intercept_linear_quadratic':
                locked_prov[target]['universal_coefficients_intercept_linear_quadratic'],
            'output_sha256': file_hash(output / f'{target}_controls.parquet')}
        del response
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: v for k, v in provenance.items() if k not in ('reference_samples', 'file_sha256')}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()
    extend(args.output_dir, args.chunk_size)


if __name__ == '__main__':
    main()
