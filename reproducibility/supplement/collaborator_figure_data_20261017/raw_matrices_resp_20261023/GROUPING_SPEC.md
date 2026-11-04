# Resp 1.5 (R1-4): rule for the matched 10-CpG groups

Reviewer 1 asked for within-CpG performance: pool disjoint sets of 10 CpGs with nearly identical mean and variance, then show the distribution of correlation and RMSE across many such sets, alongside the same distributions for m² and m³.

This file gives the exact rule, so the groups can be rebuilt from this package or from new data. `code/make_groups.py` implements it; on the PGP-UK data it reproduces the old membership exactly (see `CHECKS.tsv`). The old membership table is `PGPUK/R1-4/old_published/groups.tsv`.

## Inputs

These are needed for one cohort with K individuals, for each CpG *i* and individual *j*:

| Symbol | Meaning | File in this package |
|---|---|---|
| βᵢⱼ | Array β: the A2H input, and the grouping variable | `base/array_beta.tsv.gz` |
| yᵢⱼᵀ | Observed WGBS truth for target T (CHALM; CAMDA = CHALM − WGBS β) | `base/wgbs_CHALM.tsv.gz`, `base/wgbs_CAMDA.tsv.gz` |
| ŷᵢⱼᵀ | A2H prediction from βᵢⱼ | `base/A2H_CHALM.tsv.gz`, `base/A2H_CAMDA.tsv.gz` |
| Mᵢⱼᵀ | Eligibility mask (0/1) of your choice | `masks/<name>_<T>.tsv.gz` |

## Steps

1. **Valid cell.** Cell (i, j) is valid for target T when Mᵢⱼᵀ = 1 and βᵢⱼ, yᵢⱼᵀ and ŷᵢⱼᵀ are all present.
2. **Eligible CpG.** CpG *i* is eligible when it is valid in **all K individuals for both targets**, CHALM and CAMDA. One membership is shared by both targets and all three methods.
3. **Grouping statistics,** from the array β only (not the truth):
   - mean_m = (1/K) Σⱼ βᵢⱼ
   - var_m = (1/K) Σⱼ (βᵢⱼ − mean_m)², the population variance (ddof = 0)
4. **Blocks of 100.**
   - Sort eligible CpGs by mean_m ascending. Break ties by probe ID in ascending string order.
   - Number them mean_rank = 1…N.
   - block_id = ⌊(mean_rank − 1) / 100⌋.
5. **Groups of 10 within each block.**
   - Within each block, sort by var_m, then mean_m, then probe ID, all ascending.
   - With within-block position w = 0, 1, …:
     - group_id = 10 × block_id + ⌊w / 10⌋
     - member_rank = (w mod 10) + 1
6. **Remainder.** Keep only groups with exactly 10 CpGs. Only the last, partial block can leave CpGs over (at most 9); they are dropped.
7. **Scoring,** per group, target T and method:
   - Pool the 10 CpGs × K individuals, giving 10·K pairs of (yᵀ, prediction). The methods are:
     - **A2H:** ŷᵀ.
     - **m2:** βᵢⱼ², the per-individual power of the array β, not fitted.
     - **m3:** βᵢⱼ³, likewise.
   - Pearson r over the 10·K pairs. It is undefined (NA) when truth or prediction is constant; that group's RMSE is still kept.
   - RMSE = √( mean over the 10·K pairs of (prediction − y)² ).
8. **Reporting.** Report the distribution across groups (median and quantiles) for each target and method.

The rule matches groups on the array β, not on the truth. It is rank-based, with no absolute tolerance. No group or CpG is selected on performance.

## What the old (published) membership used

- **Mask: the primary external rule, PGP-UK.** This is `masks/mask_primary_external_<T>`:
  - truth depth ≥ 10 reads; for CAMDA the WGBS β depth must also be ≥ 10 (both depths are identical here);
  - A2H input support ≥ 0.05;
  - β, truth and A2H all present;
  - the CpG has ≥ 8 of the 10 participants passing.

  No |Δβ| filter was applied, nothing was refitted, and no extra support filter was added.
- **Probe universe.** `cg` probes with coordinates in the EPIC manifest. This included **8 probes** that were later dropped from the audited one-to-one EPIC set used for every matrix here (coordinate conflicts between the EPIC manifest and A2H's probe map).
  - Those 8 are not rows in the matrices.
  - Their published values (β, β², β³, depth-≥10 truth, A2H, for 10 participants and both targets) are in `PGPUK/R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv`.
- **Flow** (`old_published/flow.tsv`):
  - 143,503 evaluation CpGs
  - → 134,081 EPIC-mapped
  - → 40,700 complete in all 10 participants for both targets
  - → 407 blocks
  - → 4,070 groups, 0 remainder.
- **Achieved similarity** (`old_published/group_quality_summary.tsv`):
  - within-group range of mean β: median 0.00134, p95 0.00490;
  - within-group range of β variance: median 7.2 × 10⁻⁵, p95 0.00459.
- **Published medians** (`old_published/summary.tsv`):

  | Target | Method | Median r | Median RMSE |
  |---|---|---|---|
  | CHALM | A2H | 0.538 | 0.0881 |
  | CHALM | m² | 0.083 | 0.236 |
  | CHALM | m³ | 0.083 | 0.323 |
  | CAMDA | A2H | 0.527 | 0.0760 |
  | CAMDA | m² | −0.023 | 0.535 |
  | CAMDA | m³ | −0.024 | 0.407 |

## Reproducing the old membership

```
python code/make_groups.py --cohort-dir PGPUK --mask mask_primary_external \
    --extra PGPUK/R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv --out rebuilt
```

- This gives the same `probe_id`, `mean_rank`, `block_id`, `group_id` and `member_rank` for all 40,700 CpGs.
- mean_m and var_m agree to ≤ 1e-15; the summation order over individuals differs in the last bit.
- Group r and RMSE agree to ≤ 1e-12.

Without `--extra` (audited rows only), the result is 40,692 eligible CpGs, 407 blocks (the last holds 92 CpGs), 4,069 groups and 2 remainder CpGs. Membership then differs from the old table from the first block that held one of the 8 probes onward, because every later rank shifts.

## Applying the rule to new data or other filters

- Lay out the matrices the same way: `probe_id`, `locus`, one column per individual, `NA` for missing. Build a 0/1 mask from your own filters; `--mask none` uses every present cell. Then run `make_groups.py`.
- The mask fully determines the eligible set. A different filter changes the membership:
  - adding |Δβ| ≤ 0.05;
  - dropping the ≥ 8-participant rule;
  - using depth ≥ 4.

  The rule itself stays deterministic.
- With K ≠ 10 individuals, a group still holds 10 CpGs and pools 10·K pairs. GSE240482 has K = 6. No GSE240482 grouping was part of the response.
- Completeness in all K individuals is part of the rule; without it, groups would pool unequal numbers of pairs. Block size 100 and group size 10 are the published values. `--block-size` and `--group-size` exist for sensitivity runs only.
- **Caveat** (from the original analysis): pooled group correlation is not proof that A2H orders individuals correctly within a CpG. The complete-data requirement also selects well-covered CpGs.
