"""Small, site-aware helpers for the isolated R2-2 sensitivity experiments."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.fit_atlas_fold import fit_sites


def nested_group_draws(samples: pd.DataFrame, seeds=range(42, 62)) -> pd.DataFrame:
    required = {"sample_id", "donor_id", "Group"}
    if not required.issubset(samples) or samples.sample_id.duplicated().any():
        raise ValueError("Reference sample manifest needs unique IDs and donor/group fields")
    if samples[sorted(required)].isna().any().any() or len(samples) != 205 or samples.Group.nunique() != 39:
        raise ValueError("Expected the verified 205 samples in 39 groups")
    rows = []
    for replicate, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        groups = {str(group): list(rng.permutation(part.sample_id.to_numpy()))
                  for group, part in samples.groupby("Group", sort=True)}
        group_names = list(rng.permutation(sorted(groups)))
        counts = {group: 0 for group in groups}
        sizes = {group: len(ids) for group, ids in groups.items()}
        order = []
        # Give each of the 39 groups one sample before proportional allocation.
        for group in group_names:
            order.append((group, groups[group].pop()))
            counts[group] += 1
        while len(order) < len(samples):
            rank = len(order) + 1
            available = [group for group in group_names if groups[group]]
            group = max(available, key=lambda key: (rank * sizes[key] / len(samples) - counts[key], -group_names.index(key)))
            order.append((group, groups[group].pop()))
            counts[group] += 1
        donor_by_id = samples.set_index("sample_id").donor_id
        for rank, (group, sample_id) in enumerate(order, start=1):
            rows.append((replicate, int(seed), rank, sample_id, group, donor_by_id.loc[sample_id]))
    draw = pd.DataFrame(rows, columns=["replicate", "seed", "rank", "sample_id", "Group", "donor_id"])
    for replicate, part in draw.groupby("replicate"):
        if len(part) != 205 or part.sample_id.nunique() != 205 or not part["rank"].eq(range(1, 206)).all():
            raise ValueError(f"Invalid nested sample draw {replicate}")
        for size in (75, 100, 150, 205):
            if part.loc[part["rank"] <= size, "Group"].nunique() != 39:
                raise ValueError("Nested draw omitted an atlas group")
    return draw


def fit_compatible(x: np.ndarray, y: np.ndarray, min_train=50) -> pd.DataFrame:
    """Vectorized quadratic fit; scalar polyfit only for ill-conditioned sites.

    Ordinary fits must be checked numerically against the historical scalar
    routine before this helper is used for a large R2-2 run.
    """
    quality, _ = fit_sites(x, y, np.zeros((len(x), 1)), min_train=min_train,
                           include_coefficients=True)
    fallback = np.flatnonzero((quality.n_train_joint >= min_train) &
                              (quality.fit_status == "rank_deficient"))
    for i in fallback:
        valid = np.isfinite(x[i]) & np.isfinite(y[i])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coefficients = np.polyfit(x[i, valid], y[i, valid], 2)
        if np.isfinite(coefficients).all():
            quality.loc[i, ["a2h_quadratic", "a2h_linear", "a2h_intercept"]] = coefficients
            quality.loc[i, "fit_status"] = "quadratic_rank_warning"
    return quality


def site_metrics(observed: np.ndarray, predicted: np.ndarray, min_corr_n=3) -> pd.DataFrame:
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("Observed and predicted arrays must have matching site x sample shape")
    valid = np.isfinite(observed) & np.isfinite(predicted)
    n = valid.sum(axis=1)
    denom = np.maximum(n, 1)
    y = np.where(valid, observed, 0.0)
    p = np.where(valid, predicted, 0.0)
    mean_y = y.sum(axis=1) / denom
    mean_p = p.sum(axis=1) / denom
    yc = np.where(valid, observed - mean_y[:, None], 0.0)
    pc = np.where(valid, predicted - mean_p[:, None], 0.0)
    sxy = (yc * pc).sum(axis=1)
    syy = (yc * yc).sum(axis=1)
    spp = (pc * pc).sum(axis=1)
    error = p - y
    sse = (error * error).sum(axis=1)
    sae = np.abs(error).sum(axis=1)
    bias = error.sum(axis=1) / denom
    pearson = np.full(len(n), np.nan)
    varying = (syy > 1e-24) & (spp > 1e-24)
    eligible = (n >= min_corr_n) & varying
    pearson[eligible] = sxy[eligible] / np.sqrt(syy[eligible] * spp[eligible])
    reason = np.full(len(n), "defined", dtype=object)
    reason[n < min_corr_n] = "insufficient_pairs"
    reason[(n >= min_corr_n) & (syy <= 1e-24)] = "constant_observed"
    reason[(n >= min_corr_n) & (syy > 1e-24) & (spp <= 1e-24)] = "constant_prediction"
    frame = pd.DataFrame({
        "n": n, "rmse": np.sqrt(sse / denom), "mae": sae / denom,
        "bias_pred_minus_obs": bias, "observed_variance": syy / denom,
        "pearson_r": pearson, "correlation_reason": reason,
        "sse": sse, "sae": sae, "sxy": sxy, "syy": syy, "spp": spp,
    })
    frame.loc[n == 0, ["rmse", "mae", "bias_pred_minus_obs", "observed_variance"]] = np.nan
    return frame


def pooled_metrics(site: pd.DataFrame) -> dict:
    n = int(site.n.sum())
    centered = site.loc[(site.n >= 2) & (site.syy > 1e-24) & (site.spp > 1e-24)]
    sxy, syy, spp = (float(centered[column].sum()) for column in ("sxy", "syy", "spp"))
    return {
        "n_valid_pairs": n, "n_sites_with_pairs": int((site.n > 0).sum()),
        "pooled_rmse": float(np.sqrt(site.sse.sum() / n)) if n else np.nan,
        "n_centered_sites": len(centered), "n_centered_pairs": int(centered.n.sum()),
        "centered_pearson_r": sxy / np.sqrt(syy * spp) if syy > 0 and spp > 0 else np.nan,
        "median_site_rmse": float(site.rmse.median()),
        "median_defined_site_pearson_r": float(site.pearson_r.median()),
        "n_defined_site_pearson_r": int(site.pearson_r.notna().sum()),
    }
