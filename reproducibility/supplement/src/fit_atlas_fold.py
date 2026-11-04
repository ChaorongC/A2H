"""Fold-local site fits matching the quadratic and constant-KDE atlas contract."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_sites(x: np.ndarray, y: np.ndarray, x_test: np.ndarray,
              min_train: int = 50, max_condition: float = 1e12,
              include_coefficients: bool = False) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if x.shape != y.shape or x.shape[0] != x_test.shape[0] or min_train < 3:
        raise ValueError("Training/test locus alignment or minimum count is invalid")
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    safe_n = np.maximum(n, 1)
    mean_x = xx.sum(axis=1) / safe_n
    mean_y = yy.sum(axis=1) / safe_n
    x_min = np.where(valid, x, np.inf).min(axis=1)
    x_max = np.where(valid, x, -np.inf).max(axis=1)
    x_min[n == 0] = np.nan
    x_max[n == 0] = np.nan
    sorted_x = np.sort(np.where(valid, x, np.inf), axis=1)
    finite_sorted = np.isfinite(sorted_x)
    unique_count = finite_sorted[:, 0].astype(int)
    unique_count += (finite_sorted[:, 1:] & (sorted_x[:, 1:] != sorted_x[:, :-1])).sum(axis=1)
    eligible = n >= min_train
    constant = eligible & (unique_count == 1)
    quadratic_candidate = eligible & (unique_count >= 3)

    # Normal equations are solved in float64 only for full-rank, well-conditioned sites.
    sums = [np.ones_like(xx) * valid, xx, xx**2, xx**3, xx**4]
    moments = [item.sum(axis=1) for item in sums]
    gram = np.stack([
        np.stack([moments[0], moments[1], moments[2]], axis=-1),
        np.stack([moments[1], moments[2], moments[3]], axis=-1),
        np.stack([moments[2], moments[3], moments[4]], axis=-1),
    ], axis=1)
    rhs = np.stack([yy.sum(axis=1), (xx * yy).sum(axis=1), (xx * xx * yy).sum(axis=1)], axis=-1)
    coefficients = np.full((len(x), 3), np.nan)
    condition = np.full(len(x), np.nan)
    if quadratic_candidate.any():
        indices = np.flatnonzero(quadratic_candidate)
        eig = np.linalg.eigvalsh(gram[indices])
        positive = eig[:, 0] > 0
        candidate_condition = np.full(len(indices), np.inf)
        candidate_condition[positive] = eig[positive, -1] / eig[positive, 0]
        condition[indices] = candidate_condition
        solvable = indices[candidate_condition <= max_condition]
        if len(solvable):
            coefficients[solvable] = np.linalg.solve(gram[solvable], rhs[solvable, :, None])[:, :, 0]
    quadratic_fit = np.isfinite(coefficients).all(axis=1)
    intercept = coefficients[:, 0].copy()
    linear = coefficients[:, 1].copy()
    squared = coefficients[:, 2].copy()
    if constant.any():
        grid = np.linspace(0.0, 1.0, 10)
        for i in np.flatnonzero(constant):
            kernel = np.exp(-0.5 * ((grid[:, None] - y[i, valid[i]]) / 0.05) ** 2).sum(axis=1)
            intercept[i] = grid[kernel.argmax()]
            linear[i] = 0.0
            squared[i] = 0.0
    x_centered = np.where(valid, x - mean_x[:, None], 0.0)
    y_centered = np.where(valid, y - mean_y[:, None], 0.0)
    sxx = (x_centered**2).sum(axis=1)
    sxy = (x_centered * y_centered).sum(axis=1)
    linear_slope = np.zeros(len(x))
    linear_valid = eligible & (sxx > 1e-12)
    linear_slope[linear_valid] = sxy[linear_valid] / sxx[linear_valid]
    linear_intercept = mean_y - linear_slope * mean_x
    reference_mean = np.where(eligible, mean_y, np.nan)
    linear_intercept[~eligible] = np.nan
    linear_slope[~eligible] = np.nan
    status = np.full(len(x), "insufficient", dtype=object)
    status[eligible] = "rank_deficient"
    status[constant] = "constant_kde"
    status[quadratic_fit] = "quadratic"
    quality = pd.DataFrame({
        "n_train_joint": n,
        "unique_input_count": unique_count,
        "train_x_min": x_min,
        "train_x_max": x_max,
        "train_x_variance": (x_centered**2).sum(axis=1) / safe_n,
        "quadratic_condition": condition,
        "fit_status": status,
    })
    if include_coefficients:
        quality = quality.assign(a2h_intercept=intercept, a2h_linear=linear,
                                 a2h_quadratic=squared,
                                 reference_linear_intercept=linear_intercept,
                                 reference_linear_slope=linear_slope)
    prediction = {
        "A2H": intercept[:, None] + linear[:, None] * x_test + squared[:, None] * x_test**2,
        "reference_mean": np.broadcast_to(reference_mean[:, None], x_test.shape),
        "reference_linear": linear_intercept[:, None] + linear_slope[:, None] * x_test,
        "m": x_test,
        "m2": x_test**2,
        "m3": x_test**3,
    }
    return quality, prediction
