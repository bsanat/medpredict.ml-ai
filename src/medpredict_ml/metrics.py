from __future__ import annotations

import numpy as np


def regression_metrics(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict:
    y, prediction, weight = [np.asarray(value, float).reshape(-1) for value in [y, prediction, weight]]
    if len(y) < 2 or len(y) != len(prediction) or len(y) != len(weight):
        raise ValueError("Metric arrays must have equal length >=2")
    if not all(np.isfinite(value).all() for value in [y, prediction, weight]) or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError("Invalid predictions or weights")
    error = prediction - y
    absolute = np.abs(error)
    denominator = np.sum((y-y.mean())**2)
    weighted_denominator = np.sum(weight * (y-np.average(y, weights=weight))**2)
    return {"n": len(y), "mae": float(absolute.mean()), "weighted_mae": float(np.average(absolute, weights=weight)),
        "rmse": float(np.sqrt(np.mean(error**2))), "weighted_rmse": float(np.sqrt(np.average(error**2, weights=weight))),
        "r2": float(1 - np.sum(error**2)/denominator) if denominator else float(np.sum(error**2) == 0),
        "weighted_r2": float(1 - np.sum(weight*error**2)/weighted_denominator) if weighted_denominator else float(np.sum(weight*error**2) == 0),
        "median_absolute_error": float(np.median(absolute)), "mean_signed_error": float(error.mean()),
        "weighted_mean_signed_error": float(np.average(error, weights=weight))}

