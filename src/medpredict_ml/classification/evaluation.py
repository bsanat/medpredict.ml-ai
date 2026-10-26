from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, confusion_matrix


def metrics(y, p, weight=None, threshold=None):
    y, p = np.asarray(y), np.asarray(p, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned binary labels and probabilities required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid evaluation weights")
    both = len(np.unique(y[w > 0])) == 2
    result = {"n": len(y), "positives": int(y.sum()), "prevalence": float(np.average(y, weights=w)),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)) if both else None,
        "auroc": float(roc_auc_score(y, p, sample_weight=w)) if both else None,
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        "calibration_bins": []}
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any() and w[mask].sum() > 0:
            result["calibration_bins"].append({"lower": float(lo), "upper": float(hi), "n": int(mask.sum()),
                "predicted": float(np.average(p[mask], weights=w[mask])), "observed": float(np.average(y[mask], weights=w[mask]))})
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("Invalid threshold")
        tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1], sample_weight=w).ravel()
        ratio = lambda a, b: float(a/b) if b else None
        result.update(threshold=threshold, confusion_matrix=[[float(tn), float(fp)], [float(fn), float(tp)]],
            sensitivity=ratio(tp, tp+fn), specificity=ratio(tn, tn+fp), ppv=ratio(tp, tp+fp), npv=ratio(tn, tn+fn))
    return result


def bootstrap(frame, p, comparator, replicates=1000, seed=2026):
    """Masked-PSU resampling within strata; conditional internal evaluation only."""
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ["average_precision", "auroc", "brier", "ap_difference_vs_glucose_only"]}
    groups = [part.psu.unique() for _, part in frame.groupby("stratum")]
    strata = [part for _, part in frame.groupby("stratum")]
    for _ in range(replicates):
        multiplicity = np.zeros(len(frame))
        for part, units in zip(strata, groups):
            sampled = rng.choice(units, size=len(units), replace=True)
            for unit in sampled:
                multiplicity[part.index[part.psu == unit]] += 1
        mask = multiplicity > 0
        y = frame.target.to_numpy()[mask]
        if len(np.unique(y)) != 2:
            continue
        w = multiplicity[mask]
        scores = metrics(y, p[mask], w)
        other = average_precision_score(y, comparator[mask], sample_weight=w)
        for key in ["average_precision", "auroc", "brier"]:
            values[key].append(scores[key])
        values["ap_difference_vs_glucose_only"].append(scores["average_precision"]-other)
    return {"method": "conditional masked-PSU bootstrap within test strata; unweighted eligible-cohort estimand",
        "requested_replicates": replicates, "valid_replicates": len(values["brier"]),
        "intervals_95": {k: [float(v) for v in np.quantile(a, [.025, .975])] if a else None for k, a in values.items()},
        "limitations": "Conditional on selected cohort, held-out participants and fixed model; not full survey or Kazakhstan uncertainty"}
