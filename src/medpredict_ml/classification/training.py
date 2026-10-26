from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from ..common import FEATURES, read_json, sha256, versions, write_json
from .data import TASK, dataset, guard_unopened
from .evaluation import metrics, bootstrap
from .model import TabularMLP, CalibratedModel


def candidates(settings):
    for c in settings["logistic_c"]:
        yield "linear", {"C": c}, make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)), [2026]
    for lr in settings["hgb_learning_rates"]:
        for leaf in settings["hgb_leaf_nodes"]:
            config = {"learning_rate": lr, "max_leaf_nodes": leaf}
            yield "hgb", config, HistGradientBoostingClassifier(**config, max_iter=200, min_samples_leaf=20,
                l2_regularization=1, early_stopping=False, random_state=settings["seed"]), [2026]
    for hidden in settings["mlp_hidden"]:
        yield "mlp", {"hidden": hidden}, TabularMLP(hidden=tuple(hidden), max_epochs=settings["max_epochs"],
            patience=settings["patience"], batch_size=settings["batch_size"]), settings["mlp_seeds"]


def train(run: Path):
    guard_unopened(run)
    if (run / "selection.json").exists():
        raise RuntimeError("Selection already frozen; use a separate experiment for new selection")
    frame, protocol = dataset(run)
    cfg = protocol["config"]
    part = frame[frame.split == "train"]
    x, y = part[FEATURES].to_numpy(float), part.target.to_numpy(int)
    folds = list(StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"]).split(x, y))
    results, fitted_candidates, started = [], [], time.monotonic()
    for family, params, prototype, seeds in candidates(cfg):
        scores = []
        for seed in seeds:
            for fit, val in folds:
                if time.monotonic()-started > cfg["wall_clock_seconds"]:
                    write_json(run / "incomplete_cv.json", results)
                    raise RuntimeError("Registered compute budget exhausted; no selection made")
                model = clone(prototype)
                if family == "mlp":
                    model.set_params(seed=seed)
                try:
                    model.fit(x[fit], y[fit])
                    score = float(average_precision_score(y[val], model.predict_proba(x[val])[:, 1]))
                except Exception:
                    write_json(run / "failed_cv.json", {"completed_candidates": results, "family": family,
                        "parameters": params, "seed": seed, "failure": "fit_or_score_failed"})
                    raise
                scores.append(score)
        entry = {"family": family, "parameters": params, "fold_seed_ap": scores, "mean_ap": float(np.mean(scores))}
        results.append(entry)
        fitted_candidates.append(prototype)
        write_json(run / "cv_progress.json", results)
        print(json.dumps(entry), flush=True)
    best = max(r["mean_ap"] for r in results)
    eligible = [i for i, r in enumerate(results) if r["mean_ap"] >= best-cfg["simplicity_margin_ap"]]
    priority = {"linear": 0, "hgb": 1, "mlp": 2}
    winner = min(eligible, key=lambda i: (priority[results[i]["family"]], -results[i]["mean_ap"], i))
    model = clone(fitted_candidates[winner]).fit(x, y)
    glucose = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)).fit(x[:, [4]], y)
    dummy = DummyClassifier(strategy="prior").fit(x, y)
    joblib.dump({"selected": model, "glucose_only": glucose, "dummy": dummy}, run / "models.joblib")
    write_json(run / "selection.json", {"selected": results[winner], "candidates": results,
        "elapsed_seconds": time.monotonic()-started, "test_used_for_selection": False,
        "protocol_sha256": sha256(run / "protocol.json"), "models_sha256": sha256(run / "models.joblib"),
        "packages": versions(), "clinical_threshold": None})


def calibrate(run: Path):
    guard_unopened(run)
    if (run / "calibration.json").exists():
        raise RuntimeError("Calibration already frozen")
    frame, _ = dataset(run)
    selection = read_json(run / "selection.json")
    if sha256(run / "models.joblib") != selection["models_sha256"]:
        raise ValueError("Selected model hash mismatch")
    models = joblib.load(run / "models.joblib")  # Trusted local experiment artifacts only.
    part = frame[frame.split == "calibration"]
    x, y = part[FEATURES].to_numpy(float), part.target.to_numpy(int)
    calibrator = LogisticRegression(penalty=None, max_iter=2000).fit(CalibratedModel.logits(models["selected"], x), y)
    calibrated = CalibratedModel(models["selected"], calibrator)
    joblib.dump(calibrated, run / "calibrated.joblib")
    threshold = frame[frame.split == "threshold"]
    probability = calibrated.predict_proba(threshold[FEATURES].to_numpy(float))[:, 1]
    write_json(run / "calibration.json", {"method": "sigmoid on held-out calibration logits", "n": len(part),
        "positives": int(y.sum()), "selection_sha256": sha256(run / "selection.json"),
        "model_sha256": sha256(run / "calibrated.joblib"), "clinical_threshold": None,
        "threshold_policy": "No clinician-approved decision threshold; exploratory points only, not clinical recommendations",
        "threshold_cohort_points": [metrics(threshold.target.to_numpy(), probability, threshold=t) for t in [.1, .2, .5]]})


def evaluate(run: Path):
    guard_unopened(run)
    selection, calibration = read_json(run / "selection.json"), read_json(run / "calibration.json")
    if (sha256(run / "selection.json") != calibration["selection_sha256"]
        or sha256(run / "models.joblib") != selection["models_sha256"]
        or sha256(run / "calibrated.joblib") != calibration["model_sha256"]):
        raise ValueError("Model selection/calibration hashes changed")
    # Create the irreversible marker before loading held-out labels into evaluation.
    marker = run / "test_opened.json"
    with marker.open("x") as handle:
        json.dump({"selection_sha256": sha256(run / "selection.json"), "calibration_sha256": sha256(run / "calibration.json")}, handle)
    frame, protocol = dataset(run)
    test = frame[frame.split == "test"].reset_index(drop=True)
    x, y = test[FEATURES].to_numpy(float), test.target.to_numpy(int)
    models, calibrated = joblib.load(run / "models.joblib"), joblib.load(run / "calibrated.joblib")
    scores = {"selected_raw": models["selected"].predict_proba(x)[:, 1],
        "selected_calibrated": calibrated.predict_proba(x)[:, 1],
        "glucose_only": models["glucose_only"].predict_proba(x[:, [4]])[:, 1], "dummy": models["dummy"].predict_proba(x)[:, 1]}
    report = {"task": TASK, "label": "single HbA1c >=6.5%, not confirmed diabetes", "n": len(test),
        "selected_family": selection["selected"]["family"], "clinical_threshold": None,
        "unweighted": {k: metrics(y, p) for k, p in scores.items()},
        "survey_weighted_sensitivity": {k: metrics(y, p, test.weight.to_numpy()) for k, p in scores.items()},
        "subgroups": {}, "coverage": 1., "coverage_scope": "complete eligible internal test only; not all application users",
        "bootstrap": bootstrap(test, scores["selected_calibrated"], scores["glucose_only"], protocol["config"]["bootstrap_replicates"]),
        "limitations": read_json(run / "cohort_audit.json")["limitations"] + [
            "Single internal holdout; no external/clinical/Kazakhstan validation",
            "No adjudicated diagnoses or clinician-approved threshold; calibration precision limited by positive count"]}
    for label, mask in {"male": test.sex == 0, "female": test.sex == 1,
                        "age_18_39": test.age < 40, "age_40_59": test.age.between(40, 59), "age_60_79": test.age >= 60}.items():
        report["subgroups"][label] = metrics(y[mask], scores["selected_calibrated"][mask]) if mask.any() else None
    write_json(run / "test_evaluation.json", report)
    print(json.dumps({k: {m: v[m] for m in ["average_precision", "auroc", "brier"]} for k, v in report["unweighted"].items()}), flush=True)
