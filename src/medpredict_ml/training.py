from __future__ import annotations

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .common import FEATURES, ROOT, config, guard_test_unopened, read_json, sha256, source_manifest, versions, write_json
from .metrics import regression_metrics
from .model import MLP, RawInputModel, fit_preprocessor, train_network


def dataset() -> pd.DataFrame:
    path = ROOT / "data/processed/dataset.csv"
    protocol = read_json(ROOT / "reports/protocol.json")
    if sha256(path) != protocol["dataset_sha256"] or sha256(ROOT / "configs/experiment.json") != protocol["config_sha256"]:
        raise ValueError("Dataset or protocol changed after split registration")
    return pd.read_csv(path)


def arrays(frame: pd.DataFrame):
    return frame[FEATURES].to_numpy(np.float32), frame.target.to_numpy(float), frame.weight.to_numpy(float)


def transform(x: np.ndarray, preprocessing: dict) -> np.ndarray:
    return (x-np.array(preprocessing["x_mean"]))/np.array(preprocessing["x_scale"])


def load_mlp():
    checkpoint = torch.load(ROOT / "artifacts/mlp.pt", map_location="cpu", weights_only=True)
    model = MLP(checkpoint["candidate"]["hidden"], checkpoint["candidate"]["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return RawInputModel(model, checkpoint["preprocessing"]).eval()


def predictor(family: str):
    if family == "mlp":
        network = load_mlp()
        def inference(x):
            with torch.no_grad():
                return network(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1)
        return inference
    state = joblib.load(ROOT / "artifacts/baselines.joblib")
    if family == "mean":
        return lambda x: np.full(len(x), state["mean"])
    if family == "ridge":
        return lambda x: state["ridge"].predict(transform(x, state["preprocessing"]))
    if family == "gradient_boosting":
        return state["gradient_boosting"].predict
    raise ValueError("Unknown registered model family")


def predict(family: str, x: np.ndarray) -> np.ndarray:
    return predictor(family)(x)


def train() -> None:
    guard_test_unopened()
    settings = config()
    frame = dataset()
    training = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    # Test features and targets are deliberately not accessed in training/tuning.
    x, y, weight = arrays(training)
    xv, yv, wv = arrays(validation)
    prep = fit_preprocessor(x, y, weight)
    artifact = ROOT / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    candidates, chosen, timings = [], {}, {}
    start = time.perf_counter()
    chosen["mean"] = float(np.average(y, weights=weight))
    mean_metrics = regression_metrics(yv, np.full(len(yv), chosen["mean"]), wv)
    candidates.append({"family": "mean", "config": {}, "validation": mean_metrics})

    best = float("inf")
    for alpha in settings["ridge_alphas"]:
        model = Ridge(alpha=alpha).fit(transform(x, prep), y, sample_weight=weight/weight.mean())
        metrics = regression_metrics(yv, model.predict(transform(xv, prep)), wv)
        candidates.append({"family": "ridge", "config": {"alpha": alpha}, "validation": metrics})
        if metrics["weighted_mae"] < best:
            best, chosen["ridge"] = metrics["weighted_mae"], model
    timings["ridge_and_mean_seconds"] = time.perf_counter()-start

    start = time.perf_counter()
    best = float("inf")
    for candidate in settings["gradient_boosting"]:
        model = HistGradientBoostingRegressor(**candidate, early_stopping=False, random_state=settings["seed"])
        model.fit(x, y, sample_weight=weight/weight.mean())
        metrics = regression_metrics(yv, model.predict(xv), wv)
        candidates.append({"family": "gradient_boosting", "config": candidate, "validation": metrics})
        if metrics["weighted_mae"] < best:
            best, chosen["gradient_boosting"] = metrics["weighted_mae"], model
    timings["gradient_boosting_seconds"] = time.perf_counter()-start
    chosen["preprocessing"] = prep
    joblib.dump(chosen, artifact / "baselines.joblib")

    start = time.perf_counter()
    best_average, selected_checkpoint, all_curves = float("inf"), None, []
    for index, candidate in enumerate(settings["mlp"]):
        seed_results, first_checkpoint = [], None
        for seed in settings["mlp_seeds"]:
            model, history, score = train_network(x, y, weight, xv, yv, wv, prep, candidate, settings, seed)
            all_curves.append({"candidate": index, "seed": seed, "history": history})
            seed_results.append(score)
            if seed == settings["mlp_seeds"][0]:
                first_checkpoint = {"state_dict": model.state_dict(), "preprocessing": prep,
                                    "candidate": candidate, "seed": seed, "candidate_index": index}
            print(json.dumps({"event": "trained_mlp", "candidate": index, "seed": seed,
                              "epochs": len(history), "validation_weighted_mae": score}), flush=True)
        candidates.append({"family": "mlp_configuration", "config": candidate,
                           "validation_seed_mae": seed_results, "mean_seed_mae": float(np.mean(seed_results))})
        if np.mean(seed_results) < best_average:
            best_average, selected_checkpoint = float(np.mean(seed_results)), first_checkpoint
    torch.save(selected_checkpoint, artifact / "mlp.pt")
    timings["mlp_seconds"] = time.perf_counter()-start
    write_json(ROOT / "reports/training_curves.json", all_curves)

    family_validation = {family: regression_metrics(yv, predict(family, xv), wv)
                         for family in ["mean", "ridge", "gradient_boosting", "mlp"]}
    winner = min(family_validation, key=lambda family: family_validation[family]["weighted_mae"])
    selection = {"family": winner, "modelVersion": f"MedPredict-SBP-{winner.replace('_', '-')}-{settings['artifact_version']}",
        "featureOrder": FEATURES, "selectionRule": settings["selection_rule"], "selectionSplit": "validation",
        "validation": family_validation, "testUsedForSelection": False, "preprocessing": prep,
        "mlp_selected_config": selected_checkpoint["candidate"], "mlp_fixed_representative_seed": selected_checkpoint["seed"],
        "mlp_seed_selection": "architecture selected by mean validation MAE across seeds; fixed first seed used, not the best seed",
        "protocol_sha256": sha256(ROOT / "reports/protocol.json"), "packages": versions(),
        "source_sha256": source_manifest(),
        "training_seconds": timings, "platform": platform.platform(), "cpu_threads": 1,
        "artifact_sha256": {name: sha256(artifact / name) for name in ["baselines.joblib", "mlp.pt"]}}
    write_json(artifact / "selection.json", selection)
    write_json(ROOT / "reports/validation_results.json", {"candidates": candidates, "family_validation": family_validation})
    print(json.dumps({"event": "selection_frozen", "family": winner, "validation_weighted_mae": family_validation[winner]["weighted_mae"]}), flush=True)
