from __future__ import annotations

import time
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FEATURES, ROOT, config, read_json, sha256, write_json
from .metrics import regression_metrics
from .training import arrays, dataset, predictor


def cluster_bootstrap(frame: pd.DataFrame, predictions: dict, replicates: int, seed: int) -> dict:
    """Approximate stratified PSU bootstrap conditional on this complete-case test split."""
    rng = np.random.default_rng(seed)
    groups = []
    for _, stratum in frame.groupby("stratum"):
        groups.append([group.index.to_numpy() for _, group in stratum.groupby("psu")])
    samples = {family: [] for family in predictions}
    y, weights = frame.target.to_numpy(), frame.weight.to_numpy()
    for _ in range(replicates):
        indices = np.concatenate([np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))]) for units in groups])
        for family, values in predictions.items():
            samples[family].append(float(np.average(np.abs(values[indices]-y[indices]), weights=weights[indices])))
    result = {family: {"weighted_mae_95_interval": np.quantile(values, [.025, .975]).tolist()}
              for family, values in samples.items()}
    selected = read_json(ROOT / "artifacts/selection.json")["family"]
    result["selected_minus_mean_mae_95_interval"] = np.quantile(np.array(samples[selected])-np.array(samples["mean"]), [.025, .975]).tolist()
    result["method"] = "1000 stratified masked-PSU resamples, conditional complete-case held-out sample; approximate, not full survey population variance"
    return result


def evaluate() -> None:
    target_file = ROOT / "reports/test_evaluation.json"
    if target_file.exists():
        raise RuntimeError("Held-out test report already exists; use the saved results, do not retune on it")
    selection = read_json(ROOT / "artifacts/selection.json")
    if selection["protocol_sha256"] != sha256(ROOT / "reports/protocol.json"):
        raise ValueError("Protocol differs from the frozen model selection")
    for name, digest in selection["artifact_sha256"].items():
        if sha256(ROOT / "artifacts" / name) != digest:
            raise ValueError("Trained artifact changed after selection")
    opened = ROOT / "reports/test_opened.json"
    if opened.exists():
        raise RuntimeError("Test opening was already recorded. Do not train or retune; preserve this experiment.")
    write_json(opened, {"opened_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_sha256": sha256(ROOT / "artifacts/selection.json"), "protocol_sha256": selection["protocol_sha256"],
        "rule": "Persistent lock written before accessing held-out targets; training and preparation are now disabled"})
    frame = dataset()
    testing = frame[frame.split == "test"].reset_index(drop=True)
    x, y, weight = arrays(testing)
    functions = {family: predictor(family) for family in ["mean", "ridge", "gradient_boosting", "mlp"]}
    predictions = {family: fn(x) for family, fn in functions.items()}
    results = {family: regression_metrics(y, values, weight) for family, values in predictions.items()}
    selected = selection["family"]
    subgroups = {}
    grouping = {
        "sex": testing.sex.map({0: "male", 1: "female"}),
        "age": pd.cut(testing.age, [18, 40, 60, 80], right=False).astype(str),
        "target_range_research_bins": pd.cut(testing.target, config()["target_range_edges"], right=False).astype(str),
        "medication": testing.medication,
    }
    for name, labels in grouping.items():
        subgroups[name] = {}
        for label in sorted(labels.unique()):
            mask = (labels == label).to_numpy()
            if mask.sum() >= 2:
                subgroups[name][label] = regression_metrics(y[mask], predictions[selected][mask], weight[mask])
                subgroups[name][label]["small_group_caution"] = bool(mask.sum() < 50)
    interval = cluster_bootstrap(testing, predictions, config()["bootstrap_replicates"], config()["seed"]+1)
    benchmark = {}
    for family, function in functions.items():
        for _ in range(10):
            function(x[:1])
        latencies = []
        for index in range(200):
            before = time.perf_counter_ns()
            function(x[index % len(x):index % len(x)+1])
            latencies.append((time.perf_counter_ns()-before)/1e6)
        benchmark[family] = {"p50_ms": float(np.median(latencies)), "p95_ms": float(np.quantile(latencies, .95)),
                             "iterations": len(latencies), "scope": "preloaded Python model, one row, CPU"}
    # Sensitivity analysis uses validation only; never fed back into model selection.
    validation = frame[frame.split == "validation"]
    xv, yv, wv = arrays(validation)
    baseline_mae = float(np.average(np.abs(functions[selected](xv)-yv), weights=wv))
    sensitivity = {}
    rng = np.random.default_rng(config()["seed"]+2)
    for index, feature in enumerate(FEATURES):
        changes = []
        for _ in range(10):
            shuffled = xv.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            changes.append(float(np.average(np.abs(functions[selected](shuffled)-yv), weights=wv))-baseline_mae)
        sensitivity[feature] = {"mean_mae_increase": float(np.mean(changes)), "std": float(np.std(changes)),
                                "interpretation": "non-causal validation permutation sensitivity; correlated inputs limit interpretation"}
    report = {"selected_family": selected, "selected_by": "validation before test access",
              "selection_sha256": sha256(ROOT / "artifacts/selection.json"), "test_used_for_tuning": False,
              "metrics": results, "uncertainty": interval, "subgroups_selected_model": subgroups,
              "benchmark": benchmark, "validation_permutation_sensitivity": sensitivity,
              "limitations": ["cross-sectional current-measurement estimation", "complete-case fasting subsample",
                 "internal participant holdout; no external clinical validation", "US population; Kazakhstan transfer not measured",
                 "subgroup bins are descriptive research groups, not diagnostic categories"]}
    write_json(target_file, report)
    pd.DataFrame({"observed": y, **predictions}).to_csv(ROOT / "reports/test_predictions.csv", index=False)
    make_plots(frame, testing, predictions, report)
    print(f"Final test opened once: {len(testing)} participants; selected={selected}; metrics saved", flush=True)


def make_plots(frame: pd.DataFrame, testing: pd.DataFrame, predictions: dict, report: dict) -> None:
    destination = ROOT / "reports/figures"
    destination.mkdir(parents=True, exist_ok=True)
    selected = report["selected_family"]
    y = testing.target.to_numpy()
    estimated = predictions[selected]
    plt.style.use("seaborn-v0_8-whitegrid")

    def save(name):
        plt.tight_layout()
        plt.savefig(destination / name, dpi=160)
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y, estimated, s=12, alpha=.4)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="black", linestyle="--")
    plt.xlabel("Observed SBP (mmHg)")
    plt.ylabel("Estimated SBP (mmHg)")
    plt.title(f"Held-out test: {selected}, n={len(y)}")
    save("actual-vs-predicted.png")
    plt.figure(figsize=(6, 4))
    plt.hist(estimated-y, bins=30, color="#17447a")
    plt.xlabel("Estimate minus observed SBP (mmHg)")
    plt.ylabel("Participants")
    save("residuals.png")
    plt.figure(figsize=(7, 4))
    families = list(report["metrics"])
    plt.bar(families, [report["metrics"][f]["weighted_mae"] for f in families], color="#17447a")
    plt.ylabel("Weighted test MAE (mmHg)")
    save("model-comparison.png")
    for group in ["age", "sex", "target_range_research_bins", "medication"]:
        groups = report["subgroups_selected_model"][group]
        plt.figure(figsize=(7, 4))
        plt.bar(list(groups), [m["weighted_mae"] for m in groups.values()])
        plt.ylabel("Weighted MAE (mmHg)")
        plt.title(group + " (descriptive groups)")
        save(f"error-by-{group}.png")
    checkpoint_config = read_json(ROOT / "artifacts/selection.json")["mlp_selected_config"]
    histories = read_json(ROOT / "reports/training_curves.json")
    # Plot the fixed seed for the architecture selected on validation.
    candidates = config()["mlp"]
    chosen_index = candidates.index(checkpoint_config)
    history = next(item["history"] for item in histories if item["candidate"] == chosen_index and item["seed"] == config()["mlp_seeds"][0])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["train_weighted_mse_standardized"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Training weighted MSE (standardized)")
    axes[1].plot([r["epoch"] for r in history], [r["validation_weighted_mae"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation weighted MAE (mmHg)")
    save("training-validation.png")
    numeric = frame[[*FEATURES, "target"]]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, column in zip(axes.flat, numeric):
        axis.hist(numeric[column], bins=30)
        axis.set_title(column)
    save("data-distributions.png")
    plt.figure(figsize=(7, 6))
    plt.imshow(numeric.corr(), vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(6), numeric.columns, rotation=40, ha="right")
    plt.yticks(range(6), numeric.columns)
    plt.colorbar(label="Descriptive Pearson correlation")
    save("data-correlations.png")
