from __future__ import annotations

import json
import time

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from torch import nn

from .common import FEATURES, ROOT, config, read_json, sha256, write_json
from .model import RawInputModel
from .training import arrays, dataset, load_mlp, predictor


class MeanModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("value", torch.tensor(value, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :1] * 0 + self.value


def torch_for_family(family: str) -> nn.Module:
    if family == "mlp":
        return load_mlp()
    state = joblib.load(ROOT / "artifacts/baselines.joblib")
    if family == "mean":
        return MeanModel(state["mean"]).eval()
    if family == "ridge":
        regression = state["ridge"]
        linear = nn.Linear(5, 1)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(regression.coef_, dtype=torch.float32).reshape(1, -1))
            linear.bias.copy_(torch.tensor([regression.intercept_], dtype=torch.float32))
        prep = {**state["preprocessing"], "y_mean": 0., "y_scale": 1.}
        return RawInputModel(linear, prep).eval()
    raise ValueError("This family uses a tree ONNX converter")


def export_family(family: str, path) -> None:
    if family == "gradient_boosting":
        state = joblib.load(ROOT / "artifacts/baselines.joblib")
        graph = convert_sklearn(state[family], initial_types=[("features", FloatTensorType([None, 5]))], target_opset=17)
        old_name = graph.graph.output[0].name
        graph.graph.output[0].name = "predictedValue"
        for node in graph.graph.node:
            for index, name in enumerate(node.output):
                if name == old_name:
                    node.output[index] = "predictedValue"
            for index, name in enumerate(node.input):
                if name == old_name:
                    node.input[index] = "predictedValue"
    else:
        # A simple feed-forward graph needs no additional exporter service or Python runtime at serving time.
        torch.onnx.export(torch_for_family(family), torch.zeros(2, 5), str(path),
            input_names=["features"], output_names=["predictedValue"], opset_version=17,
            dynamic_axes={"features": {0: "batch"}, "predictedValue": {0: "batch"}}, dynamo=False)
        graph = onnx.load(path)
    graph.ir_version = min(graph.ir_version, 10)
    helper.set_model_props(graph, {"predictionType": "estimated_current_measurement", "featureOrder": json.dumps(FEATURES),
        "sexEncoding": json.dumps({"male": 0, "female": 1}), "unit": "mmHg", "family": family,
        "selectionSha256": sha256(ROOT / "artifacts/selection.json")})
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def export() -> None:
    selection = read_json(ROOT / "artifacts/selection.json")
    if not (ROOT / "reports/test_evaluation.json").exists():
        raise ValueError("Complete the registered held-out evaluation before packaging")
    frame = dataset()
    training = frame[frame.split == "train"]
    family = selection["family"]
    evaluation = read_json(ROOT / "reports/test_evaluation.json")
    test_mae = evaluation["metrics"][family]["weighted_mae"]
    path = ROOT / "artifacts/model.onnx"
    export_family(family, path)
    export_family("mlp", ROOT / "artifacts/mlp-experimental.onnx")
    fields = []
    labels = {"age": ("Возраст", "лет"), "bmi": ("Индекс массы тела", "кг/м²"),
              "totalCholesterol": ("Общий холестерин", "ммоль/л"), "glucose": ("Глюкоза плазмы натощак", "ммоль/л")}
    for feature in FEATURES:
        if feature == "sex":
            fields.append({"key": "sex", "label": "Пол", "type": "select", "required": True,
                "options": [{"value": "male", "label": "Мужской"}, {"value": "female", "label": "Женский"}]})
        else:
            label, unit = labels[feature]
            fields.append({"key": feature, "label": label, "type": "number", "required": True, "unit": unit,
                "min": float(training[feature].min()), "max": float(training[feature].max()),
                **({"integer": True} if feature == "age" else {})})
    metadata = {"schemaVersion": 1, "modelVersion": selection["modelVersion"], "sha256": sha256(path),
        "inputName": "features", "outputName": "predictedValue", "featureOrder": FEATURES,
        "sexEncoding": {"male": 0, "female": 1}, "indicatorName": "Оценка систолического давления",
        "unit": "мм рт. ст.", "decimalPlaces": 1, "predictionType": "estimated_current_measurement", "demo": False,
        "fields": fields, "researchOnly": True, "qualityStatus": "experimental",
        "disclaimer": f"На внутренней тестовой выборке NHANES взвешенная MAE составляет {test_mae:.1f} мм рт. ст. Эта средняя ошибка не является интервалом для отдельного результата. Высокие значения давления могут существенно недооцениваться.",
        "selection": {"family": family, "split": "validation", "weightedMae": selection["validation"][family]["weighted_mae"],
                      "mlpExperimental": True},
        "domainNote": "Input bounds are observed training support, not clinical normal ranges. Complete-case adults aged 18–79 with fasting plasma glucose.",
        "dataset": config()["dataset"], "dataManifestSha256": sha256(ROOT / "reports/data_manifest.json"),
        "protocolSha256": sha256(ROOT / "reports/protocol.json"), "packages": selection["packages"]}
    write_json(ROOT / "artifacts/metadata.json", metadata)
    cases = [
        {"age": 52, "sex": "male", "bmi": 27.4, "totalCholesterol": 5.6, "glucose": 5.9},
        {"age": 35, "sex": "female", "bmi": 23.5, "totalCholesterol": 4.5, "glucose": 4.8},
        {"age": 70, "sex": "male", "bmi": 30., "totalCholesterol": 6.1, "glucose": 6.4},
        {"age": 45, "sex": "female", "bmi": 26., "totalCholesterol": 5., "glucose": 5.2},
    ]
    numeric = np.array([[case[f] if f != "sex" else {"male": 0, "female": 1}[case[f]] for f in FEATURES] for case in cases], dtype=np.float32)
    values = predictor(family)(numeric)
    write_json(ROOT / "artifacts/golden.json", {"absoluteTolerance": config()["parity_absolute_tolerance_mmhg"],
        "featureOrder": FEATURES, "source": "Synthetic test inputs, not training observations or clinical reference values",
        "cases": [{"input": case, "expected": float(value)} for case, value in zip(cases, values)]})
    print(f"Exported selected {family} and experimental MLP with embedded transforms", flush=True)


def verify() -> None:
    metadata = read_json(ROOT / "artifacts/metadata.json")
    path = ROOT / "artifacts/model.onnx"
    if sha256(path) != metadata["sha256"]:
        raise ValueError("ONNX checksum does not match metadata")
    frame = dataset()
    testing = frame[frame.split == "test"]
    x, _, _ = arrays(testing)
    golden = read_json(ROOT / "artifacts/golden.json")
    synthetic = np.array([[case["input"][f] if f != "sex" else {"male": 0, "female": 1}[case["input"][f]] for f in FEATURES] for case in golden["cases"]], dtype=np.float32)
    x = np.concatenate([x, synthetic])
    reports = {}
    for family, filename in [(metadata["selection"]["family"], "model.onnx"), ("mlp", "mlp-experimental.onnx")]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(ROOT / "artifacts" / filename), sess_options=options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].name != "features" or session.get_inputs()[0].shape[1] != 5:
            raise ValueError("Unexpected ONNX input schema")
        output = session.run(["predictedValue"], {"features": x})[0]
        if output.shape != (len(x), 1) or not np.isfinite(output).all():
            raise ValueError("Invalid ONNX output schema/values")
        expected = predictor(family)(x)
        delta = np.abs(output.reshape(-1)-expected)
        if delta.max() > config()["parity_absolute_tolerance_mmhg"]:
            raise ValueError(f"ONNX numerical parity failed for {family}: {delta.max()}")
        times = []
        for index in range(1020):
            before = time.perf_counter_ns()
            session.run(["predictedValue"], {"features": x[index % len(x):index % len(x)+1]})
            if index >= 20:
                times.append((time.perf_counter_ns()-before)/1e6)
        reports[filename] = {"family": family, "cases": len(x), "max_absolute_difference_mmhg": float(delta.max()),
            "tolerance_mmhg": config()["parity_absolute_tolerance_mmhg"], "bytes": (ROOT / "artifacts" / filename).stat().st_size,
            "sha256": sha256(ROOT / "artifacts" / filename), "onnx_cpu_p50_ms": float(np.median(times)),
            "onnx_cpu_p95_ms": float(np.quantile(times, .95)), "benchmark_iterations": len(times)}
    write_json(ROOT / "reports/onnx_verification.json", reports)
    print("Python / ONNX parity passed for selected model and MLP", flush=True)
