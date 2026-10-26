from __future__ import annotations

from pathlib import Path
import importlib.metadata
import joblib
import numpy as np

from ..common import FEATURES, read_json, sha256, write_json
from .data import TASK, dataset

UNITS = {"age": "years", "bmi": "kg/m2", "totalCholesterol": "mmol/L", "glucose": "mmol/L"}
LIMITATIONS = ["Research only; predicts a single HbA1c criterion, not confirmed diabetes or future disease",
    "No clinical or Kazakhstan validation; no treatment decisions",
    "Unknown female pregnancy status excluded; female age support is restricted",
    "Medication questionnaire skips do not establish treatment-naive status",
    "Input bounds describe observed training support, not medical reference intervals"]


def export(run: Path, destination: Path):
    if destination.exists():
        raise ValueError("Refusing to overwrite a released bundle")
    frame, protocol = dataset(run)
    report, calibration = read_json(run / "test_evaluation.json"), read_json(run / "calibration.json")
    if sha256(run / "calibrated.joblib") != calibration["model_sha256"]:
        raise ValueError("Calibrated artifact hash mismatch")
    training = frame[frame.split == "train"]
    destination.mkdir(parents=True)
    (destination / "model.joblib").write_bytes((run / "calibrated.joblib").read_bytes())
    manifest = {"task": TASK, "schemaVersion": "2.0", "modelVersion": "D1-r2-internal-v1", "researchOnly": True,
        "clinicalValidation": "not_established", "clinicalThreshold": None, "featureOrder": FEATURES,
        "units": UNITS, "sexEncoding": {"male": 0, "female": 1}, "model_sha256": sha256(destination / "model.joblib"),
        "protocol_sha256": sha256(run / "protocol.json"), "selection_sha256": sha256(run / "selection.json"),
        "calibration_sha256": sha256(run / "calibration.json"), "sources": protocol["sources"],
        "support": {key: [float(training[key].min()), float(training[key].max())] for key in FEATURES},
        "ageBySex": {name: [int(part.age.min()), int(part.age.max())] for name, part in
            [("male", training[training.sex == 0]), ("female", training[training.sex == 1])] if len(part)},
        "packages": {k: importlib.metadata.version(k) for k in ["scikit-learn", "numpy", "torch"]},
        "selectedFamily": report["selected_family"], "limitations": LIMITATIONS}
    write_json(destination / "metadata.json", manifest)
    # Fixed synthetic vectors, never participant rows, verify serialization/serving parity.
    vectors = [[52., 0., 27.4, 5.6, 5.9], [35., 1., 23., 4., 5.], [60., 0., 30., 6., 7.]]
    model = joblib.load(run / "calibrated.joblib")
    write_json(destination / "golden.json", {"features": vectors,
        "probabilities": model.predict_proba(vectors)[:, 1].tolist(), "absolute_tolerance": 1e-7})
    manifest["golden_sha256"] = sha256(destination / "golden.json")
    write_json(destination / "metadata.json", manifest)


class Bundle:
    def __init__(self, directory: Path):
        self.metadata = m = read_json(directory / "metadata.json")
        if (m["task"] != TASK or m["schemaVersion"] != "2.0" or m["featureOrder"] != FEATURES
            or m["units"] != UNITS or m["sexEncoding"] != {"male": 0, "female": 1}
            or m["researchOnly"] is not True or m["clinicalThreshold"] is not None
            or m["clinicalValidation"] != "not_established" or not isinstance(m["modelVersion"], str)
            or not m["modelVersion"] or set(m["support"]) != set(FEATURES)):
            raise ValueError("Incompatible bundle schema")
        for key, limits in m["support"].items():
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
                raise ValueError("Invalid train support")
        if not m["ageBySex"] or set(m["ageBySex"])-{"male", "female"}:
            raise ValueError("Invalid cohort support")
        for limits in m["ageBySex"].values():
            if len(limits) != 2 or not np.isfinite(limits).all() or not 18 <= limits[0] <= limits[1] <= 79:
                raise ValueError("Invalid age support")
        for package, version in m["packages"].items():
            if package not in {"scikit-learn", "numpy", "torch"} or importlib.metadata.version(package) != version:
                raise ValueError("Incompatible model environment")
        if set(m["packages"]) != {"scikit-learn", "numpy", "torch"}:
            raise ValueError("Missing model environment")
        if (sha256(directory / "model.joblib") != m["model_sha256"]
            or sha256(directory / "golden.json") != m["golden_sha256"]):
            raise ValueError("Bundle checksum mismatch")
        self.model = joblib.load(directory / "model.joblib")  # Trusted administrator-selected local path only.
        golden = read_json(directory / "golden.json")
        if not 0 < golden["absolute_tolerance"] <= 1e-6:
            raise ValueError("Invalid reference tolerance")
        np.testing.assert_allclose(self.probabilities(golden["features"]), golden["probabilities"],
                                   rtol=0, atol=golden["absolute_tolerance"])

    def probabilities(self, features):
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all():
            raise ValueError("Invalid feature tensor")
        probability = np.asarray(self.model.predict_proba(x)[:, 1], float)
        if probability.shape != (len(x),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Invalid probability output")
        return probability
