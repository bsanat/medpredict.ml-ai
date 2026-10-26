from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT
FEATURES = ["age", "sex", "bmi", "totalCholesterol", "glucose"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config() -> dict:
    return read_json(ROOT / "configs/experiment.json")


def versions() -> dict:
    packages = ["numpy", "pandas", "pyreadstat", "scipy", "scikit-learn", "torch", "onnx", "onnxruntime", "skl2onnx"]
    return {name: importlib.metadata.version(name) for name in packages}


def source_manifest() -> dict:
    return {str(path.relative_to(SOURCE_ROOT)): sha256(path)
            for path in sorted((SOURCE_ROOT / "src/medpredict_ml").glob("*.py"))}


def guard_test_unopened() -> None:
    if any((ROOT / "reports" / name).exists() for name in ["test_evaluation.json", "test_opened.json"]):
        raise RuntimeError("Test has already been opened. Preserve this experiment; use a new experiment workspace for further development.")

