import numpy as np
import onnxruntime as ort
import pytest
import torch

from medpredict_ml.common import guard_test_unopened
from medpredict_ml.model import MLP, RawInputModel, fit_preprocessor, train_network


def test_mlp_training_is_repeatable_and_has_finite_predictions():
    # Synthetic unit fixture only. Actual training/evaluation use official NHANES exclusively.
    rng = np.random.default_rng(20)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    y = 120 + 3*x[:, 0]
    w = np.ones(40)
    prep = fit_preprocessor(x[:30], y[:30], w[:30])
    candidate = {"hidden": [4], "dropout": 0., "learning_rate": .01, "weight_decay": .01}
    settings = {"max_epochs": 5, "batch_size": 10, "patience": 3}
    a = train_network(x[:30], y[:30], w[:30], x[30:], y[30:], w[30:], prep, candidate, settings, 42)
    b = train_network(x[:30], y[:30], w[:30], x[30:], y[30:], w[30:], prep, candidate, settings, 42)
    assert a[1] == b[1]
    assert a[2] == b[2]
    assert np.isfinite(a[2])


def test_embedded_preprocessing_onnx_matches_pytorch(tmp_path):
    torch.manual_seed(42)
    model = RawInputModel(MLP([4]), {"x_mean": [30., .5, 25., 5., 6.], "x_scale": [10., .5, 5., 1., 1.], "y_mean": 120., "y_scale": 15.}).eval()
    raw = torch.tensor([[52., 0., 27.4, 5.6, 5.9], [35., 1., 23., 4., 5.]])
    path = tmp_path / "synthetic.onnx"
    torch.onnx.export(model, raw, path, input_names=["features"], output_names=["predictedValue"], opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"features": raw.numpy()})[0], model(raw).detach().numpy(), atol=.001)


def test_test_open_guard_fails_before_retraining(tmp_path, monkeypatch):
    import medpredict_ml.common as common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    guard_test_unopened()
    (tmp_path / "reports/test_evaluation.json").write_text('{}')
    with pytest.raises(RuntimeError, match="already been opened"):
        guard_test_unopened()
    (tmp_path / "reports/test_evaluation.json").unlink()
    (tmp_path / "reports/test_opened.json").write_text('{}')
    with pytest.raises(RuntimeError, match="already been opened"):
        guard_test_unopened()
