import numpy as np
import pytest
import torch

from medpredict_ml.model import MLP, RawInputModel, fit_preprocessor
from medpredict_ml.metrics import regression_metrics


def test_scaling_statistics_use_only_supplied_training_rows():
    x = np.array([[1., 0., 20., 4., 5.], [3., 1., 30., 6., 7.]])
    prep = fit_preprocessor(x, np.array([100., 140.]), np.ones(2))
    assert prep["x_mean"] == [2., .5, 25., 5., 6.]
    assert prep["y_mean"] == 120.
    held_out = np.full((1, 5), 9999.)
    assert held_out.mean() > prep["x_mean"][0]
    assert prep["x_mean"][0] == 2.


def test_raw_model_includes_preprocessing_and_inverse_target_transform():
    model = MLP([2], 0.)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    prep = {"x_mean": [0.]*5, "x_scale": [1.]*5, "y_mean": 123., "y_scale": 10.}
    wrapper = RawInputModel(model, prep)
    assert wrapper(torch.ones(3, 5)).shape == (3, 1)
    np.testing.assert_allclose(wrapper(torch.ones(3, 5)).detach(), 123.)


def test_weighted_regression_metrics_have_known_values():
    result = regression_metrics(np.array([0., 10.]), np.array([2., 6.]), np.array([3., 1.]))
    assert result["weighted_mae"] == 2.5
    assert result["mae"] == 3.
    assert result["weighted_rmse"] == pytest.approx(np.sqrt(7.))
    assert result["median_absolute_error"] == 3.
    assert result["r2"] == pytest.approx(.6)
    assert result["n"] == 2


def test_metrics_reject_invalid_predictions_and_weights():
    with pytest.raises(ValueError):
        regression_metrics(np.array([1., 2.]), np.array([np.nan, 2.]), np.ones(2))
    with pytest.raises(ValueError):
        regression_metrics(np.array([1., 2.]), np.ones(2), np.array([0., 0.]))


def test_constant_target_r2_matches_explicit_finite_convention():
    assert regression_metrics(np.ones(3), np.ones(3), np.ones(3))["r2"] == 1.
    assert regression_metrics(np.ones(3), np.zeros(3), np.ones(3))["r2"] == 0.
