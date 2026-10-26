import numpy as np
import pytest
from medpredict_ml.classification.evaluation import metrics
from medpredict_ml.classification.model import TabularMLP


def test_confusion_metrics_and_class_order_are_hand_checkable():
    result = metrics([0, 0, 1, 1], [.1, .8, .3, .9], threshold=.5)
    assert result["confusion_matrix"] == [[1., 1.], [1., 1.]]
    assert result["sensitivity"] == result["specificity"] == result["ppv"] == result["npv"] == .5
    assert metrics([0, 0], [.1, .2])["auroc"] is None
    assert metrics([0, 0], [.1, .2])["average_precision"] is None


@pytest.mark.parametrize('p', [[np.nan], [np.inf], [-.1], [1.1]])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError):
        metrics([1], p)


def test_mlp_is_repeatable_and_handles_one_row_inference():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(80, 5))
    y = (x[:, 0] > 0).astype(int)
    a = TabularMLP(max_epochs=3, patience=2).fit(x, y)
    b = TabularMLP(max_epochs=3, patience=2).fit(x, y)
    np.testing.assert_allclose(a.predict_proba(x[:1]), b.predict_proba(x[:1]), rtol=0, atol=0)
    assert a.predict_proba(x[:1]).shape == (1, 2)
