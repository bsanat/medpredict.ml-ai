import json

import numpy as np
import pytest

from medpredict_ml.common import read_json, sha256, write_json, FEATURES
from medpredict_ml.classification.data import cohort, dataset, fingerprints
from medpredict_ml.classification.bundle import Bundle, export
from medpredict_ml.classification.training import train, calibrate, evaluate
from test_classification_data import tables


def test_small_synthetic_pipeline_freezes_selection_test_and_roundtrips_bundle(tmp_path):
    # This short synthetic run is a software test, not a disease-performance experiment.
    source = tables(240)
    rng = np.random.default_rng(32)
    source['P_GLU']['LBDGLUSI'] = rng.uniform(4, 8, 240)
    source['P_BMX']['BMXBMI'] = rng.uniform(20, 35, 240)
    source['P_TCHOL']['LBDTCSI'] = rng.uniform(3, 7, 240)
    source['P_GHB']['LBXGH'] = np.where(source['P_GLU']['LBDGLUSI'] > 6, 7, 5)
    frame, audit = cohort(source)
    from medpredict_ml.classification.data import split
    frame['split'] = split(frame, 2026)
    run = tmp_path/'run'
    run.mkdir()
    frame.to_csv(run/'dataset.csv', index=False)
    cfg = {'seed': 2026, 'cv_folds': 2, 'logistic_c': [1], 'hgb_learning_rates': [.1], 'hgb_leaf_nodes': [7],
        'mlp_hidden': [[4]], 'mlp_seeds': [2026], 'max_epochs': 2, 'patience': 1, 'batch_size': 128,
        'wall_clock_seconds': 120, 'simplicity_margin_ap': .01, 'bootstrap_replicates': 5}
    write_json(run/'protocol.json', {'config': cfg, 'sources': fingerprints(), 'dataset_sha256': sha256(run/'dataset.csv')})
    write_json(run/'cohort_audit.json', audit)
    train(run)
    assert read_json(run/'selection.json')['test_used_for_selection'] is False
    with pytest.raises(RuntimeError, match='already frozen'):
        train(run)
    calibrate(run)
    evaluate(run)
    assert read_json(run/'test_evaluation.json')['clinical_threshold'] is None
    with pytest.raises(RuntimeError, match='already opened'):
        train(run)
    with pytest.raises(RuntimeError, match='already opened'):
        evaluate(run)
    target = tmp_path/'bundle'
    export(run, target)
    result = Bundle(target)
    assert np.isfinite(result.probabilities([[35, 0, 25, 5, 5.5]])).all()
    with pytest.raises(ValueError, match='overwrite'):
        export(run, target)
    with (run/'dataset.csv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='changed'):
        dataset(run)
