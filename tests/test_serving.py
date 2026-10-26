import copy
import importlib.metadata
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from medpredict_ml.common import FEATURES, sha256, write_json
from medpredict_ml.classification.bundle import Bundle, UNITS
from medpredict_ml.classification.data import TASK
from medpredict_ml.classification.model import CalibratedModel
from medpredict_ml.serving import create_app


@pytest.fixture
def bundle(tmp_path):
    # Synthetic fixtures test software/parity only, never medical performance.
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 5))
    y = (x[:, 4] > 0).astype(int)
    estimator = LogisticRegression().fit(x, y)
    calibrator = LogisticRegression().fit(CalibratedModel.logits(estimator, x), y)
    model = CalibratedModel(estimator, calibrator)
    joblib.dump(model, tmp_path/'model.joblib')
    vectors = [[35, 0, 25, 5, 5.5]]
    write_json(tmp_path/'golden.json', {'features': vectors, 'probabilities': model.predict_proba(vectors)[:, 1].tolist(), 'absolute_tolerance': 1e-7})
    write_json(tmp_path/'metadata.json', {'task': TASK, 'schemaVersion': '2.0', 'modelVersion': 'synthetic-test',
        'researchOnly': True, 'clinicalValidation': 'not_established', 'clinicalThreshold': None,
        'featureOrder': FEATURES, 'units': UNITS, 'sexEncoding': {'male': 0, 'female': 1},
        'support': {'age': [18, 79], 'sex': [0, 1], 'bmi': [15, 50], 'totalCholesterol': [2, 10], 'glucose': [3, 15]},
        'ageBySex': {'male': [18, 79], 'female': [20, 44]},
        'packages': {k: importlib.metadata.version(k) for k in ['numpy', 'scikit-learn', 'torch']},
        'model_sha256': sha256(tmp_path/'model.joblib'), 'golden_sha256': sha256(tmp_path/'golden.json')})
    return tmp_path


@pytest.fixture
def payload():
    return {'task': TASK, 'schemaVersion': '2.0', 'values': {'age': 35, 'sex': 'male', 'bmi': 25., 'totalCholesterol': 5., 'glucose': 5.5},
        'units': dict(UNITS), 'context': {'fasting': True, 'sameExamination': True, 'hba1cKnown': False,
        'knownDiabetes': False, 'glucoseLoweringMedication': False, 'pregnancyStatus': 'not_applicable'}}


def test_ready_assessment_and_missing_inputs(bundle, payload):
    with TestClient(create_app(bundle)) as client:
        assert client.get('/ready').status_code == 200
        assert client.get('/api/v2/assessments/schema').json()['featureOrder'] == FEATURES
        response = client.post('/api/v2/assessments', json=payload)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        assert response.json()['status'] == 'assessed'
        direct = Bundle(bundle).probabilities([[35, 0, 25, 5, 5.5]])[0]
        assert response.json()['findings'][0]['probability'] == pytest.approx(direct, abs=1e-7)
        del payload['values']['glucose']
        result = client.post('/api/v2/assessments', json=payload).json()
        assert result['status'] == 'insufficient_data' and result['findings'] == []


@pytest.mark.parametrize('group,key,value', [('values', 'age', True), ('values', 'glucose', '5.5'),
    ('values', 'glucose', float('nan')), ('values', 'glucose', -1), ('units', 'glucose', 'mg/dL'), ('values', 'hba1c', 9), ('context', 'fasting', 'yes')])
def test_strict_invalid_input_has_safe_errors(bundle, payload, group, key, value):
    payload[group][key] = value
    with TestClient(create_app(bundle)) as client:
        response = client.post('/api/v2/assessments', content=json.dumps(payload), headers={'Content-Type': 'application/json'})
        assert response.status_code == 400
        assert response.json()['error'] == 'INVALID_REQUEST'
        assert '5.5' not in response.text and 'hba1c' not in response.text


@pytest.mark.parametrize('group,key,value', [('context', 'fasting', False), ('context', 'hba1cKnown', True),
    ('context', 'knownDiabetes', True), ('values', 'age', 80), ('context', 'pregnancyStatus', 'pregnant')])
def test_ineligible_input_abstains(bundle, payload, group, key, value):
    payload[group][key] = value
    with TestClient(create_app(bundle)) as client:
        result = client.post('/api/v2/assessments', json=payload).json()
        assert result['status'] == 'not_applicable' and result['findings'] == []


def test_duplicate_keys_unknown_task_and_unavailable_never_become_demo(bundle, payload):
    with TestClient(create_app(bundle)) as client:
        response = client.post('/api/v2/assessments', content='{"task":"a","task":"b"}', headers={'Content-Type': 'application/json'})
        assert response.status_code == 400
        payload['task'] = 'unimplemented-diagnosis'
        assert client.post('/api/v2/assessments', json=payload).status_code == 400
    with TestClient(create_app(None)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
    with (bundle/'model.joblib').open('ab') as handle:
        handle.write(b'corrupt')
    with TestClient(create_app(bundle)) as client:
        assert client.get('/ready').status_code == 503
        assert str(bundle) not in client.get('/ready').text


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_provider_output_is_unavailable(bundle, payload, monkeypatch, probability):
    with TestClient(create_app(bundle)) as client:
        monkeypatch.setattr(client.app.state.bundle, 'probabilities', lambda _: np.array([probability]))
        assert client.post('/api/v2/assessments', json=payload).status_code == 503


def test_female_age_support_is_distinct(bundle, payload):
    payload['values'].update(sex='female', age=70)
    payload['context']['pregnancyStatus'] = 'not_pregnant'
    with TestClient(create_app(bundle)) as client:
        assert client.post('/api/v2/assessments', json=payload).json()['status'] == 'not_applicable'


def test_large_payload_and_non_json_content_type_are_rejected(bundle):
    with TestClient(create_app(bundle)) as client:
        assert client.post('/api/v2/assessments', content=b'x'*16385, headers={'Content-Type': 'application/json'}).status_code == 400
        assert client.post('/api/v2/assessments', content='{}', headers={'Content-Type': 'text/plain'}).status_code == 400
