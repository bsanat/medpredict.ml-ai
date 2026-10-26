"""Check real loopback HTTP, reference parity, readiness and operational latency."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import httpx
import numpy as np

from medpredict_ml.classification.bundle import Bundle, UNITS
from medpredict_ml.classification.data import TASK
from medpredict_ml.common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/d1'))
    parser.add_argument('--output', type=Path, default=Path('reports/d1/runtime_verification.json'))
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    payload = {'task': TASK, 'schemaVersion': '2.0', 'values': {'age': 35, 'sex': 'male', 'bmi': 25., 'totalCholesterol': 5., 'glucose': 5.5},
        'units': UNITS, 'context': {'fasting': True, 'sameExamination': True, 'hba1cKnown': False,
        'knownDiabetes': False, 'glucoseLoweringMedication': False, 'pregnancyStatus': 'not_applicable'}}
    expected = bundle.probabilities([[35, 0, 25, 5, 5.5]])[0]
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, MEDPREDICT_D1_BUNDLE=str(args.bundle.resolve()))
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'medpredict_ml.serving:app', '--host', '127.0.0.1',
        '--port', str(port), '--no-access-log', '--log-level', 'error'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError('Inference process exited before readiness')
                try:
                    if client.get('/ready').status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError('Inference readiness deadline exceeded')
            cold_ms = (time.perf_counter()-started)*1000
            durations, errors, max_difference = [], 0, 0.
            for _ in range(30):
                tick = time.perf_counter()
                response = client.post('/api/v2/assessments', json=payload)
                if response.status_code != 200 or response.json().get('status') != 'assessed':
                    errors += 1
                    continue
                assert response.headers['cache-control'] == 'no-store'
                difference = abs(response.json()['findings'][0]['probability']-expected)
                max_difference = max(max_difference, difference)
                assert difference <= 1e-7, 'HTTP/reference parity failure'
                durations.append((time.perf_counter()-tick)*1000)
            assert errors == 0 and len(durations) == 30
            missing = dict(payload, values={k:v for k,v in payload['values'].items() if k != 'glucose'})
            assert client.post('/api/v2/assessments', json=missing).json()['status'] == 'insufficient_data'
            wrong_unit = dict(payload, units=dict(UNITS, glucose='mg/dL'))
            assert client.post('/api/v2/assessments', json=wrong_unit).status_code == 400
            assert client.get('/health').status_code == 200
            assert client.get('/api/v2/assessments/schema').json()['task'] == TASK
            write_json(args.output, {'scope': 'real loopback HTTP; synthetic input; one concurrent request',
                'platform': platform.platform(), 'python': platform.python_version(), 'modelVersion': bundle.metadata['modelVersion'],
                'model_sha256': bundle.metadata['model_sha256'], 'attempts': 30, 'successes': len(durations), 'errors': errors,
                'cold_start_to_ready_ms': cold_ms, 'warm_p50_ms': float(np.median(durations)),
                'warm_p95_ms': float(np.quantile(durations, .95)), 'max_absolute_probability_difference': float(max_difference),
                'engineering_target_p95_ms': 1000, 'engineering_target_met': bool(np.quantile(durations, .95) <= 1000),
                'limitations': 'Local machine only; not a deployment SLA, load test, browser check or clinical evaluation'})
            print(json.dumps({'http_checks': 'passed', 'attempts': 30, 'errors': errors, 'p95_ms': float(np.quantile(durations, .95))}))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
