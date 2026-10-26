# D1 Python API contract

Status: implemented in this repository only. No frontend/Java backend contract was changed. The `/api/v2` name does not mean the other repositories have adopted it. Bind to loopback for research; no public hosting, authentication, persistence or CORS deployment is included.

## Input

`POST /api/v2/assessments`, `Content-Type: application/json`, maximum 16 KiB. See [synthetic request](../examples/d1-request.json). Exact task `D1-HBA1C-SCREEN-v1`, schema `2.0`. Unknown/duplicate keys, malformed JSON, numeric strings, booleans as numbers, nonfinite values, nonpositive BMI/labs and unknown units fail with safe 400 `INVALID_REQUEST`. Errors do not echo submitted values or untrusted field names.

Numeric units are explicit: age `years`, BMI `kg/m2`, glucose/cholesterol `mmol/L`; sex is `male`/`female`. Age is an integer. Missing values/units/context are accepted as a structured `insufficient_data` state rather than imputed. Full input JSON Schema is returned by `GET /api/v2/assessments/schema`.

Required context: fasting, same-examination panel, HbA1c not known, no reported known diabetes or glucose-lowering medication; pregnancy `not_pregnant` for female / `not_applicable` for male. Unknown pregnancy is insufficient data. Contrary context, support bounds or sex-specific age support produce `not_applicable`, not a low disease probability.

## Output

Successful assessment:

```json
{
  "status": "assessed",
  "task": "D1-HBA1C-SCREEN-v1",
  "schemaVersion": "2.0",
  "modelVersion": "D1-r2-internal-v1",
  "researchOnly": true,
  "clinicalValidation": "not_established",
  "findings": [{"condition": "single_hba1c_ge_6_5_percent", "probability": 0.0}],
  "reasons": [],
  "limitations": ["Research-only single-measurement screening criterion"]
}
```

The number above is a schema illustration, **not a measured model output**. Actual probability is finite in [0,1]. Do not render it as confirmed diabetes probability. A displayed medical diagnosis, clinical decision threshold, treatment suggestion and confidence score are absent.

Other business states have `findings=[]` and safe reason codes, with the same task/version/limitations. `insufficient_data` indicates fields to supply; `not_applicable` indicates unsupported use. `503 MODEL_UNAVAILABLE` covers absent/corrupt/incompatible bundle or invalid inference. No auto-retry/retraining/demo fallback. Responses carry `Cache-Control: no-store`.

## Readiness and trusted artifact boundary

`GET /health` is liveness only. `/ready` requires exact package versions, manifest schema, feature order/encoding, SHA-256 and synthetic reference parity. Public readiness failures do not expose exceptions/paths. Select bundle only through administrator-controlled `MEDPREDICT_D1_BUNDLE`; requests cannot upload or choose artifacts. Joblib loading executes trusted local serialization and must never accept arbitrary bytes.

## Future consumer integration

Frontend/backend changes need their own agreed contract and tests. A consumer must preserve field errors, input snapshot, task/version matching, cancellation, stale-response protection and status-specific PDF output. Keep v1 numerical pressure and v2 screening types separate. A future Java gateway must map safe validation/error envelopes and enforce a bounded timeout; it was not implemented or tested here.

Do not log payloads in reverse proxies/APM or send them to analytics. Operational logging can use status/version/timing only. Scope does not include persistent history, patient accounts, scheduled model updates or clinical deployment.
