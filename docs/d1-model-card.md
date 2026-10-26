# D1-HBA1C-SCREEN-v1 — research candidate

Date: 2026-10-27. Model version: `D1-r2-internal-v1`. Status: **research only; not promoted for clinical use; no Kazakhstan validation**.

## Target and intended use

Estimate whether one same-examination NHANES HbA1c measurement is at least 6.5%, without HbA1c as an input. This is a retrospective screening experiment. It does not establish a diagnosis, diabetes type, future disease risk, causal effects or clinical benefit. A value below the criterion does not mean healthy. Actual HbA1c, when available, should not be replaced by this estimate.

## Population and source

CDC NHANES 2017–March 2020 public-use data; unique participant joins across P_DEMO, P_BMX, P_TCHOL, P_GLU, P_GHB, P_DIQ. Positive fasting weight, age 18–79, five complete finite inputs, observed HbA1c, available survey design, explicit DIQ010=2. Male pregnancy is structurally inapplicable; female participants require observed RIDEXPRG=2, restricting their age coverage to 20–44. Unknown female status is excluded.

**Protocol revision before splitting/training:** the planning phrase "not receiving treatment" could not be verified in NHANES. DIQ050 is unasked for the undiagnosed cohort; DIQ070 may be skipped through DIQ065. Explicitly reported medication use and unknown/refused answers are excluded; documented pill skips require DIQ160=2. These skips are retained as **unassessed medication status**, never proof of no treatment. The target is not described as treatment-naive disease. See [official questionnaire/codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.htm).

After exclusions: 2,027 participants, 49 positive. Training 1,215 (29 positive), calibration 203 (5), threshold-analysis 203 (5), final test 406 (10). The five-positive partition floor is only a software feasibility gate; it is **not an adequate clinical sample-size justification**. Missing targets/features are not imputed. IDs/design/weights/HbA1c/diagnosis/medication are not predictors.

## Features and preprocessing

Order: `age`, encoded `sex` (male=0/female=1), `bmi`, `totalCholesterol`, fasting `glucose`. Age in years; BMI kg/m²; both labs mmol/L. No automatic unit guessing. Observed train bounds and sex-specific age support are in [metadata](../artifacts/d1/metadata.json); they are not medical reference intervals or a comprehensive OOD detector.

The selected estimator is StandardScaler → LogisticRegression, L2 `C=0.01`. Scaler is fitted inside train folds, selected estimator refitted on training only. Held-out sigmoid calibration is fitted on model-probability logits using 203 separate participants. Clipping to [1e-7,1−1e-7] stabilizes logits. No clinical threshold is selected. Predicted probability is not an individual confidence interval.

## Model comparison and selection

Three train CV folds, fixed seed 2026. Four C values, four HGB combinations and three MLP architectures with seeds 2026/2027/2028. MLP has train-internal early stopping and an epoch-fixed fold refit. Mean unweighted fold average precision determines selection; within 0.01 absolute AP of best, simpler family is preferred. Clinical equivalence is not implied by that margin. Primary fitting is unweighted; weighted evaluation is sensitivity analysis.

Linear C=.01 CV AP 0.5602. Best MLP configuration [32,16] CV AP 0.5282. Best HGB CV AP 0.4156. These are model-selection CV estimates, not independent generalization estimates. Candidate/fold/seed details are in [selection.json](../reports/d1/selection.json). No selected configuration/seed changed after viewing final test.

## Internal evaluation

| Final test n=406, positives=10 | Selected calibrated | Glucose-only logistic | Dummy |
|---|---:|---:|---:|
| Average precision | 0.520548 | 0.554714 | 0.024631 |
| AUROC | 0.943939 | 0.950884 | 0.500000 |
| Brier | 0.016118 | 0.015992 | 0.024024 |

The full panel **does not show added value over glucose alone**. It was not swapped for the comparator after test. Sigmoid calibration leaves ranking/AP unchanged here; improved Brier versus the raw selected model on this test does not establish reliable clinical calibration with only five calibration positives.

Conditional within-stratum masked-PSU bootstrap, 1,000 valid replicates: AP 95% interval [0.295603,0.685661], AUROC [0.902024,0.978659], Brier [0.007846,0.023921]. Paired AP difference versus glucose-only [-0.069011,-0.005616]. These intervals condition on this selected cohort/split/model; they are not full survey uncertainty, clinical superiority testing or Kazakhstan validation. Finite test-stratum resampling is not a substitute for a proper external validation design.

Female test n=137 has **zero positive cases**; female AUROC/AP are undefined, not zero or perfect. The male test n=269 has all ten positives. Age 60–79 has one positive. Tiny subgroup estimates cannot support safety/fairness claims. Survey-weighted metrics, calibration bins and all subgroup counts are in [test_evaluation.json](../reports/d1/test_evaluation.json).

## Packaging and verification

Artifact: trusted local joblib research candidate, preprocessing/model/calibration together. [Metadata](../artifacts/d1/metadata.json) records schema, dependencies, source/config/selection hashes and file hashes. [Synthetic golden cases](../artifacts/d1/golden.json) check serialization, not medical performance. Loading arbitrary uploaded joblib is forbidden; matching checksum alone is not proof of trustworthy provenance.

Same-environment fresh-workspace reproduction produced identical selected configuration, evaluation JSON and golden probabilities. Real local HTTP: 30 successful synthetic requests, zero errors, reference max probability difference 0. [Runtime report](../reports/d1/runtime_verification.json). No frontend/browser/Java integration or external validation was performed in this task.

## Release decision

Publish reproducible research code/evidence as requested. Keep default serving unavailable unless this specific candidate is explicitly configured. Do not promote for diagnosis, treatment, pregnancy, children, older unsupported populations or local clinical use. Next scientific move is improved labels/cohort and an independent protocol, not more tuning against this viewed test. A confirmed-diagnosis task requires separate reference-standard outcomes and clinical review.
