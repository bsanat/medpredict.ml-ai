---
name: medpredict-ml-engineering
description: Plan, implement and evaluate MedPredict tabular medical ML tasks with explicit labels, leakage controls, frozen tests and versioned serving contracts. Use for this project's ML data, experiments and integration work.
---

# MedPredict ML engineering

Use the project's current code and evidence before choosing a model. This skill adds project-specific decisions; it does not change the user's model, reasoning settings, permissions or requested scope.

## Orient to the task

- Read the relevant root `AGENTS.md`, then only the files needed for the change. Current CLI is `src/medpredict_ml/__main__.py`; data/training/evaluation/export live beside it. Dependencies are pinned in `pyproject.toml` and `requirements.lock`.
- For diagnostic planning, read `docs/planning/diagnostic-ml-plan-2026-10-26-kk.md`. Its new services, task IDs and commands are proposals until implemented. Existing SBP inference estimates a current measurement and supplies no disease probability.
- Distinguish local workspace, public source revision, retained reports and checks run now. Do not infer a functioning provider from HTTP `real` mode or a README.

## Before training or changing a target

Write a compact task contract: intended use, population, prediction time, independent unit, label and confirmation method, feature availability, exclusions, units, missingness, split, primary metric, comparator and decision owner. Identify unresolved facts rather than supplying guessed labels or sample counts.

- A questionnaire report, single-test criterion, diagnosis and future event are different targets. Name the output after the evidence actually available. A diagnostic model needs an appropriate reference standard; HbA1c alone does not establish diabetes type.
- If labels are generated from an input threshold, compare the explicit rule and disclose the circular target. For a screening-before-confirmatory-test task, exclude that confirmatory measurement and its derivatives from inputs.
- Patient identity belongs to joins/splits, never predictors. Repeated patients stay in one partition; longitudinal joins use information available by prediction time. Unknown labels are excluded or explicitly modeled, never converted to negative labels.
- Preserve source terms, checksums, unit mappings and participant flow. Separate sensitive rows from aggregate reports. Do not upload data, weights or private code through an optional third-party skill integration.

## Experiment decisions

- Keep existing M2 artifacts and already viewed test untouched. New tasks use separate task/run paths. Reproduction uses the implemented `--workspace` support; a new training run must not silently reopen an old test.
- Fit transformations, selection and resampling inside training folds. Select models, calibration and thresholds without final-test labels. Record every selection run, including failures; bound the search before running it.
- Compare a dummy and appropriate rule/linear/tree baselines before increasing MLP complexity. Reuse the existing PyTorch approach. Choose the simplest model meeting declared scientific and engineering requirements; a negative neural comparison is reportable.
- For classification report prevalence, sample counts, AP/AUPRC definition, AUROC, sensitivity, specificity, PPV/NPV, calibration and uncertainty. Report threshold selection and abstention coverage. A high AUROC is not calibrated probability or clinical acceptance.
- Confidence intervals respect the independent unit; NHANES analyses also account for survey design. Mark unstable subgroup or one-class results undefined rather than inventing a score.
- Additional compute requires a demonstrated need. CPU and local aggregate JSON reports are the starting point. No automatic retraining from application requests.

## Integration decisions

- Preserve v1 SBP semantics. Classification needs a separate agreed schema and tests across backend DTO/service, frontend types/parsers/hook, result, translations, fixtures and PDF. Follow the existing API coordination rule.
- Bundle preprocessing, estimator, calibration, threshold policy and schema together, with actual code/dependency/model versions and checksums. Load only trusted artifacts. Include synthetic reference cases and test served/reference parity.
- Validate types, units, categories and eligibility at trust boundaries. Missing or out-of-domain input must follow a tested policy. Unknown is not healthy. An unavailable provider never becomes demo output.
- Do not log medical values, persist them in web storage, or claim that input-domain checks detect every distribution shift. Keep cancellation, stale-response protection and field accessibility.

## Use other skills selectively

The sibling `scikit-learn` skill supplies algorithm/pipeline references. It targets 1.9.1; MedPredict is pinned to 1.7.2. Verify examples against [1.7 documentation](https://scikit-learn.org/1.7/) and do not run its installation commands or examples merely to load guidance.

The sibling `scientific-critical-thinking` skill supports evidence appraisal; predictive studies use appropriate prediction-model bias/reporting tools rather than an unrelated intervention checklist. Its optional diagram API is not needed here.

When available, use ECC `mle-workflow` for lifecycle decisions, `pytorch-patterns` for training-loop changes and `python-testing` for Python tests. They are optional supporting guidance, not permission to add infrastructure. Upstream provenance is in `docs/planning/ml-agent-skills-2026-10-26.md`.

## Deliver observable evidence

For a plan, produce named deliverables, dependencies, owners, acceptance checks and stop/go conditions. For code, run the smallest meaningful tests for the changed behavior, extending to required integration checks when warranted. Do not train for a documentation edit.

Report what changed, concrete validation, unresolved risks and the next decision. Separate saved metrics from new measurements. Do not claim that installing skills improved accuracy, reasoning quality, token use or clinical validity without an evaluation.
