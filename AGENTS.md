# MedPredict ML-AI

- Python package `src/medpredict_ml`; dependencies are pinned in `pyproject.toml` and lock files.
- Existing SBP/M2 code is retained as a separate research regression task. D1 is a single-measurement HbA1c screening criterion, not confirmed diabetes or future risk.
- Read `.agents/skills/medpredict-ml-engineering/SKILL.md` for ML work and `docs/planning/diagnostic-ml-plan-2026-10-26-kk.md` for the execution plan. Paths in skills are relative to this standalone ML repository; there is no nested `ml/` directory here.
- Raw and processed participants, identifiers, split manifests and per-person predictions stay ignored. Publish aggregate evidence only. Never log medical request values.
- Keep participants disjoint; fit preprocessing inside train folds; lock selection/calibration before final evaluation. Never reopen a viewed test to select a new configuration.
- New tasks use independent run paths. Never overwrite the existing experiment or weaken its test lock.
- API errors must be safe, unit/type checks strict, and unavailable inference never falls back to demo.
- Use CPU first and preserve pinned sklearn 1.7.2 despite external skills targeting newer versions. No cloud jobs or patient data uploads.
- `python -m pytest` runs unit/integration fixtures. D1 commands and serving setup are documented in README.md. Run actual pipeline checks for changes to training and export; report retained evidence separately from checks run now.
- Do not claim clinical validation or Kazakhstan applicability from internal NHANES evaluation. Clinical decisions need independent reference-standard data and review.
