# LCBAMv2 formal structure-search freeze — TDD evidence

## Source and journeys

No separate plan file was supplied. Guarantees were derived from the formal
protocol-freeze request:

- A researcher can audit fixed RSNA splits at path, patient, and exact-content
  level without treating legitimate missing/empty negative labels as errors.
- A formal structure search can start only from a current PASS manifest and can
  use only the three real LCBAMv2 structure variables.
- Every trial receives the same explicit training recipe, validation-only
  objective, initialization, seed policy, and output isolation.
- Resume preserves a total 20-trial budget and rejects protocol/data drift.

No checkpoint commits were created because this task began in a heavily dirty
working tree containing the maintainer's uncommitted LCBAMv2 work; RED/GREEN
evidence is preserved here without staging or rewriting those changes.

## RED and GREEN evidence

| behavior | RED evidence | GREEN evidence |
|---|---|---|
| split audit module | Target collection failed with `ModuleNotFoundError: scripts.audit_rsna_split` | Seven audit tests pass |
| direct audit CLI | First real command failed because direct script execution could not resolve `scripts.validate_rsna` | Same command returned real-data `PASS` and wrote JSON/CSV |
| path/patient/content overlap and negative handling | Tests referenced missing audit API and reports | Cross-content, cross-patient, alias, no-txt negative, artifact, and within-split duplicate cases pass |
| frozen objective arguments | Assertions initially expected newly frozen explicit optimizer/schedule fields | Objective test verifies AdamW/lr/warmup/patience/AMP/cache/augmentation/resume and val-only use |
| formal manifest guard | Formal acceptance initially lacked a current dataset-backed manifest | Passing manifest is accepted; post-audit image mutation is rejected |
| total resume budget | Original implementation would append the requested trial count on each call | Resume test proves a two-trial total remains two, not four |

## Test specification

| guarantee | test target | type | result |
|---|---|---|---|
| Cross-split exact content overlap fails | `test_audit_fails_on_cross_split_content_overlap` | integration | PASS |
| Cross-split patient ID overlap fails | `test_audit_fails_on_patient_overlap_even_when_content_differs` | integration | PASS |
| Normalized split aliases fail | `test_audit_rejects_train_val_test_path_aliases` | unit | PASS |
| Missing txt is a legal negative/background candidate | `test_unlabeled_negative_image_is_counted_but_does_not_fail` | unit | PASS |
| Within-train duplicate is reported without false leakage failure | `test_within_split_duplicate_is_reported_without_leakage_failure` | unit | PASS |
| Objective uses only validation mAP50-95 and explicit frozen arguments | `test_objective_returns_validation_map_and_never_calls_test` | integration | PASS |
| P3/P4/P5 still build and forward | `test_every_supported_insertion_position_builds_and_forwards` | integration | PASS |
| Formal startup requires a current PASS audit | formal precondition tests in `tests/test_optimize_lcbam.py` | integration | PASS |
| Resume honors the total trial budget | `test_resume_treats_trials_as_total_budget` | integration | PASS |
| Existing smoke CLI remains intact | `tests/test_train_smoke.py` | regression | PASS |

## Validation evidence

Pre-final suite:

```text
.venv\Scripts\python.exe -m pytest -v
73 passed in 9.03s
```

Coverage:

```text
.venv\Scripts\python.exe -m pytest --cov=scripts --cov=src --cov-report=term-missing -q
73 passed in 10.27s
TOTAL 86%; scripts/audit_rsna_split.py 82%; src/optimize_lcbam.py 86%
```

Real data audit:

```text
RSNA split audit: PASS
train/val/test images: 18677 / 5336 / 2670
cross-split basenames/patients/content groups: 0 / 0 / 0
```

Real formal preflight, without study creation or trials:

```text
FORMAL_PREFLIGHT=PASS
PROTOCOL_SHA256=f2e3afb93422ca821206379d7b57897120af6f8ee86cd34d7e3efe8370acc611
```

Final suite after the post-audit mutation regression test:

```text
.venv\Scripts\python.exe -m pytest -v
74 passed in 9.31s
```

## Known gaps

- Exact byte hashes do not detect perceptual near-duplicates.
- Patient identity is the RSNA challenge `patientId`; DICOM Study/SOP UIDs were
  not parsed because pydicom is not installed.
- TPE sampler RNG state is not stored in SQLite, so a process restart preserves
  study history and protocol but is not claimed to reproduce the uninterrupted
  suggestion sequence exactly.
- No training, formal Optuna study, test evaluation, or performance claim was
  produced in this task.
