# Optuna + LCBAMv2 TDD evidence

> Post-audit correction (2026-09-01): the real `lcbam_optuna_smoke_20260901_v2` CUDA run predates the final `definition_identities` study fingerprint. It validates the earlier end-to-end training/persistence path, while the final fingerprint behavior is validated by automated tests only and still needs a new real CUDA smoke. The earlier `lcbam_optuna_smoke_20260901` run remains invalid because of its relative output path even though its preserved machine-generated CSV/SQLite fields say `COMPLETE`/best; colocated `INVALID_RUN.md` files carry the research-validity classification.

## Source and scope

No separate `*.plan.md` was supplied. The journeys below were derived from the
Optuna/TPE structure-search request and the repository's `AGENTS.md`. No
checkpoint commits were created during this working-tree task; the RED and
GREEN checkpoints are preserved here as command evidence.

## User journeys

- As an experimenter, I can search only structure choices implemented by the
  real LCBAMv2 and YOLOv8 graph, so every sampled trial is meaningful.
- As an experimenter, I can resume a seeded TPE study without overwriting trial
  outputs, and can inspect CSV and best-parameter YAML artifacts.
- As a researcher, I can trust that the objective selects on validation
  mAP50-95 and never evaluates the test split during tuning.
- As a maintainer, I can still build and forward the existing P5 model with its
  original default behavior, while P3 and P4 are explicit verified variants.

## RED evidence

Initial feature RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_lcbam_v2.py tests\test_yolov8_lcbam_integration.py tests\test_train_smoke.py tests\test_optimize_lcbam.py -v
collected 25 items / 1 error
ImportError: cannot import name 'optimize_lcbam' from 'src'
```

This was an intended compile-time RED: the new tests referenced the missing
independent optimization module before any production implementation existed.

Data-guard normalization RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_optimize_lcbam.py::test_data_guard_requires_distinct_nonempty_train_val_test -v
1 failed, 4 passed
Failed: DID NOT RAISE ValueError
```

The failing case used `val: images/val` and `test: ./images/val`, proving that
lexical aliases were not yet recognized as the same split path.

Deterministic CUDA backend RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_optimize_lcbam.py::test_reproducibility_seed_covers_python_numpy_and_torch -v
1 failed; torch.backends.cudnn.deterministic was False
```

After setting deterministic mode and disabling cuDNN benchmark in the shared
seed helper, the same target returned `1 passed in 3.67s`.

Trainer-boundary pretrained-model RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_train_smoke.py::test_lcbamv2_smoke_preserves_initialized_model_at_trainer_boundary -v
1 failed; the trainer captured no transferred custom-model weights
```

This exposed that Ultralytics rebuilt from YAML at `Model.train(...)`, dropping
the custom model prepared before training. The project now preserves that model
at the trainer boundary; the regression test passes.

Absolute trial-output RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_optimize_lcbam.py::test_objective_passes_absolute_project_path_to_ultralytics -v
1 failed; project was a relative path
```

The first exploratory GPU smoke confirmed that a relative `project` could be
nested under Ultralytics' default `runs/detect`. The objective now passes an
absolute project path, and the same test passes.

Resume-definition fingerprint RED:

```text
.venv\Scripts\python.exe -m pytest tests\test_optimize_lcbam.py::test_sqlite_resume_rejects_changed_model_definition -v
1 failed; changed YAML content at the same path was accepted
```

The protocol now records SHA-256 identities for the data/model YAMLs, key model
and optimizer sources, and the pretrained weight. The changed-definition test
and the local-weight identity test both pass.

## GREEN evidence

The initial target after the minimal implementation returned `44 passed in
7.69s`. After the architecture/audit corrections and integration cases, the
final validations were:

```text
.venv\Scripts\python.exe -m pytest -v
63 passed in 7.71s
```

```text
.venv\Scripts\python.exe -m pytest --cov=scripts --cov=src --cov-report=term-missing -q
63 passed in 9.61s
TOTAL 545 statements, 58 missed, 89% coverage
src/optimize_lcbam.py: 88% coverage
```

```text
.venv\Scripts\python.exe -m src.optimize_lcbam --help
exit code 0; required CLI arguments and --n-trials alias displayed
```

`git diff --check` also completed without whitespace errors; Git only reported
the repository's normal LF-to-CRLF working-copy warnings.

## Test specification

| # | What is guaranteed | Test target | Type | Result |
|---|---|---|---|---|
| 1 | Channel and spatial kernels 3/5/7 preserve shape; invalid spatial kernels fail | `tests/test_lcbam_v2.py` | unit | PASS |
| 2 | P3, P4, and P5 YAMLs each contain one attention layer and complete a YOLO forward | `test_every_supported_insertion_position_builds_and_forwards` | integration | PASS |
| 3 | Trial snapshots use only legal search values and build kernel variants in YOLO | `tests/test_optimize_lcbam.py` config/build cases | integration | PASS |
| 4 | Pretrained transfer detects the single attention index instead of assuming P5/index 10 | `test_lcbamv2_pretrained_migration_detects_non_p5_attention_index` | integration | PASS |
| 5 | Data configs require distinct, nonempty train/val/test paths, including normalized aliases | `test_data_guard_requires_distinct_nonempty_train_val_test` | unit | PASS |
| 6 | Objective returns validation mAP50-95, records all four validation metrics, and never calls test evaluation | `test_objective_returns_validation_map_and_never_calls_test` | unit/integration | PASS |
| 7 | Missing or nonfinite validation metrics fail loudly | `test_objective_rejects_missing_or_nonfinite_validation_metric` | unit | PASS |
| 8 | Trial directories are unique and a collision fails instead of overwriting | `test_trial_directories_are_isolated_and_collisions_fail_closed` | integration | PASS |
| 9 | Python, NumPy, and PyTorch reproduce the same random sequence from the fixed training seed | `test_reproducibility_seed_covers_python_numpy_and_torch` | unit | PASS |
| 10 | SQLite studies resume and persist trial CSV plus best-parameter YAML | `test_sqlite_study_resumes_and_persists_csv_and_best_yaml` | integration | PASS |
| 11 | An explicitly classified training failure marks one trial FAIL and does not stop the next trial | `test_run_search_marks_expected_training_failure_and_continues` | integration | PASS |
| 12 | Existing smoke defaults and existing P5 model behavior remain intact | existing smoke/integration regression tests | regression | PASS |
| 13 | The transferred custom model survives the Ultralytics trainer boundary | `test_lcbamv2_smoke_preserves_initialized_model_at_trainer_boundary` | regression | PASS |
| 14 | Resume rejects changed fixed conditions/search protocol | `test_sqlite_resume_rejects_changed_fixed_conditions` | integration | PASS |
| 15 | Objective gives Ultralytics an absolute project path | `test_objective_passes_absolute_project_path_to_ultralytics` | regression | PASS |
| 16 | Resume rejects changed model-definition content at the same path | `test_sqlite_resume_rejects_changed_model_definition` | integration | PASS |
| 17 | Pretrained weight must exist before its identity is recorded | `test_study_requires_pretrained_weight_to_exist_before_protocol_is_recorded` | unit | PASS |

## Real GPU smoke evidence

The latest usable GPU-chain smoke used the real fixed RSNA config, local `yolov8n.pt`,
PyTorch 2.12.1+cu132, and an NVIDIA GeForce RTX 3050 Laptop GPU:

```text
.venv\Scripts\python.exe -m src.optimize_lcbam --trials 1 --epochs 1 --imgsz 512 --batch 4 --device 0 --workers 0 --seed 0 --sampler-seed 0 --study-name lcbam_optuna_smoke_20260901_v2 --storage sqlite:///results/optuna/lcbam_optuna_smoke_20260901_v2.db --data configs/rsna.yaml --output-root runs/optuna --results-root results/optuna --weights yolov8n.pt --fraction 0.02
exit code 0
trial 0 COMPLETE: channel_kernel=5, spatial_kernel=7, insert_position=p5
validation mAP50-95=0.00011168922747715054
```

The trial trained for one epoch, validated only on `split=val`, wrote the actual
run to `runs/optuna/lcbam_optuna_smoke_20260901_v2/trial_000`, and persisted a
SQLite study, `trials.csv`, and `best_params.yaml`. These metrics only prove the
then-current end-to-end search path and are not a performance or best-structure
conclusion. Its SQLite protocol has no `definition_identities`, so current code
will reject resume and the later content-fingerprint repair is not covered by
this CUDA run.

An earlier exploratory smoke used a relative Ultralytics project path and wrote
under `runs/detect/runs/optuna/...`; that attempt is classified as invalid for
trial-output isolation. It directly motivated the absolute-path regression test
and was not used as the final smoke evidence.

## Coverage and known gaps

- Combined source/script coverage is 89%, above the required 80%; the new
  optimization module is 88% covered.
- One real 1-trial/1-epoch/2%-fraction CUDA smoke was completed. No formal search
  or long training was run.
- No formal 20-30-trial search, best-model claim, multi-seed validation, or test
  evaluation was performed.
- Epoch-level pruning is intentionally not implemented because the current
  Ultralytics integration does not expose it cheaply without widening scope.
