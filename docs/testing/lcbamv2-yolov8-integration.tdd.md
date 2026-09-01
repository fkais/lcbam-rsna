# LCBAMv2 YOLOv8 integration — TDD evidence

> Historical evidence snapshot (2026-08-30). The environment and known gaps below describe that task at that time and are not the current project state. Read `PROJECT_CONTEXT.md` and the latest `PROJECT_LOG.md` entry for current facts.

## Source and journey

No plan file was supplied. The validation journey was derived from the integration task: as an experiment maintainer, I want the project model YAML to be parsed by Ultralytics into a YOLOv8 model that actually contains LCBAMv2 and completes a forward pass.

## RED / GREEN evidence

| Stage | Command | Result | Guarantee |
|---|---|---|---|
| RED | `python -m pytest -v -s -p no:cacheprovider tests/test_yolov8_lcbam_integration.py` | Collection failed because `build_lcbamv2_yolov8` did not exist | The test exposed the missing project-level YOLO integration entry point. |
| GREEN | Same targeted command after the minimal implementation | `2 passed` | Ultralytics builds the YAML with one LCBAMv2 layer and completes a detection forward pass. |
| Suite | `python -m pytest -v` | `6 passed in 2.06s` | Existing module behavior and the new integration behavior pass together. |
| Coverage | `python -m pytest --cov=src --cov-report=term-missing -q` | `94%`, `6 passed in 2.98s` | The tested source package exceeds the 80% workflow threshold. |

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | Project registration allows Ultralytics to parse the custom module from YAML, and the built network contains exactly one LCBAMv2 instance. | `tests/test_yolov8_lcbam_integration.py::test_yolov8_yaml_builds_model_with_lcbamv2` | Integration | PASS |
| 2 | The built detector accepts a `(1, 3, 64, 64)` tensor, produces finite one-class detection predictions, and exposes three output feature scales. | `tests/test_yolov8_lcbam_integration.py::test_yolov8_lcbamv2_complete_forward` | Integration | PASS |

## Known gaps

Smoke training was not run because the configured dataset root does not exist and the installed PyTorch build has no CUDA support. Formal training, pretrained-weight transfer, dataset integrity, multi-seed reproducibility, and efficiency benchmarking remain unverified.

## Checkpoints

- RED: `fd1a914 test: add YOLOv8 LCBAMv2 integration reproducer`
- GREEN: `3715657 feat: integrate LCBAMv2 with YOLOv8 parser`
