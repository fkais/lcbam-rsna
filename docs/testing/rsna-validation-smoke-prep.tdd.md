# RSNA validation and smoke-training preparation — TDD evidence

## Source and user journeys

No plan file was supplied. The guarantees were derived from the 2026-08-30 request:

1. A maintainer can validate the fixed local RSNA split and distinguish trainable images, unlabeled background candidates, empty labels, orphan labels, invalid rows, and class counts.
2. A maintainer can run a reproducible 1～3 epoch Baseline or LCBAMv2 smoke command with explicit data, CUDA device, epochs, batch, image size, and seed.
3. LCBAMv2 can reuse all shape-compatible YOLOv8n parameters despite the post-SPPF index shift, while its new parameters keep their intended initialization.

## RED and GREEN evidence

| Behavior | RED evidence | GREEN evidence | Guarantee |
|---|---|---|---|
| Dataset validation | Initial target run failed collection with `ModuleNotFoundError: scripts.validate_rsna` | `tests/test_validate_rsna.py` passes | Missing txt files are counted as candidates, while layout, orphan, empty, format, range, and class statistics are checked. |
| Smoke CLI | Initial target run failed collection because `scripts.train_smoke` did not exist | `tests/test_train_smoke.py::test_smoke_entry_forwards_all_reproducibility_arguments` passes | Every requested reproducibility argument reaches `YOLO.train`. |
| Compatible pretrained transfer | Initial target run could not import `load_lcbamv2_pretrained` | Migration and initialization test passes | Post-SPPF source keys are shifted by one; LCBAMv2 parameters are not overwritten. |
| Actionable orphan diagnostics | New assertion initially failed because only the orphan count was recorded | The same test passes after adding the relative label path | A real orphan can be located without modifying the dataset. |

## Commands and results

- Target GREEN: `python -m pytest tests/test_validate_rsna.py tests/test_train_smoke.py -v` → `4 passed`.
- Coverage: `python -m pytest --cov=scripts --cov=src --cov-report=term-missing -q` → `12 passed`, total 90%.
- Real data: `python -m scripts.validate_rsna --data configs/rsna.yaml` → completed with one orphan empty label and exit code 1; no invalid label rows.

## Known gaps

- CUDA training was deliberately not run. The current environment has CPU-only PyTorch 2.13.0 and exposes zero CUDA devices.
- Official `yolov8n.pt` is not present in the repository; the smoke command needs a local weight path or network access for Ultralytics' first download.
- The one orphan empty train label requires a data-owner decision; this task did not alter the dataset.
- A first coverage invocation with multiple exact `--cov=<module>` sources hit a Windows access violation while importing Torch. The stable package-level coverage command above then passed at 90%.
