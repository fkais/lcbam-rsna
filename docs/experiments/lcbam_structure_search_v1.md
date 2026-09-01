# LCBAMv2 structure search v1 — frozen formal protocol

Status: **FROZEN / READY FOR A SEPARATE FORMAL RUN**
Frozen on: 2026-09-01
Formal study: `lcbam_structure_search_v1`
Total budget: 20 attempted trials

## Research question

With the RSNA split, initialization, training recipe, and validation rule held
constant, which implemented LCBAMv2 channel kernel, spatial kernel, and insertion
position gives the highest validation box mAP50-95?

This is a structure-only search. Training hyperparameters are not search variables.

## Fixed dataset and split audit

`configs/rsna.yaml` resolves to `E:/paperdata/meddet/datasets/rsna`. Existing
train/val/test membership is frozen and is not regenerated.

The reproducible audit command is:

```powershell
.venv\Scripts\python.exe scripts\audit_rsna_split.py --data configs\rsna.yaml --metadata-csv E:\paperdata\meddet\raw\stage_2_train_labels.csv --patient-id-column patientId --report results\data_audits\rsna_split_audit_v1.json --manifest results\data_audits\rsna_split_manifest_v1.csv
```

The 2026-09-01 audit result is `PASS`:

| split | images | txt present | positive/nonempty | negative/empty | no-txt background | patients |
|---|---:|---:|---:|---:|---:|---:|
| train | 18,677 | 18,677 | 4,198 | 14,479 | 0 | 18,677 |
| val | 5,336 | 5,336 | 1,207 | 4,129 | 0 | 5,336 |
| test | 2,670 | 2,670 | 607 | 2,063 | 0 | 2,670 |

- Resolved split path aliases: none.
- Cross-split basename overlap: 0.
- Cross-split RSNA `patientId` overlap: 0.
- Cross-split exact SHA-256 content duplicate groups/images: 0/0.
- Every 26,683 trainable PNG is content-hashed in the manifest. Formal startup
  and resume recheck path membership, byte size, and SHA-256.
- One exact duplicate group exists only inside train: patient IDs
  `08d88a60-146c-49a7-9450-45293024cf46` and
  `8f96ca8e-ff76-41bc-bf03-da5b031cabee`. Both are legal negatives. It is a
  recorded data-quality caveat, not cross-split leakage; this task does not alter it.
- Original metadata has 26,684 patient IDs. The frozen converted dataset omits
  the known negative `6227d915-6ca0-430d-8905-34a61e29e074`. It is recorded but
  is not reinserted or randomly reassigned.

Missing label txt files are legal background/negative candidates and do not make
the audit fail. Empty txt files are legal negative labels. Path, patient, basename,
or exact-content overlap across splits makes the audit fail closed.

Limitation: the content check detects byte-identical files, not perceptual
near-duplicates, and patient identity uses the challenge's `patientId` mapping
rather than parsed DICOM Study/SOP UIDs.

## Search space

Only these variables are sampled:

| variable | choices |
|---|---|
| `channel_kernel` | `[3, 5, 7]` |
| `spatial_kernel` | `[3, 5, 7]` |
| `insert_position` | `[p3, p4, p5]` |

Fixed LCBAMv2 structure conditions are `dilation=3`, `gamma_init=0.0`, and
`num_lcbam=1`. LCBAMv2 has no `reduction_ratio`; none is searched or recorded.

## Frozen training conditions

| category | frozen value |
|---|---|
| epochs | 20 complete epochs per successful trial |
| image size | 512 |
| physical batch | 4; `nbs=64` (gradient accumulation policy fixed) |
| optimizer | explicit `AdamW`; never `auto` |
| learning rate | `lr0=0.002`, `lrf=0.01`, linear schedule (`cos_lr=false`) |
| Adam beta1 | `momentum=0.9`; beta2 is pinned implementation value 0.999 |
| weight decay | 0.0005 |
| warmup | epochs 3.0, momentum 0.8, bias LR 0.0 |
| early stopping | `patience=0` (disabled so trials receive equal epoch budgets) |
| pretrained initialization | local `yolov8n.pt`, SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`; exact transfer layout 319/358 items and 3 new LCBAMv2 items |
| training seed | 0, reset identically before every trial |
| deterministic | true; Python/NumPy/Torch/CUDA seeded, cuDNN deterministic true and benchmark false |
| AMP | true |
| workers | 0 |
| cache | false |
| fraction | 1.0 |
| device | `0`; current verified device is NVIDIA GeForce RTX 3050 Laptop GPU |
| validation | `val=true`, `split="val"` |
| objective | validation `metrics/mAP50-95(B)` |
| loss weights | box 7.5, cls 0.5, dfl 1.5 |
| other geometry | `rect=false`, `multi_scale=0.0`, `single_cls=false` |
| checkpointing | `save=true`, `save_period=-1`, training resume false |

Detection augmentation is explicitly frozen to the pinned Ultralytics 8.4.135
values: `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=0.0`,
`translate=0.1`, `scale=0.5`, `shear=0.0`, `perspective=0.0`, `flipud=0.0`,
`fliplr=0.5`, `bgr=0.0`, `mosaic=1.0`, `close_mosaic=10`, `mixup=0.0`,
`cutmix=0.0`, `copy_paste=0.0`, `copy_paste_mode="flip"`,
`auto_augment="randaugment"`, and `erasing=0.4`. The last three are retained
explicitly even when they are not active for this detection pipeline, so a future
dependency version cannot silently change the recorded argument set.

## Optuna and selection rule

- Optuna 4.9.0 `TPESampler`.
- Sampler seed 0, `n_startup_trials=10`, `n_ei_candidates=24`,
  `multivariate=false`, `group=false`, `constant_liar=false`.
- Direction: maximize.
- `n_jobs=1`; only one formal process may access this study/GPU at a time.
- `--trials 20` means a total study budget of 20 attempted trials. Resume runs
  only the remaining count and never silently extend the budget.
- The selected architecture is the COMPLETE trial with maximum validation
  mAP50-95. An exact numerical tie uses the lower Optuna trial number, matching
  `study.best_trial` behavior.
- TPE may repeat one of the 27 discrete structures. The final report must state
  both attempted-trial count and unique-structure count; 20 trials must not be
  described as 20 unique architectures.
- Failed/invalid exploratory runs and smoke studies are excluded. A formal study
  with an unresolved `RUNNING` trial fails closed instead of guessing recovery.
- Optuna does not persist the sampler RNG state in SQLite. Resume preserves all
  historical trials and the frozen seed/protocol, but the post-restart suggestion
  sequence is not claimed to equal an uninterrupted process. Restart boundaries
  must be recorded.

Each trial uses `runs/optuna/lcbam_structure_search_v1/trial_NNN`; the SQLite
database is `results/optuna/lcbam_structure_search_v1.db`; summaries are under
`results/optuna/lcbam_structure_search_v1/`. Existing trial directories fail
closed. Storage, absolute output/results roots, source/config/weight identities,
audit identity, sampler settings, and every frozen condition are part of the
study protocol fingerprint.

## Test-set and result policy

> Test set remains untouched during hyperparameter/architecture search. It will be evaluated only after the final architecture and training protocol have been selected.

The objective never loads or evaluates `split="test"`; `trials.csv` and the
study database contain only validation metrics. Smoke studies are engineering
checks and are not formal results. Failed or invalid exploratory runs are not
paper results.

After this formal study, select the architecture using validation only, freeze
the final full-training/multi-seed protocol, complete that protocol, and only
then run one final test evaluation under a separately documented command. Test
performance cannot be used to return to this study or change its search space.

Once `lcbam_structure_search_v1` starts, its space and frozen conditions cannot
change in response to intermediate results. Any necessary protocol change creates
a new versioned study and database (for example v2); it is never mixed into v1.

## Reproducibility commands

First regenerate and verify the audit artifacts with the command above. Then,
in a separate authorized task, the frozen 20-trial formal command is:

```powershell
.venv\Scripts\python.exe -m src.optimize_lcbam --formal --trials 20 --epochs 20 --imgsz 512 --batch 4 --device 0 --workers 0 --seed 0 --sampler-seed 0 --study-name lcbam_structure_search_v1 --storage sqlite:///results/optuna/lcbam_structure_search_v1.db --data configs/rsna.yaml --split-audit-report results/data_audits/rsna_split_audit_v1.json --weights yolov8n.pt --fraction 1.0 --optimizer AdamW --lr0 0.002 --lrf 0.01 --momentum 0.9 --weight-decay 0.0005 --warmup-epochs 3.0 --warmup-momentum 0.8 --warmup-bias-lr 0.0 --patience 0 --deterministic --amp --no-cache --output-root runs/optuna --results-root results/optuna
```

This command is recorded only. It was not executed during protocol freezing.
