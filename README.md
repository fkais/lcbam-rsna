# LCBAM-RSNA

A reproducible experimental project for pneumonia lesion detection on the RSNA Pneumonia Detection Challenge dataset using YOLOv8n and lightweight attention mechanisms.

> **Project status:** Under reconstruction.  
> The current repository mainly contains archived legacy experiments. The training pipeline, dataset preparation scripts, and the redesigned lightweight attention module are being reorganized for reproducibility.

---

## Overview

This project investigates the effect of attention mechanisms on lightweight object detection for chest X-ray pneumonia lesion detection.

The main experimental models are:

- **YOLOv8n** — baseline detector
- **YOLOv8n + CBAM** — standard Convolutional Block Attention Module
- **YOLOv8n + LCBAMv2** — redesigned lightweight attention module

The project is based on the **RSNA Pneumonia Detection Challenge** dataset.

The current goal is not to reproduce the previous experimental pipeline exactly, but to rebuild the project with:

- consistent training configurations;
- reproducible dataset splits;
- independent model source code;
- standardized evaluation;
- multiple random seeds;
- efficiency benchmarking.

---

## Repository Status

The repository is currently being reorganized.

At this stage, the main available content is:

```text
archive/
└── legacy_experiments/
```

The files under `archive/legacy_experiments/` are preserved historical materials from earlier experiments.

They may include:

- previous model configuration files;
- legacy CBAM / LCBAM implementations;
- training arguments;
- experiment metrics;
- figures and visualization results;
- other historical experiment files.

These files are retained for reference and experiment traceability.

> **Important:** Legacy experiments should not be considered the final reproducible implementation of this project.

Some earlier experiments used inconsistent configurations, and the old LCBAM implementation does not fully match the architecture description used in the early report draft.

Therefore, legacy results will not be used directly as the final experimental conclusion.

---

## Planned Repository Structure

The project will gradually be reorganized into the following structure:

```text
lcbam-rsna/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── dataset/
│   └── models/
│
├── src/
│   ├── models/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── scripts/
│   └── prepare_rsna.py
│
├── tests/
├── results/
├── paper/
│
└── archive/
    └── legacy_experiments/
```

---

## Dataset

The project uses the **RSNA Pneumonia Detection Challenge** dataset.

The original dataset is **not included in this repository**.

The dataset has been converted into a YOLO-style object detection format consisting of:

- chest X-ray images;
- YOLO-format bounding-box labels;
- dataset configuration files.

Before the final experiments, the dataset pipeline will be checked for:

- image-label consistency;
- missing or extra annotation files;
- empty-label validity;
- fixed train / validation / test splits;
- positive-case statistics;
- bounding-box statistics.

A reproducible split manifest will also be generated.

---

## Model Design

### YOLOv8n

YOLOv8n is used as the lightweight baseline detector.

It provides a reasonable balance between:

- detection accuracy;
- model size;
- computational cost;
- deployment efficiency.

### CBAM

CBAM introduces sequential:

1. channel attention;
2. spatial attention.

It is used as the standard attention baseline.

### LCBAMv2

The original experimental LCBAM implementation is archived as a legacy version.

A new implementation, temporarily named **LCBAMv2**, is being redesigned.

The current design direction is:

- lightweight channel attention based on an ECA-style 1D convolution;
- lightweight spatial attention using a smaller dilated convolution;
- explicit channel initialization;
- no parameter creation during the first forward pass;
- full compatibility with model saving and loading;
- lower attention-module parameter cost than standard CBAM.

The final implementation will only be added after the architecture, formulas, parameter count, and source code are verified to be mutually consistent.

---

## Experimental Plan

The final comparison is planned to include:

| Model | Seeds | Epochs |
|---|---:|---:|
| YOLOv8n | 0, 1, 2 | 100 |
| YOLOv8n + CBAM | 0, 1, 2 | 100 |
| YOLOv8n + LCBAMv2 | 0, 1, 2 | 100 |

The main training settings will be kept consistent across all models.

Planned evaluation metrics include:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Parameters
- GFLOPs
- Model size
- Inference latency
- FPS
- Peak GPU memory

For multi-seed experiments, detection metrics will be reported as:

```text
mean ± standard deviation
```

---

## Reproducibility

The reconstructed version of this project will aim to provide:

- fixed dataset splits;
- consistent hyperparameters;
- explicit random seeds;
- independent model source code;
- training scripts;
- evaluation scripts;
- benchmark scripts;
- environment requirements;
- reproducible commands.

The final environment will be documented after the local and cloud GPU environments are fully verified.

---

## Legacy Experiments

Previous experiments are intentionally preserved under:

```text
archive/legacy_experiments/
```

They are useful for:

- tracking project evolution;
- comparing historical implementations;
- preserving original experiment evidence;
- identifying differences between earlier and reconstructed experiments.

However, they may contain:

- inconsistent batch sizes;
- incomplete random-seed coverage;
- legacy source-code dependencies;
- implementation/report mismatches.

For this reason, archived results should be treated as **historical references only**.

---

## Current Progress

- [x] RSNA dataset converted to YOLO format
- [x] Legacy YOLOv8n experiment completed
- [x] Legacy YOLOv8n + CBAM experiment completed
- [x] Legacy YOLOv8n + LCBAM experiment completed
- [x] Legacy experiment files archived
- [ ] Clean project structure
- [ ] LCBAMv2 architecture design
- [ ] LCBAMv2 implementation
- [ ] Unit tests
- [ ] Dataset integrity verification
- [ ] Fixed dataset split manifest
- [ ] Cloud training pipeline verification
- [ ] Multi-seed formal experiments
- [ ] Efficiency benchmark
- [ ] Technical report
- [ ] Final reproducible release

---

## Disclaimer

This repository is an academic and engineering experiment for medical image object detection.

It is **not intended for clinical diagnosis or medical decision-making**.

---

## License

The source code license will be specified before the first stable release.

The RSNA dataset is not redistributed by this repository and remains subject to its original dataset license and usage terms.
