"""Reproducible Optuna/TPE structure search for YOLOv8n + LCBAMv2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import torch
import ultralytics
from ultralytics import YOLO
import yaml

from src.models import (
    build_lcbamv2_yolov8,
    load_lcbamv2_pretrained,
    preserve_lcbamv2_model_for_training,
)


ROOT = Path(__file__).resolve().parents[1]
POSITION_CONFIGS = {
    "p3": ROOT / "configs" / "models" / "yolov8n-lcbamv2-p3.yaml",
    "p4": ROOT / "configs" / "models" / "yolov8n-lcbamv2-p4.yaml",
    "p5": ROOT / "configs" / "models" / "yolov8n-lcbamv2.yaml",
}
CHANNEL_KERNEL_CHOICES = (3, 5, 7)
SPATIAL_KERNEL_CHOICES = (3, 5, 7)
INSERT_POSITION_CHOICES = tuple(POSITION_CONFIGS)
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
STUDY_PROTOCOL_ATTR = "lcbam_search_protocol_v1"
EXPECTED_TRANSFERRED_ITEMS = 319
EXPECTED_TARGET_ITEMS = 358
EXPECTED_NEW_LAYER_ITEMS = 3
SAMPLER_SETTINGS = {
    "name": "TPESampler",
    "n_startup_trials": 10,
    "n_ei_candidates": 24,
    "multivariate": False,
    "group": False,
    "constant_liar": False,
}
FORMAL_FROZEN_VALUES = {
    "trial_budget": 20,
    "epochs": 20,
    "imgsz": 512,
    "batch": 4,
    "device": "0",
    "workers": 0,
    "seed": 0,
    "sampler_seed": 0,
    "fraction": 1.0,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "patience": 0,
    "deterministic": True,
    "amp": True,
    "cache": False,
}


@dataclass(frozen=True)
class LCBAMSearchParams:
    channel_kernel: int
    spatial_kernel: int
    insert_position: str


@dataclass(frozen=True)
class SearchSettings:
    data: Path
    epochs: int
    imgsz: int
    batch: int
    device: str
    workers: int
    seed: int
    study_name: str
    storage: str | None
    sampler_seed: int
    output_root: Path
    results_root: Path
    weights: str | None
    fraction: float = 1.0
    trial_budget: int = 20
    formal: bool = False
    split_audit_report: Path | None = None
    optimizer: str = "AdamW"
    lr0: float = 0.002
    lrf: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.0
    deterministic: bool = True
    amp: bool = True
    patience: int = 0
    cache: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "results_root", Path(self.results_root))
        if self.split_audit_report is not None:
            object.__setattr__(self, "split_audit_report", Path(self.split_audit_report))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.study_name):
            raise ValueError("study_name must be a filesystem-safe name")
        if self.epochs < 1 or self.imgsz < 1 or self.batch < 1:
            raise ValueError("epochs, imgsz, and batch must be positive")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if self.trial_budget < 1:
            raise ValueError("trial_budget must be positive")


class TrialTrainingError(RuntimeError):
    """An expected infrastructure failure that should fail only one trial."""


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and every visible CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_search_params(trial: optuna.Trial) -> LCBAMSearchParams:
    """Sample only the three structure choices implemented by LCBAMv2 YAMLs."""
    return LCBAMSearchParams(
        channel_kernel=trial.suggest_categorical(
            "channel_kernel", CHANNEL_KERNEL_CHOICES
        ),
        spatial_kernel=trial.suggest_categorical(
            "spatial_kernel", SPATIAL_KERNEL_CHOICES
        ),
        insert_position=trial.suggest_categorical(
            "insert_position", INSERT_POSITION_CHOICES
        ),
    )


def validate_data_config(path: str | Path) -> Path:
    """Fail closed unless train, val, and test are explicit distinct paths."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"data config does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("data config must contain a YAML mapping")

    dataset_root_value = data.get("path", config_path.parent)
    if not isinstance(dataset_root_value, (str, Path)):
        raise ValueError("data config path must be a filesystem path")
    dataset_root = Path(dataset_root_value)
    if not dataset_root.is_absolute():
        dataset_root = config_path.parent / dataset_root

    split_paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data config requires a nonempty {split} path")
        split_path = Path(value.strip())
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        split_paths[split] = str(split_path.resolve(strict=False)).casefold()
    if len(set(split_paths.values())) != 3:
        raise ValueError("train, val, and test paths must be distinct")
    return config_path.resolve()


def _attention_layers(config: Mapping[str, Any]) -> list[list[Any]]:
    return [
        layer
        for section in ("backbone", "head")
        for layer in config.get(section, [])
        if isinstance(layer, list) and len(layer) >= 4 and layer[2] == "LCBAMv2"
    ]


def write_trial_model_config(
    params: LCBAMSearchParams,
    trial_dir: str | Path,
) -> Path:
    """Write one trial-local snapshot derived from a verified static position YAML."""
    if params.channel_kernel not in CHANNEL_KERNEL_CHOICES:
        raise ValueError(f"unsupported channel_kernel: {params.channel_kernel}")
    if params.spatial_kernel not in SPATIAL_KERNEL_CHOICES:
        raise ValueError(f"unsupported spatial_kernel: {params.spatial_kernel}")
    try:
        source_path = POSITION_CONFIGS[params.insert_position]
    except KeyError as error:
        raise ValueError(f"unsupported insert_position: {params.insert_position}") from error

    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    layers = _attention_layers(config)
    if len(layers) != 1:
        raise ValueError(
            f"{source_path} must define exactly one LCBAMv2 layer; found {len(layers)}"
        )
    layers[0][3] = [params.channel_kernel, 3, 0.0, params.spatial_kernel]
    snapshot = Path(trial_dir) / "model_config.yaml"
    snapshot.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return snapshot


def build_trial_model(model_config: str | Path, weights: str | None) -> YOLO:
    """Build a trial model and optionally transfer compatible YOLOv8n weights."""
    target = build_lcbamv2_yolov8(model_config)
    if weights:
        report = load_lcbamv2_pretrained(target, YOLO(weights))
        if (
            report.transferred != EXPECTED_TRANSFERRED_ITEMS
            or report.target_items != EXPECTED_TARGET_ITEMS
            or len(report.new_layer_keys) != EXPECTED_NEW_LAYER_ITEMS
        ):
            raise ValueError(
                "pretrained source is not the frozen YOLOv8n transfer layout: "
                f"transferred={report.transferred}/{report.target_items}, "
                f"new_layer_items={len(report.new_layer_keys)}"
            )
        target.lcbam_transfer_report = report
    return preserve_lcbamv2_model_for_training(target)


def _extract_validation_metrics(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", None)
    if not isinstance(values, Mapping):
        raise ValueError("training result has no validation results_dict")
    metrics: dict[str, float] = {}
    for name, key in METRIC_KEYS.items():
        if key not in values:
            raise ValueError(f"validation metric is missing: {key}")
        value = float(values[key])
        if not math.isfinite(value):
            raise TrialTrainingError(
                f"validation metric must be finite: {key}={value}"
            )
        metrics[name] = value
    return metrics


def _reserve_trial_dir(settings: SearchSettings, trial_number: int) -> Path:
    study_dir = settings.output_root / settings.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = study_dir / f"trial_{trial_number:03d}"
    trial_dir.mkdir(exist_ok=False)
    return trial_dir


class LCBAMObjective:
    """Callable Optuna objective that evaluates validation mAP50-95 only."""

    def __init__(self, settings: SearchSettings):
        self.settings = settings
        self.data_config = validate_data_config(settings.data)

    def __call__(self, trial: optuna.Trial) -> float:
        params = sample_search_params(trial)
        set_reproducible_seed(self.settings.seed)
        trial_dir = _reserve_trial_dir(self.settings, trial.number)
        trial.set_user_attr("output_dir", str(trial_dir.resolve()))
        snapshot = write_trial_model_config(params, trial_dir)
        model = None

        def record_failure(error: BaseException) -> None:
            message = f"{type(error).__name__}: {error}"
            trial.set_user_attr("error", message)
            (trial_dir / "trial_error.txt").write_text(
                message + "\n", encoding="utf-8"
            )

        try:
            model = build_trial_model(snapshot, self.settings.weights)
            report = getattr(model, "lcbam_transfer_report", None)
            if report is not None:
                trial.set_user_attr("transferred_items", report.transferred)
                trial.set_user_attr("target_items", report.target_items)
            result = model.train(
                data=str(self.data_config),
                epochs=self.settings.epochs,
                imgsz=self.settings.imgsz,
                batch=self.settings.batch,
                device=self.settings.device,
                workers=self.settings.workers,
                seed=self.settings.seed,
                deterministic=self.settings.deterministic,
                optimizer=self.settings.optimizer,
                lr0=self.settings.lr0,
                lrf=self.settings.lrf,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weight_decay,
                warmup_epochs=self.settings.warmup_epochs,
                warmup_momentum=self.settings.warmup_momentum,
                warmup_bias_lr=self.settings.warmup_bias_lr,
                patience=self.settings.patience,
                amp=self.settings.amp,
                cache=self.settings.cache,
                fraction=self.settings.fraction,
                val=True,
                split="val",
                pretrained=True,
                resume=False,
                nbs=64,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                close_mosaic=10,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                bgr=0.0,
                mosaic=1.0,
                mixup=0.0,
                cutmix=0.0,
                copy_paste=0.0,
                copy_paste_mode="flip",
                auto_augment="randaugment",
                erasing=0.4,
                rect=False,
                cos_lr=False,
                multi_scale=0.0,
                single_cls=False,
                dropout=0.0,
                freeze=None,
                compile=False,
                save=True,
                save_period=-1,
                project=str(trial_dir.parent.resolve()),
                name=trial_dir.name,
                exist_ok=True,
            )
            actual_batch = getattr(
                getattr(model, "trainer", None),
                "batch_size",
                self.settings.batch,
            )
            trial.set_user_attr("actual_batch", int(actual_batch))
            if int(actual_batch) != self.settings.batch:
                raise TrialTrainingError(
                    "Ultralytics changed the batch size from "
                    f"{self.settings.batch} to {actual_batch}; trial is not comparable"
                )
            metrics = _extract_validation_metrics(result)
        except torch.cuda.OutOfMemoryError as error:
            failure = TrialTrainingError(f"CUDA out of memory: {error}")
            record_failure(failure)
            raise failure from error
        except TrialTrainingError as error:
            record_failure(error)
            raise
        except Exception as error:
            record_failure(error)
            raise
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["mAP50-95"]


def _ensure_sqlite_parent(storage: str | None) -> None:
    if not storage or not storage.startswith("sqlite:///"):
        return
    database = storage.removeprefix("sqlite:///").split("?", 1)[0]
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)


def _storage_identity(storage: str | None) -> str | None:
    if not storage or not storage.startswith("sqlite:///"):
        return storage
    database = storage.removeprefix("sqlite:///").split("?", 1)[0]
    if database == ":memory:":
        return "sqlite:///:memory:"
    return "sqlite:///" + Path(database).resolve(strict=False).as_posix()


def validate_formal_preconditions(settings: SearchSettings) -> None:
    """Fail closed unless the frozen formal protocol and split audit match."""
    if not settings.formal:
        return
    if "smoke" in settings.study_name.casefold():
        raise ValueError("formal study name must not contain 'smoke'")
    for name, expected in FORMAL_FROZEN_VALUES.items():
        actual = getattr(settings, name)
        if actual != expected:
            raise ValueError(
                f"formal frozen setting {name} must be {expected!r}, got {actual!r}; "
                "create a new study protocol version to change it"
            )
    if settings.weights is None:
        raise ValueError("formal search requires frozen pretrained weights")
    if settings.split_audit_report is None:
        raise ValueError("formal search requires a passing split audit report")
    audit_path = settings.split_audit_report.resolve()
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read split audit report {audit_path}: {error}") from error
    if audit.get("schema_version") != 1 or audit.get("status") != "PASS":
        raise ValueError("split audit report must be schema version 1 with PASS status")
    identity = audit.get("data_config", {})
    if (
        Path(identity.get("path", "")).resolve(strict=False)
        != settings.data.resolve(strict=False)
        or identity.get("sha256") != _sha256_file(settings.data.resolve())
    ):
        raise ValueError("split audit report does not match the current data config")
    manifest = audit.get("manifest", {})
    manifest_path = Path(manifest.get("path", ""))
    if (
        not manifest_path.is_file()
        or manifest.get("sha256") != _sha256_file(manifest_path)
    ):
        raise ValueError("split audit manifest is missing or its SHA-256 does not match")
    _verify_manifest_against_current_data(audit, manifest_path)


def _verify_manifest_against_current_data(
    audit: Mapping[str, Any], manifest_path: Path
) -> None:
    """Verify every frozen image path, size, and SHA-256 against current storage."""
    from scripts.validate_rsna import IMAGE_SUFFIXES

    raw_paths = audit.get("split_paths")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != {"train", "val", "test"}:
        raise ValueError("split audit report has invalid split_paths")
    split_paths = {name: Path(raw_paths[name]).resolve() for name in raw_paths}
    expected: dict[str, set[str]] = {name: set() for name in split_paths}
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split = row.get("split", "")
                relative_value = row.get("relative_path", "")
                if split not in split_paths or not relative_value:
                    raise ValueError("split audit manifest contains an invalid row")
                relative = Path(relative_value)
                path = (split_paths[split] / relative).resolve()
                try:
                    path.relative_to(split_paths[split])
                except ValueError as error:
                    raise ValueError("split audit manifest path escapes its split") from error
                if not path.is_file():
                    raise ValueError(f"split audit image is missing: {path}")
                if path.stat().st_size != int(row.get("size_bytes", "-1")):
                    raise ValueError(f"split audit image size changed: {path}")
                if row.get("sha256") != _sha256_file(path):
                    raise ValueError(f"split audit image content changed: {path}")
                expected[split].add(relative.as_posix())
    except (OSError, csv.Error, TypeError) as error:
        raise ValueError(f"cannot verify split audit manifest: {error}") from error
    for split, root in split_paths.items():
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        }
        if actual != expected[split]:
            raise ValueError(f"current {split} image membership differs from split audit manifest")


def create_study(settings: SearchSettings) -> optuna.Study:
    """Create or resume one maximize study with a seeded TPE sampler."""
    validate_formal_preconditions(settings)
    _ensure_sqlite_parent(settings.storage)
    sampler = optuna.samplers.TPESampler(
        seed=settings.sampler_seed,
        n_startup_trials=SAMPLER_SETTINGS["n_startup_trials"],
        n_ei_candidates=SAMPLER_SETTINGS["n_ei_candidates"],
        multivariate=SAMPLER_SETTINGS["multivariate"],
        group=SAMPLER_SETTINGS["group"],
        constant_liar=SAMPLER_SETTINGS["constant_liar"],
    )
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=settings.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    protocol = _study_protocol(settings)
    existing = study.user_attrs.get(STUDY_PROTOCOL_ATTR)
    if existing is None:
        if study.trials:
            raise ValueError(
                "existing study has no recorded search protocol; use a new study name"
            )
        study.set_user_attr(STUDY_PROTOCOL_ATTR, protocol)
    elif existing != protocol:
        raise ValueError(
            "existing study settings do not match this search protocol; "
            "use the original settings or a new study name"
        )
    return study


def _trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "trial_number": trial.number,
        "channel_kernel": trial.params.get("channel_kernel", ""),
        "spatial_kernel": trial.params.get("spatial_kernel", ""),
        "insert_position": trial.params.get("insert_position", ""),
        "mAP50": trial.user_attrs.get("mAP50", ""),
        "mAP50-95": trial.user_attrs.get("mAP50-95", ""),
        "precision": trial.user_attrs.get("precision", ""),
        "recall": trial.user_attrs.get("recall", ""),
        "output_dir": trial.user_attrs.get("output_dir", ""),
        "state": trial.state.name,
        "error": trial.user_attrs.get("error", ""),
    }


def _fixed_conditions(settings: SearchSettings) -> dict[str, Any]:
    return {
        "data": str(settings.data.resolve(strict=False)),
        "run_kind": "formal" if settings.formal else "smoke_or_debug",
        "trial_budget": settings.trial_budget,
        "epochs": settings.epochs,
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "device": settings.device,
        "workers": settings.workers,
        "seed": settings.seed,
        "sampler_seed": settings.sampler_seed,
        "optimizer": settings.optimizer,
        "lr0": settings.lr0,
        "lrf": settings.lrf,
        "momentum": settings.momentum,
        "weight_decay": settings.weight_decay,
        "warmup_epochs": settings.warmup_epochs,
        "warmup_momentum": settings.warmup_momentum,
        "warmup_bias_lr": settings.warmup_bias_lr,
        "patience": settings.patience,
        "deterministic": settings.deterministic,
        "amp": settings.amp,
        "cache": settings.cache,
        "augmentation": {
            "close_mosaic": 10,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "bgr": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "copy_paste_mode": "flip",
            "auto_augment": "randaugment",
            "erasing": 0.4,
        },
        "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "nbs": 64,
        "rect": False,
        "cos_lr": False,
        "multi_scale": 0.0,
        "single_cls": False,
        "dropout": 0.0,
        "freeze": None,
        "compile": False,
        "resume_training": False,
        "save": True,
        "save_period": -1,
        "validation": {"val": True, "split": "val"},
        "fraction": settings.fraction,
        "weights": _weight_identity(settings.weights),
        "split_audit": (
            _file_identity(settings.split_audit_report, label="split audit report")
            if settings.split_audit_report is not None
            else None
        ),
        "output_root": str(settings.output_root.resolve(strict=False)),
        "results_root": str(settings.results_root.resolve(strict=False)),
        "storage": _storage_identity(settings.storage),
        "dilation": 3,
        "gamma_init": 0.0,
        "num_lcbam": 1,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "ultralytics": ultralytics.__version__,
        "optuna": optuna.__version__,
        "cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str | Path, *, label: str) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _weight_identity(weights: str | None) -> dict[str, Any] | None:
    if weights is None:
        return None
    identity = _file_identity(weights, label="pretrained weight")
    return {"source": identity["path"], "sha256": identity["sha256"]}


def _definition_identities(settings: SearchSettings) -> dict[str, dict[str, str]]:
    files = {
        "data_yaml": settings.data,
        **{
            f"model_yaml_{position}": path
            for position, path in POSITION_CONFIGS.items()
        },
        "lcbam_source": ROOT / "src" / "models" / "lcbam_v2.py",
        "yolov8_integration_source": ROOT / "src" / "models" / "yolov8.py",
        "optimizer_source": Path(__file__),
    }
    return {
        name: _file_identity(path, label=name)
        for name, path in files.items()
    }


def _study_protocol(settings: SearchSettings) -> dict[str, Any]:
    return {
        "direction": "maximize",
        "objective": METRIC_KEYS["mAP50-95"],
        "sampler": {**SAMPLER_SETTINGS, "seed": settings.sampler_seed},
        "search_space": {
            "channel_kernel": list(CHANNEL_KERNEL_CHOICES),
            "spatial_kernel": list(SPATIAL_KERNEL_CHOICES),
            "insert_position": list(INSERT_POSITION_CHOICES),
        },
        "fixed_conditions": _fixed_conditions(settings),
        "definition_identities": _definition_identities(settings),
    }


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_study_results(study: optuna.Study, settings: SearchSettings) -> None:
    """Refresh the human-readable CSV and best-parameter YAML."""
    result_dir = settings.results_root / settings.study_name
    result_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_trial_row(study.trials[0]).keys()) if study.trials else [
        "trial_number",
        "channel_kernel",
        "spatial_kernel",
        "insert_position",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "output_dir",
        "state",
        "error",
    ]
    with (result_dir / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_trial_row(trial) for trial in study.trials)

    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if complete:
        best = study.best_trial
        payload = {
            "study_name": settings.study_name,
            "best_trial": best.number,
            "best_value": float(best.value),
            "best_params": dict(best.params),
            "objective": METRIC_KEYS["mAP50-95"],
            "direction": "maximize",
            "protocol_sha256": _protocol_digest(_study_protocol(settings)),
            "fixed_conditions": _fixed_conditions(settings),
        }
        (result_dir / "best_params.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def run_search(
    settings: SearchSettings, trials: int | None = None
) -> optuna.Study:
    budget = settings.trial_budget if trials is None else trials
    if budget != settings.trial_budget:
        raise ValueError("trial budget must match SearchSettings.trial_budget")
    study = create_study(settings)
    if settings.formal and any(
        trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials
    ):
        raise ValueError(
            "formal study contains a stale/running trial; resolve it explicitly before resume"
        )
    remaining = budget - len(study.trials)
    if remaining < 0:
        raise ValueError(
            f"study already contains {len(study.trials)} trials, above frozen budget {budget}"
        )
    objective = LCBAMObjective(settings)

    def persist_callback(
        current_study: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        persist_study_results(current_study, settings)

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            catch=(TrialTrainingError,),
            callbacks=[persist_callback],
        )
    persist_study_results(study, settings)
    return study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", "--n-trials", dest="trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study-name", default="lcbam_structure_smoke")
    parser.add_argument(
        "--storage",
        default="sqlite:///results/optuna/lcbam_structure_smoke.db",
    )
    parser.add_argument("--data", default="configs/rsna.yaml")
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/optuna"))
    parser.add_argument("--results-root", type=Path, default=Path("results/optuna"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--split-audit-report", type=Path)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = SearchSettings(
        data=Path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        study_name=args.study_name,
        storage=args.storage,
        sampler_seed=args.sampler_seed,
        output_root=args.output_root,
        results_root=args.results_root,
        weights=None if args.no_pretrained else args.weights,
        fraction=args.fraction,
        trial_budget=args.trials,
        formal=args.formal,
        split_audit_report=args.split_audit_report,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        deterministic=args.deterministic,
        amp=args.amp,
        patience=args.patience,
        cache=args.cache,
    )
    study = run_search(settings)
    if any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        print(f"best_trial={study.best_trial.number}")
        print(f"best_value={study.best_value}")
        print(f"best_params={study.best_params}")
    else:
        print("No trial completed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
