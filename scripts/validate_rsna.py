"""Validate an RSNA dataset arranged in Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml


# Kept in sync with the formats accepted by the pinned Ultralytics release.
IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPLITS = ("train", "val", "test")


class DatasetValidationError(RuntimeError):
    """Raised when the dataset layout or configuration cannot be inspected."""


@dataclass
class SplitReport:
    directory_files: int = 0
    images: int = 0
    ignored_image_files: int = 0
    labels: int = 0
    unlabeled_images: int = 0
    orphan_labels: int = 0
    empty_labels: int = 0
    invalid_label_rows: int = 0


@dataclass
class DatasetReport:
    root: Path
    nc: int
    names: list[str]
    splits: dict[str, SplitReport] = field(default_factory=dict)
    class_distribution: dict[int, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(
            split.orphan_labels or split.invalid_label_rows
            for split in self.splits.values()
        )


def _relative_stem(path: Path, parent: Path) -> Path:
    return path.relative_to(parent).with_suffix("")


def _validate_row(
    row: str,
    *,
    nc: int,
) -> tuple[bool, int | None, str | None]:
    fields = row.split()
    if len(fields) != 5:
        return False, None, "expected 5 fields"

    try:
        values = [float(value) for value in fields]
    except ValueError:
        return False, None, "contains a non-numeric value"

    if not all(math.isfinite(value) for value in values):
        return False, None, "contains a non-finite value"

    class_value, x_center, y_center, width, height = values
    if not class_value.is_integer():
        return False, None, "class id is not an integer"

    class_id = int(class_value)
    if not 0 <= class_id < nc:
        return False, None, f"class id {class_id} is outside [0, {nc - 1}]"
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return False, None, "box center is outside [0, 1]"
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False, None, "box width/height must be in (0, 1]"
    return True, class_id, None


def validate_dataset(
    root: str | Path,
    *,
    nc: int,
    names: Sequence[str] | None = None,
) -> DatasetReport:
    """Inspect all required splits without treating missing label files as errors."""
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise DatasetValidationError(f"dataset root does not exist: {dataset_root}")
    if nc < 1:
        raise DatasetValidationError(f"nc must be positive, got {nc}")

    required = [
        dataset_root / kind / split
        for split in SPLITS
        for kind in ("images", "labels")
    ]
    missing = [path.relative_to(dataset_root).as_posix() for path in required if not path.is_dir()]
    if missing:
        raise DatasetValidationError("missing required directories: " + ", ".join(missing))

    report = DatasetReport(root=dataset_root, nc=nc, names=list(names or []))
    classes: Counter[int] = Counter()

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        directory_files = sorted(path for path in image_dir.rglob("*") if path.is_file())
        images = [path for path in directory_files if path.suffix.lower() in IMAGE_SUFFIXES]
        labels = sorted(
            path for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"
        )
        image_keys = {_relative_stem(path, image_dir) for path in images}
        label_keys = {_relative_stem(path, label_dir) for path in labels}

        split_report = SplitReport(
            directory_files=len(directory_files),
            images=len(images),
            ignored_image_files=len(directory_files) - len(images),
            labels=len(labels),
            unlabeled_images=sum(
                1 for image in images if _relative_stem(image, image_dir) not in label_keys
            ),
            orphan_labels=sum(
                1 for label in labels if _relative_stem(label, label_dir) not in image_keys
            ),
        )
        for label in labels:
            if _relative_stem(label, label_dir) not in image_keys and len(report.issues) < 20:
                relative = label.relative_to(dataset_root).as_posix()
                report.issues.append(f"{relative}: no matching supported image")

        for label in labels:
            text = label.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                split_report.empty_labels += 1
                continue
            for line_number, row in enumerate(text.splitlines(), start=1):
                if not row.strip():
                    continue
                valid, class_id, reason = _validate_row(row, nc=nc)
                if valid:
                    classes[class_id] += 1
                    continue
                split_report.invalid_label_rows += 1
                if len(report.issues) < 20:
                    relative = label.relative_to(dataset_root).as_posix()
                    report.issues.append(f"{relative}:{line_number}: {reason}")

        report.splits[split] = split_report

    report.class_distribution = dict(sorted(classes.items()))
    return report


def validate_config(config_path: str | Path) -> DatasetReport:
    """Load the dataset root, class count, and names from an Ultralytics data YAML."""
    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetValidationError(f"cannot read data YAML {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DatasetValidationError(f"data YAML must contain a mapping: {path}")

    missing_keys = [key for key in ("path", "train", "val", "test", "nc", "names") if key not in config]
    if missing_keys:
        raise DatasetValidationError("data YAML is missing keys: " + ", ".join(missing_keys))
    expected_splits = {split: f"images/{split}" for split in SPLITS}
    actual_splits = {split: str(config[split]).replace("\\", "/") for split in SPLITS}
    if actual_splits != expected_splits:
        raise DatasetValidationError(
            f"this validator expects fixed split paths {expected_splits}, got {actual_splits}"
        )
    return validate_dataset(config["path"], nc=int(config["nc"]), names=config["names"])


def format_report(report: DatasetReport) -> str:
    lines = [
        f"Dataset root: {report.root}",
        f"Classes: nc={report.nc}, names={report.names}",
        "",
        "split | directory files | supported images | ignored files | labels | "
        "unlabeled images | orphan labels | empty labels | invalid rows",
        "-" * 122,
    ]
    for name, split in report.splits.items():
        lines.append(
            f"{name:<5} | {split.directory_files:>15} | {split.images:>16} | "
            f"{split.ignored_image_files:>13} | {split.labels:>6} | "
            f"{split.unlabeled_images:>16} | {split.orphan_labels:>13} | "
            f"{split.empty_labels:>12} | {split.invalid_label_rows:>12}"
        )
    lines.extend(["", f"Class distribution (valid rows): {report.class_distribution}"])
    if report.issues:
        lines.append("Issue examples (up to 20):")
        lines.extend(f"  - {issue}" for issue in report.issues)
    lines.append(
        "Result: FAIL (orphan labels or invalid rows found)"
        if report.has_errors
        else "Result: PASS"
    )
    lines.append(
        "Note: unlabeled supported images are reported as background candidates, not errors."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/rsna.yaml", help="Ultralytics data YAML")
    args = parser.parse_args(argv)
    try:
        report = validate_config(args.data)
    except DatasetValidationError as exc:
        parser.exit(2, f"Dataset validation could not start: {exc}\n")
    print(format_report(report))
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
