"""Audit RSNA split isolation and write a reproducible image manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:
    from scripts.validate_rsna import IMAGE_SUFFIXES, SPLITS
except ModuleNotFoundError:  # Direct ``python scripts/audit_rsna_split.py`` execution.
    from validate_rsna import IMAGE_SUFFIXES, SPLITS


SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 4 * 1024 * 1024


class SplitAuditError(RuntimeError):
    """Raised when a split audit cannot be performed safely."""


@dataclass
class ManifestRow:
    split: str
    relative_path: str
    basename: str
    patient_id: str
    size_bytes: int
    sha256: str
    label_status: str
    negative: bool
    metadata_patient_match: bool | None = None


@dataclass
class SplitSummary:
    image_count: int = 0
    label_file_count: int = 0
    labeled_images: int = 0
    unlabeled_images: int = 0
    positive_images: int = 0
    negative_images: int = 0
    empty_label_images: int = 0
    patient_count: int = 0
    duplicate_basenames: int = 0
    duplicate_content_images: int = 0
    duplicate_content_groups: int = 0
    duplicate_content_excess: int = 0


@dataclass
class SplitAuditReport:
    data_config_path: str
    data_config_sha256: str
    dataset_root: str
    split_paths: dict[str, str]
    patient_id_method: str
    metadata_csv: str | None
    metadata_patient_count: int | None
    metadata_matched_images: int | None
    metadata_unmatched_images: int | None
    metadata_missing_patient_ids: list[str]
    splits: dict[str, SplitSummary]
    status: str = "PASS"
    cross_split_basename_count: int = 0
    cross_split_patient_count: int = 0
    cross_split_content_duplicate_groups: int = 0
    cross_split_content_duplicate_images: int = 0
    hash_candidate_images: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    hash_algorithm: str = "sha256"
    hash_scope: str = "all trainable image files"
    issues: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["data_config"] = {
            "path": payload.pop("data_config_path"),
            "sha256": payload.pop("data_config_sha256"),
        }
        payload["manifest"] = {
            "path": payload.pop("manifest_path"),
            "sha256": payload.pop("manifest_sha256"),
        }
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SplitAuditError(f"cannot read data config {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise SplitAuditError(f"data config must contain a mapping: {path}")
    return value


def _resolve_split_paths(config_path: Path) -> tuple[Path, dict[str, Path]]:
    config = _load_mapping(config_path)
    root_value = config.get("path", config_path.parent)
    if not isinstance(root_value, (str, Path)):
        raise SplitAuditError("data config path must be a filesystem path")
    root = Path(root_value)
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()

    paths: dict[str, Path] = {}
    for split in SPLITS:
        value = config.get(split)
        if not isinstance(value, str) or not value.strip():
            raise SplitAuditError(f"data config requires a nonempty {split} path")
        path = Path(value.strip())
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_dir():
            raise SplitAuditError(f"split directory does not exist: {split}={path}")
        paths[split] = path

    canonical = [str(path).casefold() for path in paths.values()]
    if len(set(canonical)) != len(SPLITS):
        raise SplitAuditError("train, val, and test paths must be distinct; path alias found")
    return root, paths


def _load_metadata_patient_ids(path: Path | None, column: str) -> set[str] | None:
    if path is None:
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise SplitAuditError(
                    f"metadata CSV does not contain patient ID column {column!r}: {path}"
                )
            return {
                row[column].strip()
                for row in reader
                if row.get(column) and row[column].strip()
            }
    except OSError as error:
        raise SplitAuditError(f"cannot read metadata CSV {path}: {error}") from error


def _label_status(label_path: Path) -> str:
    if not label_path.is_file():
        return "missing"
    return "nonempty" if label_path.stat().st_size else "empty"


def _cross_split_keys(rows: Sequence[ManifestRow], attribute: str) -> set[str]:
    split_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_by_key[str(getattr(row, attribute)).casefold()].add(row.split)
    return {key for key, splits in split_by_key.items() if len(splits) > 1}


def audit_split_config(
    config_path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    patient_id_column: str = "patientId",
) -> tuple[SplitAuditReport, list[ManifestRow]]:
    """Audit path, patient/basename, and exact-content isolation across splits."""
    config = Path(config_path).resolve()
    if not config.is_file():
        raise SplitAuditError(f"data config does not exist: {config}")
    root, split_paths = _resolve_split_paths(config)
    metadata_path = Path(metadata_csv).resolve() if metadata_csv else None
    metadata_ids = _load_metadata_patient_ids(metadata_path, patient_id_column)

    rows: list[ManifestRow] = []
    image_paths: dict[tuple[str, str], Path] = {}
    summaries: dict[str, SplitSummary] = {}

    for split, image_dir in split_paths.items():
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            raise SplitAuditError(f"label directory does not exist: {label_dir}")
        label_files = [
            path
            for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".txt"
        ]
        basename_counts = Counter(path.name.casefold() for path in images)
        patient_ids: set[str] = set()
        summary = SplitSummary(
            image_count=len(images),
            label_file_count=len(label_files),
            duplicate_basenames=sum(count for count in basename_counts.values() if count > 1),
        )
        for image in images:
            relative = image.relative_to(image_dir)
            patient_id = image.stem
            patient_ids.add(patient_id.casefold())
            status = _label_status((label_dir / relative).with_suffix(".txt"))
            negative = status != "nonempty"
            summary.labeled_images += status != "missing"
            summary.unlabeled_images += status == "missing"
            summary.empty_label_images += status == "empty"
            summary.positive_images += status == "nonempty"
            summary.negative_images += negative
            row = ManifestRow(
                split=split,
                relative_path=relative.as_posix(),
                basename=image.name,
                patient_id=patient_id,
                size_bytes=image.stat().st_size,
                sha256="",
                label_status=status,
                negative=negative,
                metadata_patient_match=(
                    patient_id in metadata_ids if metadata_ids is not None else None
                ),
            )
            rows.append(row)
            image_paths[(split, row.relative_path)] = image
        summary.patient_count = len(patient_ids)
        summaries[split] = summary

    # Hash every trainable image. Besides finding exact duplicates, the complete
    # manifest can prove that the dataset is unchanged when a formal study resumes.
    size_counts = Counter(row.size_bytes for row in rows)
    for row in rows:
        row.sha256 = _sha256_file(image_paths[(row.split, row.relative_path)])

    content_groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        if row.sha256:
            content_groups[row.sha256].append(row)
    duplicate_groups = [group for group in content_groups.values() if len(group) > 1]
    for split, summary in summaries.items():
        member_counts = [sum(row.split == split for row in group) for group in duplicate_groups]
        summary.duplicate_content_groups = sum(count > 1 for count in member_counts)
        summary.duplicate_content_images = sum(count for count in member_counts if count > 1)
        summary.duplicate_content_excess = sum(count - 1 for count in member_counts if count > 1)

    cross_content = [
        group for group in duplicate_groups if len({row.split for row in group}) > 1
    ]
    cross_basenames = _cross_split_keys(rows, "basename")
    cross_patients = _cross_split_keys(rows, "patient_id")
    metadata_matches = (
        sum(row.metadata_patient_match is True for row in rows)
        if metadata_ids is not None
        else None
    )
    metadata_unmatched = (
        sum(row.metadata_patient_match is False for row in rows)
        if metadata_ids is not None
        else None
    )
    method = (
        f"image stem as RSNA patientId, verified against {patient_id_column} in metadata CSV"
        if metadata_ids is not None
        else "image stem as candidate patient ID; no metadata CSV supplied"
    )
    report = SplitAuditReport(
        data_config_path=str(config),
        data_config_sha256=_sha256_file(config),
        dataset_root=str(root),
        split_paths={name: str(path) for name, path in split_paths.items()},
        patient_id_method=method,
        metadata_csv=str(metadata_path) if metadata_path else None,
        metadata_patient_count=len(metadata_ids) if metadata_ids is not None else None,
        metadata_matched_images=metadata_matches,
        metadata_unmatched_images=metadata_unmatched,
        metadata_missing_patient_ids=(
            sorted(
                metadata_ids
                - {row.patient_id for row in rows}
            )
            if metadata_ids is not None
            else []
        ),
        splits=summaries,
        cross_split_basename_count=len(cross_basenames),
        cross_split_patient_count=len(cross_patients),
        cross_split_content_duplicate_groups=len(cross_content),
        cross_split_content_duplicate_images=sum(len(group) for group in cross_content),
        hash_candidate_images=sum(size_counts[row.size_bytes] > 1 for row in rows),
        duplicate_groups=[
            {
                "sha256": group[0].sha256,
                "cross_split": len({row.split for row in group}) > 1,
                "members": [
                    {"split": row.split, "relative_path": row.relative_path}
                    for row in group
                ],
            }
            for group in duplicate_groups
        ],
    )
    if cross_basenames:
        report.issues.append(
            f"cross-split basename overlap: {len(cross_basenames)} unique basenames"
        )
    if cross_patients:
        report.issues.append(
            f"cross-split patient overlap: {len(cross_patients)} unique patient IDs"
        )
    if cross_content:
        report.issues.append(
            f"cross-split content overlap: {len(cross_content)} SHA-256 groups, "
            f"{sum(len(group) for group in cross_content)} images"
        )
    if report.issues:
        report.status = "FAIL"
    return report, rows


def write_audit_artifacts(
    report: SplitAuditReport,
    rows: Sequence[ManifestRow],
    *,
    report_path: str | Path,
    manifest_path: str | Path,
) -> None:
    manifest = Path(manifest_path).resolve()
    result = Path(report_path).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRow.__annotations__)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    report.manifest_path = str(manifest)
    report.manifest_sha256 = _sha256_file(manifest)
    result.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_report(report: SplitAuditReport) -> str:
    lines = [
        f"RSNA split audit: {report.status}",
        f"Dataset root: {report.dataset_root}",
        f"Patient ID: {report.patient_id_method}",
        "split | images | labeled | unlabeled | positive | negative | patients | within-split duplicates",
        "-" * 105,
    ]
    for name, summary in report.splits.items():
        lines.append(
            f"{name:<5} | {summary.image_count:>6} | {summary.labeled_images:>7} | "
            f"{summary.unlabeled_images:>9} | {summary.positive_images:>8} | "
            f"{summary.negative_images:>8} | {summary.patient_count:>8} | "
            f"{summary.duplicate_content_images:>23}"
        )
    lines.extend(
        [
            f"Cross-split basenames: {report.cross_split_basename_count}",
            f"Cross-split patients: {report.cross_split_patient_count}",
            f"Cross-split content groups: {report.cross_split_content_duplicate_groups}",
        ]
    )
    lines.extend(f"ERROR: {issue}" for issue in report.issues)
    lines.append("Missing label txt files are counted as negative/background candidates, not errors.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/rsna.yaml")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--patient-id-column", default="patientId")
    parser.add_argument(
        "--report", default="results/data_audits/rsna_split_audit_v1.json"
    )
    parser.add_argument(
        "--manifest", default="results/data_audits/rsna_split_manifest_v1.csv"
    )
    args = parser.parse_args(argv)
    try:
        report, rows = audit_split_config(
            args.data,
            metadata_csv=args.metadata_csv,
            patient_id_column=args.patient_id_column,
        )
        write_audit_artifacts(
            report,
            rows,
            report_path=args.report,
            manifest_path=args.manifest,
        )
    except SplitAuditError as error:
        parser.exit(2, f"Split audit could not start: {error}\n")
    print(format_report(report))
    print(f"Report: {Path(args.report).resolve()}")
    print(f"Manifest: {Path(args.manifest).resolve()}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
