import json
from pathlib import Path

import pytest
import yaml

from scripts.audit_rsna_split import (
    SplitAuditError,
    audit_split_config,
    write_audit_artifacts,
)


def _write_config(tmp_path: Path, **overrides) -> Path:
    dataset = tmp_path / "dataset"
    data = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["pneumonia"],
    }
    data.update(overrides)
    config = tmp_path / "rsna.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config


def _make_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)


def _make_clean_fixture(tmp_path: Path) -> Path:
    config = _write_config(tmp_path)
    root = tmp_path / "dataset"
    for split, stem in (("train", "patient-a"), ("val", "patient-b"), ("test", "patient-c")):
        _make_split(root, split)
        (root / "images" / split / f"{stem}.png").write_bytes(split.encode())
    return config


def test_audit_fails_on_cross_split_content_overlap(tmp_path: Path):
    config = _make_clean_fixture(tmp_path)
    root = tmp_path / "dataset"
    duplicate = b"same-pixels"
    (root / "images" / "train" / "patient-a.png").write_bytes(duplicate)
    (root / "images" / "val" / "patient-b.png").write_bytes(duplicate)

    report, _ = audit_split_config(config)

    assert report.status == "FAIL"
    assert report.cross_split_content_duplicate_groups == 1
    assert any("content" in issue.casefold() for issue in report.issues)


def test_audit_fails_on_patient_overlap_even_when_content_differs(tmp_path: Path):
    config = _make_clean_fixture(tmp_path)
    root = tmp_path / "dataset"
    (root / "images" / "val" / "patient-a.png").write_bytes(b"different")

    report, _ = audit_split_config(config)

    assert report.status == "FAIL"
    assert report.cross_split_patient_count == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"test": "images/val"},
        {"test": "./images/val"},
    ],
)
def test_audit_rejects_train_val_test_path_aliases(tmp_path: Path, overrides):
    config = _write_config(tmp_path, **overrides)
    for split in ("train", "val", "test"):
        _make_split(tmp_path / "dataset", split)

    with pytest.raises(SplitAuditError, match="alias|distinct"):
        audit_split_config(config)


def test_unlabeled_negative_image_is_counted_but_does_not_fail(tmp_path: Path):
    config = _make_clean_fixture(tmp_path)

    report, rows = audit_split_config(config)

    assert report.status == "PASS"
    assert report.splits["train"].unlabeled_images == 1
    assert report.splits["train"].negative_images == 1
    assert next(row for row in rows if row.split == "train").label_status == "missing"


def test_audit_writes_reusable_manifest_and_pass_report(tmp_path: Path):
    config = _make_clean_fixture(tmp_path)
    report, rows = audit_split_config(config)
    report_path = tmp_path / "audit.json"
    manifest_path = tmp_path / "manifest.csv"

    write_audit_artifacts(report, rows, report_path=report_path, manifest_path=manifest_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["manifest"]["sha256"]
    assert payload["data_config"]["sha256"]
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 4


def test_within_split_duplicate_is_reported_without_leakage_failure(tmp_path: Path):
    config = _make_clean_fixture(tmp_path)
    root = tmp_path / "dataset"
    (root / "images" / "train" / "patient-d.png").write_bytes(b"train")

    report, _ = audit_split_config(config)

    assert report.status == "PASS"
    assert report.splits["train"].duplicate_content_groups == 1
    assert report.splits["train"].duplicate_content_excess == 1
    assert report.duplicate_groups[0]["cross_split"] is False
