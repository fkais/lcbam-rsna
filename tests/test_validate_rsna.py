from pathlib import Path

import pytest

from scripts.validate_rsna import (
    DatasetValidationError,
    format_report,
    main,
    validate_config,
    validate_dataset,
)


def _make_split(root: Path, split: str) -> None:
    (root / "images" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)


def test_validation_reports_dataset_integrity_without_treating_unlabeled_images_as_errors(
    tmp_path: Path,
):
    for split in ("train", "val", "test"):
        _make_split(tmp_path, split)

    (tmp_path / "images" / "train" / "positive.png").touch()
    (tmp_path / "images" / "train" / "background.png").touch()
    (tmp_path / "images" / "train" / "positive.npy").touch()
    (tmp_path / "labels" / "train" / "positive.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.3 0.4\n", encoding="utf-8"
    )
    (tmp_path / "labels" / "train" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    (tmp_path / "images" / "val" / "empty.png").touch()
    (tmp_path / "labels" / "val" / "empty.txt").touch()

    (tmp_path / "images" / "test" / "bad.png").touch()
    (tmp_path / "labels" / "test" / "bad.txt").write_text(
        "0 0.5 0.5 0.2\n"
        "2 0.5 0.5 0.2 0.2\n"
        "0 nan 0.5 0.2 0.2\n"
        "0 1.2 0.5 0.2 0.2\n"
        "0 0.5 0.5 0 0.2\n",
        encoding="utf-8",
    )

    report = validate_dataset(tmp_path, nc=2)

    train = report.splits["train"]
    assert train.directory_files == 3
    assert train.images == 2
    assert train.ignored_image_files == 1
    assert train.labels == 2
    assert train.unlabeled_images == 1
    assert train.orphan_labels == 1
    assert report.splits["val"].empty_labels == 1
    assert report.splits["test"].invalid_label_rows == 5
    assert report.class_distribution == {0: 2, 1: 1}
    assert any("labels/train/orphan.txt" in issue for issue in report.issues)
    assert report.has_errors


def test_validation_rejects_missing_required_split_directories(tmp_path: Path):
    with pytest.raises(DatasetValidationError, match="images/train"):
        validate_dataset(tmp_path, nc=1)


def test_config_cli_prints_a_passing_summary_for_empty_fixture(tmp_path: Path, capsys):
    for split in ("train", "val", "test"):
        _make_split(tmp_path, split)
    config = tmp_path / "rsna.yaml"
    config.write_text(
        f"path: {tmp_path.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [pneumonia]\n",
        encoding="utf-8",
    )

    report = validate_config(config)
    assert "Result: PASS" in format_report(report)
    assert main(["--data", str(config)]) == 0
    assert "background candidates, not errors" in capsys.readouterr().out
