import csv
import random
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
import yaml

from src import optimize_lcbam
from src.models import LCBAMv2, build_lcbamv2_yolov8


class FakeTrial:
    def __init__(self, number=0, choices=None):
        self.number = number
        self.choices = choices or {
            "channel_kernel": 5,
            "spatial_kernel": 7,
            "insert_position": "p4",
        }
        self.params = {}
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        assert self.choices[name] in choices
        self.params[name] = self.choices[name]
        return self.choices[name]

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class FakeMetrics:
    def __init__(self, values=None):
        self.results_dict = values or {
            "metrics/precision(B)": 0.61,
            "metrics/recall(B)": 0.52,
            "metrics/mAP50(B)": 0.48,
            "metrics/mAP50-95(B)": 0.31,
        }


def write_data_config(config_path: Path, **overrides) -> Path:
    data = {
        "path": "dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["pneumonia"],
    }
    data.update(overrides)
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def make_settings(tmp_path: Path, data: Path) -> optimize_lcbam.SearchSettings:
    return optimize_lcbam.SearchSettings(
        data=data,
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
        workers=0,
        seed=17,
        study_name="unit-study",
        storage=None,
        sampler_seed=29,
        output_root=tmp_path / "runs",
        results_root=tmp_path / "results",
        weights=None,
        fraction=1.0,
        trial_budget=2,
    )


def test_search_space_only_generates_legal_lcbamv2_parameters():
    trial = FakeTrial()

    params = optimize_lcbam.sample_search_params(trial)

    assert params == optimize_lcbam.LCBAMSearchParams(
        channel_kernel=5,
        spatial_kernel=7,
        insert_position="p4",
    )
    assert set(trial.params) == {
        "channel_kernel",
        "spatial_kernel",
        "insert_position",
    }


@pytest.mark.parametrize("position", ["p3", "p4", "p5"])
@pytest.mark.parametrize("channel_kernel,spatial_kernel", [(3, 3), (5, 5), (7, 7)])
def test_trial_model_config_uses_static_legal_position_yaml(
    tmp_path, position, channel_kernel, spatial_kernel
):
    trial_dir = tmp_path / f"{position}-{channel_kernel}-{spatial_kernel}"
    trial_dir.mkdir()
    params = optimize_lcbam.LCBAMSearchParams(
        channel_kernel=channel_kernel,
        spatial_kernel=spatial_kernel,
        insert_position=position,
    )

    snapshot = optimize_lcbam.write_trial_model_config(params, trial_dir)
    config = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    attention_layers = [
        layer
        for section in ("backbone", "head")
        for layer in config[section]
        if layer[2] == "LCBAMv2"
    ]

    assert snapshot == trial_dir / "model_config.yaml"
    assert len(attention_layers) == 1
    assert attention_layers[0][3] == [channel_kernel, 3, 0.0, spatial_kernel]


@pytest.mark.parametrize("channel_kernel,spatial_kernel", [(3, 7), (5, 3), (7, 5)])
def test_trial_kernel_variants_build_a_yolov8_model(
    tmp_path, channel_kernel, spatial_kernel
):
    trial_dir = tmp_path / f"model-{channel_kernel}-{spatial_kernel}"
    trial_dir.mkdir()
    params = optimize_lcbam.LCBAMSearchParams(
        channel_kernel=channel_kernel,
        spatial_kernel=spatial_kernel,
        insert_position="p5",
    )
    snapshot = optimize_lcbam.write_trial_model_config(params, trial_dir)

    model = build_lcbamv2_yolov8(snapshot)
    attention = next(
        module for module in model.model.modules() if isinstance(module, LCBAMv2)
    )

    assert attention.channel_conv.kernel_size == (channel_kernel,)
    assert attention.spatial_conv.kernel_size == (spatial_kernel, spatial_kernel)


@pytest.mark.parametrize(
    "overrides",
    [
        {"test": None},
        {"val": ""},
        {"test": "images/val"},
        {"test": "./images/val"},
        {"train": "images/val"},
    ],
)
def test_data_guard_requires_distinct_nonempty_train_val_test(tmp_path, overrides):
    data = write_data_config(tmp_path / "data.yaml", **overrides)

    with pytest.raises(ValueError, match="train|val|test|distinct"):
        optimize_lcbam.validate_data_config(data)


def test_data_guard_resolves_split_aliases_against_dataset_root(tmp_path):
    dataset_root = tmp_path / "dataset"
    data = write_data_config(
        tmp_path / "data.yaml",
        path=str(dataset_root),
        val="images/val",
        test=str(dataset_root / "images" / "val"),
    )

    with pytest.raises(ValueError, match="distinct"):
        optimize_lcbam.validate_data_config(data)


def test_objective_returns_validation_map_and_never_calls_test(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)
    trial = FakeTrial(number=4)
    train_calls = []

    class FakeModel:
        def train(self, **kwargs):
            train_calls.append(kwargs)
            return FakeMetrics()

        def val(self, **kwargs):
            raise AssertionError(f"objective must not call model.val: {kwargs}")

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    value = optimize_lcbam.LCBAMObjective(settings)(trial)

    assert value == pytest.approx(0.31)
    assert train_calls[0]["val"] is True
    assert train_calls[0]["split"] == "val"
    assert train_calls[0]["seed"] == 17
    assert train_calls[0]["deterministic"] is True
    assert train_calls[0]["optimizer"] == "AdamW"
    assert train_calls[0]["lr0"] == pytest.approx(0.002)
    assert train_calls[0]["lrf"] == pytest.approx(0.01)
    assert train_calls[0]["momentum"] == pytest.approx(0.9)
    assert train_calls[0]["weight_decay"] == pytest.approx(0.0005)
    assert train_calls[0]["warmup_epochs"] == pytest.approx(3.0)
    assert train_calls[0]["patience"] == 0
    assert train_calls[0]["amp"] is True
    assert train_calls[0]["cache"] is False
    assert train_calls[0]["mosaic"] == pytest.approx(1.0)
    assert train_calls[0]["mixup"] == pytest.approx(0.0)
    assert train_calls[0]["resume"] is False
    assert "test" not in train_calls[0]
    assert trial.user_attrs["mAP50"] == pytest.approx(0.48)
    assert trial.user_attrs["mAP50-95"] == pytest.approx(0.31)
    assert trial.user_attrs["precision"] == pytest.approx(0.61)
    assert trial.user_attrs["recall"] == pytest.approx(0.52)
    assert Path(trial.user_attrs["output_dir"]).name == "trial_004"


def test_objective_passes_absolute_project_path_to_ultralytics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = write_data_config(tmp_path / "data.yaml")
    base = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{**base.__dict__, "output_root": Path("relative-runs")}
    )
    train_calls = []

    class FakeModel:
        def train(self, **kwargs):
            train_calls.append(kwargs)
            return FakeMetrics()

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    optimize_lcbam.LCBAMObjective(settings)(FakeTrial())

    assert Path(train_calls[0]["project"]).is_absolute()
    assert Path(train_calls[0]["project"]) == (
        tmp_path / "relative-runs" / settings.study_name
    )


def test_objective_rejects_missing_or_nonfinite_validation_metric(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)

    class FakeModel:
        def train(self, **_):
            return FakeMetrics(
                {
                    "metrics/precision(B)": 0.6,
                    "metrics/recall(B)": 0.5,
                    "metrics/mAP50(B)": 0.4,
                    "metrics/mAP50-95(B)": float("nan"),
                }
            )

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    with pytest.raises(optimize_lcbam.TrialTrainingError, match="finite"):
        optimize_lcbam.LCBAMObjective(settings)(FakeTrial())


def test_objective_rejects_missing_validation_metric(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)

    class FakeModel:
        def train(self, **_):
            return FakeMetrics(
                {
                    "metrics/precision(B)": 0.6,
                    "metrics/recall(B)": 0.5,
                    "metrics/mAP50(B)": 0.4,
                }
            )

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    with pytest.raises(ValueError, match="missing"):
        optimize_lcbam.LCBAMObjective(settings)(FakeTrial())


def test_objective_rejects_automatic_batch_reduction(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)

    class FakeTrainer:
        batch_size = settings.batch // 2

    class FakeModel:
        trainer = FakeTrainer()

        def train(self, **_):
            return FakeMetrics()

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    with pytest.raises(optimize_lcbam.TrialTrainingError, match="batch"):
        optimize_lcbam.LCBAMObjective(settings)(FakeTrial())


def test_objective_does_not_hide_data_path_errors(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)

    class FakeModel:
        def train(self, **_):
            raise FileNotFoundError("missing train split")

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())

    with pytest.raises(FileNotFoundError, match="missing train split"):
        optimize_lcbam.LCBAMObjective(settings)(FakeTrial())


def test_trial_directories_are_isolated_and_collisions_fail_closed(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)

    class FakeModel:
        def train(self, **_):
            return FakeMetrics()

    monkeypatch.setattr(optimize_lcbam, "build_trial_model", lambda *_: FakeModel())
    objective = optimize_lcbam.LCBAMObjective(settings)

    objective(FakeTrial(number=0))
    objective(FakeTrial(number=1))

    assert (settings.output_root / settings.study_name / "trial_000").is_dir()
    assert (settings.output_root / settings.study_name / "trial_001").is_dir()
    with pytest.raises(FileExistsError):
        objective(FakeTrial(number=0))


def test_reproducibility_seed_covers_python_numpy_and_torch():
    optimize_lcbam.set_reproducible_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    optimize_lcbam.set_reproducible_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_sqlite_study_resumes_and_persists_csv_and_best_yaml(tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    db = tmp_path / "storage" / "study.db"
    settings = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{**settings.__dict__, "storage": f"sqlite:///{db.as_posix()}"}
    )

    first = optimize_lcbam.create_study(settings)
    first.optimize(
        lambda trial: (
            trial.suggest_categorical("channel_kernel", [3]),
            trial.set_user_attr("spatial_kernel", 5),
            trial.set_user_attr("insert_position", "p5"),
            trial.set_user_attr("mAP50", 0.5),
            trial.set_user_attr("mAP50-95", 0.3),
            trial.set_user_attr("precision", 0.6),
            trial.set_user_attr("recall", 0.4),
            trial.set_user_attr("output_dir", "trial_000"),
            0.3,
        )[-1],
        n_trials=1,
    )
    optimize_lcbam.persist_study_results(first, settings)
    resumed = optimize_lcbam.create_study(settings)

    assert len(resumed.trials) == 1
    csv_path = settings.results_root / settings.study_name / "trials.csv"
    best_path = settings.results_root / settings.study_name / "best_params.yaml"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    best = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    assert rows[0]["state"] == "COMPLETE"
    assert best["best_trial"] == 0
    assert best["best_value"] == pytest.approx(0.3)
    assert best["fixed_conditions"]["seed"] == 17
    assert resumed.sampler._rng.rng.seed is not None


def test_sqlite_resume_rejects_changed_fixed_conditions(tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    db = tmp_path / "storage" / "study.db"
    settings = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{**settings.__dict__, "storage": f"sqlite:///{db.as_posix()}"}
    )
    optimize_lcbam.create_study(settings)
    changed = optimize_lcbam.SearchSettings(
        **{**settings.__dict__, "batch": settings.batch + 1}
    )

    with pytest.raises(ValueError, match="protocol|settings"):
        optimize_lcbam.create_study(changed)


def test_sqlite_resume_rejects_changed_model_definition(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    db = tmp_path / "storage" / "study.db"
    settings = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{**settings.__dict__, "storage": f"sqlite:///{db.as_posix()}"}
    )
    model_yaml = tmp_path / "p3.yaml"
    model_yaml.write_text("version: one\n", encoding="utf-8")
    monkeypatch.setitem(optimize_lcbam.POSITION_CONFIGS, "p3", model_yaml)
    optimize_lcbam.create_study(settings)

    model_yaml.write_text("version: two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protocol|settings"):
        optimize_lcbam.create_study(settings)


def test_study_requires_pretrained_weight_to_exist_before_protocol_is_recorded(
    tmp_path,
):
    data = write_data_config(tmp_path / "data.yaml")
    base = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{**base.__dict__, "weights": str(tmp_path / "missing.pt")}
    )

    with pytest.raises(FileNotFoundError, match="weight"):
        optimize_lcbam.create_study(settings)


def test_run_search_marks_expected_training_failure_and_continues(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    settings = make_settings(tmp_path, data)
    calls = 0

    class SometimesFails:
        def __init__(self, _settings):
            pass

        def __call__(self, trial):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise optimize_lcbam.TrialTrainingError("simulated OOM")
            trial.suggest_categorical("channel_kernel", [3])
            return 0.2

    monkeypatch.setattr(optimize_lcbam, "LCBAMObjective", SometimesFails)

    study = optimize_lcbam.run_search(settings, trials=2)

    assert [trial.state.name for trial in study.trials] == ["FAIL", "COMPLETE"]
    assert study.best_value == pytest.approx(0.2)
    assert (settings.results_root / settings.study_name / "trials.csv").is_file()


def test_cli_supports_trials_alias_and_keeps_full_fraction_by_default():
    args = optimize_lcbam.build_parser().parse_args(["--n-trials", "3"])

    assert args.trials == 3
    assert args.fraction == 1.0
    assert args.optimizer == "AdamW"
    assert args.patience == 0


def test_formal_search_requires_passing_current_split_audit(tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    base = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{
            **base.__dict__,
            **optimize_lcbam.FORMAL_FROZEN_VALUES,
            "formal": True,
            "weights": "yolov8n.pt",
            "split_audit_report": None,
        }
    )

    with pytest.raises(ValueError, match="split audit"):
        optimize_lcbam.validate_formal_preconditions(settings)


def test_formal_search_accepts_matching_passing_split_audit(tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / split / f"{split}.png").write_bytes(split.encode())
    base = make_settings(tmp_path, data)
    report = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.csv"
    from scripts.audit_rsna_split import audit_split_config, write_audit_artifacts

    audit, rows = audit_split_config(data)
    write_audit_artifacts(
        audit, rows, report_path=report, manifest_path=manifest
    )
    settings = optimize_lcbam.SearchSettings(
        **{
            **base.__dict__,
            **optimize_lcbam.FORMAL_FROZEN_VALUES,
            "formal": True,
            "study_name": "lcbam_structure_search_v1",
            "weights": "yolov8n.pt",
            "split_audit_report": report,
        }
    )

    optimize_lcbam.validate_formal_preconditions(settings)


def test_formal_search_rejects_data_changed_after_passing_audit(tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / split / f"{split}.png").write_bytes(split.encode())
    from scripts.audit_rsna_split import audit_split_config, write_audit_artifacts

    report = tmp_path / "audit.json"
    manifest = tmp_path / "manifest.csv"
    audit, rows = audit_split_config(data)
    write_audit_artifacts(audit, rows, report_path=report, manifest_path=manifest)
    (dataset / "images" / "train" / "train.png").write_bytes(b"changed")
    base = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{
            **base.__dict__,
            **optimize_lcbam.FORMAL_FROZEN_VALUES,
            "formal": True,
            "study_name": "lcbam_structure_search_v1",
            "weights": "yolov8n.pt",
            "split_audit_report": report,
        }
    )

    with pytest.raises(ValueError, match="size changed|content changed"):
        optimize_lcbam.validate_formal_preconditions(settings)


def test_resume_treats_trials_as_total_budget(monkeypatch, tmp_path):
    data = write_data_config(tmp_path / "data.yaml")
    base = make_settings(tmp_path, data)
    settings = optimize_lcbam.SearchSettings(
        **{
            **base.__dict__,
            "storage": f"sqlite:///{(tmp_path / 'study.db').as_posix()}",
        }
    )
    calls = 0

    class Objective:
        def __init__(self, _settings):
            pass

        def __call__(self, trial):
            nonlocal calls
            calls += 1
            return float(trial.number)

    monkeypatch.setattr(optimize_lcbam, "LCBAMObjective", Objective)

    first = optimize_lcbam.run_search(settings)
    resumed = optimize_lcbam.run_search(settings)

    assert len(first.trials) == settings.trial_budget
    assert len(resumed.trials) == settings.trial_budget
    assert calls == settings.trial_budget


def test_cli_returns_failure_when_no_trial_completes(monkeypatch):
    class EmptyStudy:
        trials = []

    monkeypatch.setattr(optimize_lcbam, "run_search", lambda *_: EmptyStudy())

    exit_code = optimize_lcbam.main(["--trials", "1", "--no-pretrained"])

    assert exit_code == 1
