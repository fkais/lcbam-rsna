from pathlib import Path

import torch
from ultralytics import YOLO

from scripts import train_smoke
from src.models import build_lcbamv2_yolov8
from src.models.yolov8 import load_lcbamv2_pretrained


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "models" / "yolov8n-lcbamv2.yaml"


def test_smoke_entry_defaults_workers_to_zero_and_fraction_to_two_percent(monkeypatch):
    calls = []

    class FakeModel:
        def train(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(train_smoke, "create_smoke_model", lambda **_: FakeModel())

    exit_code = train_smoke.main(["--model", "baseline", "--no-pretrained"])

    assert exit_code == 0
    assert calls[0]["workers"] == 0
    assert calls[0]["fraction"] == 0.02


def test_smoke_entry_forwards_all_reproducibility_arguments(monkeypatch):
    calls = []

    class FakeModel:
        def train(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(train_smoke, "create_smoke_model", lambda **_: FakeModel())

    exit_code = train_smoke.main(
        [
            "--model",
            "LCBAMv2",
            "--data",
            "custom.yaml",
            "--device",
            "0",
            "--epochs",
            "3",
            "--batch",
            "4",
            "--imgsz",
            "320",
            "--seed",
            "17",
            "--workers",
            "2",
            "--fraction",
            "0.15",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "data": "custom.yaml",
            "device": "0",
            "epochs": 3,
            "batch": 4,
            "imgsz": 320,
            "seed": 17,
            "workers": 2,
            "fraction": 0.15,
        }
    ]


def test_lcbamv2_pretrained_migration_shifts_downstream_layer_indices():
    source = YOLO("yolov8n.yaml")
    target = build_lcbamv2_yolov8(MODEL_CONFIG)
    source_key = "model.12.cv1.conv.weight"
    target_key = "model.13.cv1.conv.weight"
    attention_before = {
        key: value.clone()
        for key, value in target.model.state_dict().items()
        if key.startswith("model.10.")
    }
    source.model.state_dict()[source_key].fill_(0.125)

    report = load_lcbamv2_pretrained(target, source)

    assert torch.all(target.model.state_dict()[target_key] == 0.125)
    assert report.transferred > 300
    assert set(report.new_layer_keys) == set(attention_before)
    for key, value in attention_before.items():
        assert torch.equal(target.model.state_dict()[key], value)
    assert target.model.state_dict()["model.10.gamma"].item() == 0.0


def test_smoke_models_can_be_built_without_downloading_weights():
    baseline = train_smoke.create_smoke_model(model="baseline", weights=None)
    lcbamv2 = train_smoke.create_smoke_model(model="lcbamv2", weights=None)

    assert isinstance(baseline, YOLO)
    assert any(module.__class__.__name__ == "LCBAMv2" for module in lcbamv2.model.modules())
