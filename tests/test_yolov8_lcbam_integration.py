from pathlib import Path

import torch

from src.models import LCBAMv2, build_lcbamv2_yolov8


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "models" / "yolov8n-lcbamv2.yaml"
POSITION_CONFIGS = {
    "p3": ROOT / "configs" / "models" / "yolov8n-lcbamv2-p3.yaml",
    "p4": ROOT / "configs" / "models" / "yolov8n-lcbamv2-p4.yaml",
    "p5": MODEL_CONFIG,
}


def test_yolov8_yaml_builds_model_with_lcbamv2():
    model = build_lcbamv2_yolov8(MODEL_CONFIG)

    lcbam_layers = [module for module in model.model.modules() if isinstance(module, LCBAMv2)]

    assert len(lcbam_layers) == 1


def test_yolov8_lcbamv2_complete_forward():
    model = build_lcbamv2_yolov8(MODEL_CONFIG)
    model.model.eval()

    with torch.inference_mode():
        outputs = model.model(torch.randn(1, 3, 64, 64))

    predictions, raw_outputs = outputs

    assert predictions.shape[:2] == (1, 5)
    assert torch.isfinite(predictions).all()
    assert len(raw_outputs["feats"]) == 3


@torch.inference_mode()
def test_every_supported_insertion_position_builds_and_forwards():
    for position, config in POSITION_CONFIGS.items():
        model = build_lcbamv2_yolov8(config)
        model.model.eval()

        outputs = model.model(torch.randn(1, 3, 64, 64))
        attention_indices = [
            index
            for index, module in enumerate(model.model.model)
            if isinstance(module, LCBAMv2)
        ]

        assert attention_indices == [{"p3": 5, "p4": 7, "p5": 10}[position]]
        assert outputs[0].shape[:2] == (1, 5)
        assert torch.isfinite(outputs[0]).all()
