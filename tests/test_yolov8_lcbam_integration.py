from pathlib import Path

import torch

from src.models import LCBAMv2, build_lcbamv2_yolov8


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "models" / "yolov8n-lcbamv2.yaml"


def test_yolov8_yaml_builds_model_with_lcbamv2():
    model = build_lcbamv2_yolov8(MODEL_CONFIG)

    lcbam_layers = [module for module in model.model.modules() if isinstance(module, LCBAMv2)]

    assert len(lcbam_layers) == 1


def test_yolov8_lcbamv2_complete_forward():
    model = build_lcbamv2_yolov8(MODEL_CONFIG)
    model.model.eval()

    with torch.inference_mode():
        outputs = model.model(torch.randn(1, 3, 64, 64))

    predictions, feature_maps = outputs
    assert predictions.shape[0] == 1
    assert len(feature_maps) == 3

