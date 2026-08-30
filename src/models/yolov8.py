from pathlib import Path
from dataclasses import dataclass
import re

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn import tasks

from .lcbam_v2 import LCBAMv2


def register_lcbamv2() -> None:
    """Expose LCBAMv2 to Ultralytics' YAML model parser."""
    registered = getattr(tasks, "LCBAMv2", None)
    if registered not in (None, LCBAMv2):
        raise RuntimeError("Ultralytics already has a different LCBAMv2 registration.")

    tasks.LCBAMv2 = LCBAMv2


def build_lcbamv2_yolov8(model_config: str | Path) -> YOLO:
    """Build a YOLO model after registering the project's custom layer."""
    register_lcbamv2()
    return YOLO(str(Path(model_config)))


@dataclass(frozen=True)
class PretrainedTransferReport:
    transferred: int
    target_items: int
    new_layer_keys: tuple[str, ...]
    unmatched_target_keys: tuple[str, ...]


def load_lcbamv2_pretrained(
    target: YOLO,
    source: YOLO | nn.Module,
) -> PretrainedTransferReport:
    """Transfer YOLOv8n weights while accounting for LCBAMv2 at layer index 10.

    The custom layer follows SPPF (index 9), so parameters from every later
    source layer are shifted by one in the target state dict. Shape-incompatible
    task-head tensors (for example COCO's 80 classes versus RSNA's 1 class) are
    intentionally left at the target model's initialization.
    """
    source_model = source.model if isinstance(source, YOLO) else source
    source_state = source_model.float().state_dict()
    target_state = target.model.state_dict()
    mapped = {}

    for key, value in source_state.items():
        match = re.match(r"^model\.(\d+)\.(.+)$", key)
        if not match:
            candidate = key
        else:
            index = int(match.group(1))
            candidate = (
                f"model.{index + 1}.{match.group(2)}" if index >= 10 else key
            )
        if candidate in target_state and target_state[candidate].shape == value.shape:
            mapped[candidate] = value

    target.model.load_state_dict(mapped, strict=False)
    new_layer_keys = tuple(key for key in target_state if key.startswith("model.10."))
    unmatched = tuple(key for key in target_state if key not in mapped)
    return PretrainedTransferReport(
        transferred=len(mapped),
        target_items=len(target_state),
        new_layer_keys=new_layer_keys,
        unmatched_target_keys=unmatched,
    )
