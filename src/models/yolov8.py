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


def preserve_lcbamv2_model_for_training(target: YOLO) -> YOLO:
    """Make Ultralytics train from the already-built custom model.

    Ultralytics 8.4.135 rebuilds YAML-created models inside ``Model.train``
    unless the wrapper represents a loaded checkpoint. Marking the in-memory
    model as that source keeps transferred weights and the trial-specific
    LCBAMv2 initialization at the Trainer boundary without editing Ultralytics.
    """
    target.ckpt = {"model": target.model, "epoch": -1}
    return target


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
    """Transfer YOLOv8n weights around exactly one inserted LCBAMv2 layer."""
    source_model = source.model if isinstance(source, YOLO) else source
    source_state = source_model.float().state_dict()
    target_state = target.model.state_dict()
    mapped = {}
    attention_indices = [
        index
        for index, module in enumerate(target.model.model)
        if isinstance(module, LCBAMv2)
    ]
    if len(attention_indices) != 1:
        raise ValueError(
            "pretrained transfer requires exactly one top-level LCBAMv2 layer; "
            f"found {len(attention_indices)}"
        )
    attention_index = attention_indices[0]

    for key, value in source_state.items():
        match = re.match(r"^model\.(\d+)\.(.+)$", key)
        if not match:
            candidate = key
        else:
            index = int(match.group(1))
            candidate = (
                f"model.{index + 1}.{match.group(2)}"
                if index >= attention_index
                else key
            )
        if candidate in target_state and target_state[candidate].shape == value.shape:
            mapped[candidate] = value

    target.model.load_state_dict(mapped, strict=False)
    new_layer_prefix = f"model.{attention_index}."
    new_layer_keys = tuple(key for key in target_state if key.startswith(new_layer_prefix))
    unmatched = tuple(key for key in target_state if key not in mapped)
    return PretrainedTransferReport(
        transferred=len(mapped),
        target_items=len(target_state),
        new_layer_keys=new_layer_keys,
        unmatched_target_keys=unmatched,
    )
