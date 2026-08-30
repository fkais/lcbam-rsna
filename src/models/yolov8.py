from pathlib import Path

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
