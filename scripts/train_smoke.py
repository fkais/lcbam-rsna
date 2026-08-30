"""Run a short, explicitly configured Baseline or LCBAMv2 training smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

from src.models import build_lcbamv2_yolov8
from src.models.yolov8 import load_lcbamv2_pretrained


ROOT = Path(__file__).resolve().parents[1]
LCBAMV2_CONFIG = ROOT / "configs" / "models" / "yolov8n-lcbamv2.yaml"


def create_smoke_model(*, model: str, weights: str | None = "yolov8n.pt") -> YOLO:
    model_name = model.casefold()
    if model_name == "baseline":
        return YOLO(weights or "yolov8n.yaml")
    if model_name != "lcbamv2":
        raise ValueError(f"unsupported smoke model: {model}")

    target = build_lcbamv2_yolov8(LCBAMV2_CONFIG)
    if weights:
        report = load_lcbamv2_pretrained(target, YOLO(weights))
        print(
            f"Transferred {report.transferred}/{report.target_items} compatible items; "
            f"LCBAMv2 keeps {len(report.new_layer_keys)} newly initialized items."
        )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str.casefold, choices=("baseline", "lcbamv2"), required=True)
    parser.add_argument("--data", default="configs/rsna.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.02)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Build from YAML without loading YOLOv8n pretrained weights.",
    )
    args = parser.parse_args(argv)

    model = create_smoke_model(
        model=args.model,
        weights=None if args.no_pretrained else args.weights,
    )
    model.train(
        data=args.data,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        workers=args.workers,
        fraction=args.fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
