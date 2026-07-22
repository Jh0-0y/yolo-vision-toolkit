"""Auto-labeling CLI.

Usage:
    uv run python cli.py label --models a.pt b.pt --images ./imgs --out ./out \
        [--conf-min 0.10] [--conf-confirm 0.60] [--iou-wbf 0.55] [--device auto]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.decision import DecisionConfig
from app.core.inference import LabelJobConfig, run_labeling


def main() -> None:
    parser = argparse.ArgumentParser(prog="yvt")
    sub = parser.add_subparsers(dest="command", required=True)

    label = sub.add_parser("label", help="Run ensemble auto-labeling over an image folder")
    label.add_argument("--models", nargs="+", required=True, help="1..N .pt model paths")
    label.add_argument("--images", required=True, help="input image folder")
    label.add_argument("--out", required=True, help="output folder (confirmed/ + review/)")
    label.add_argument("--conf-min", type=float, default=0.10)
    label.add_argument("--conf-confirm", type=float, default=0.60)
    label.add_argument("--iou-wbf", type=float, default=0.55)
    label.add_argument("--iou-conflict", type=float, default=0.50)
    label.add_argument("--empty-policy", choices=["review", "negative"], default="review")
    label.add_argument("--device", default=None, help="auto | cpu | mps | 0")
    label.add_argument("--imgsz", type=int, default=640)
    label.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    if args.command == "label":
        cfg = LabelJobConfig(
            model_paths=[Path(m) for m in args.models],
            images_dir=Path(args.images),
            out_dir=Path(args.out),
            decision=DecisionConfig(
                conf_min=args.conf_min,
                conf_confirm=args.conf_confirm,
                iou_conflict=args.iou_conflict,
                empty_policy=args.empty_policy,
            ),
            iou_wbf=args.iou_wbf,
            device=args.device,
            batch_size=args.batch,
            imgsz=args.imgsz,
        )

        def progress(ev: dict) -> None:
            if ev.get("phase") == "inference":
                print(
                    f"\r[{ev['done']}/{ev['total']}] "
                    f"confirmed={ev['confirmed']} review={ev['review']} negative={ev['negative']}",
                    end="",
                    flush=True,
                )

        result = run_labeling(cfg, progress)
        print()
        print(json.dumps(
            {
                "total": result.total,
                "confirmed": result.confirmed,
                "review": result.review,
                "negative": result.negative,
                "classes": [c["name"] for c in result.registry["classes"]],
            },
            indent=2,
            ensure_ascii=False,
        ))


if __name__ == "__main__":
    main()
