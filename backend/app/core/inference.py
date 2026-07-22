"""Ensemble labeling pipeline: N models over an image folder.

This module is the worker-process entrypoint — ultralytics/torch are imported
lazily so the API process never touches CUDA.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.class_registry import ClassRegistry
from app.core.decision import (
    BoxVerdict,
    DecisionConfig,
    judge_boxes,
    route_image,
    uncertainty_score,
)
from app.core.ensemble import Detection, merge_detections
from app.core.yolo_io import atomic_write_text, write_label_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ProgressFn = Callable[[dict], None]
CancelFn = Callable[[], bool]


class JobCancelled(Exception):
    pass


@dataclass
class LabelJobConfig:
    model_paths: list[Path]
    images_dir: Path
    out_dir: Path
    # display ids per model (defaults to the .pt filename stem)
    model_names: list[str] | None = None
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    iou_wbf: float = 0.55
    device: str | None = None
    batch_size: int = 16
    imgsz: int = 640
    copy_images: bool = True  # copy confirmed images next to labels (else symlink)
    job_id: str | None = None


@dataclass
class LabelJobResult:
    total: int = 0
    confirmed: int = 0
    review: int = 0
    negative: int = 0
    registry: dict = field(default_factory=dict)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def _review_item_payload(
    image_rel: str,
    width: int,
    height: int,
    verdicts: list[BoxVerdict],
    job_id: str | None,
) -> dict:
    return {
        "version": 1,
        "image": image_rel,
        "width": width,
        "height": height,
        "job_id": job_id,
        "uncertainty": uncertainty_score(verdicts),
        "boxes": [
            {
                "id": f"b{i}",
                "cls": v.box.cls,
                "xyxy_n": [round(c, 6) for c in v.box.xyxy],
                "score": round(v.box.score, 4),
                "status": v.status,
                "reason": v.reason,
                "sources": [
                    {"model": m, "score": round(s, 4)} for m, s in v.box.sources
                ],
            }
            for i, v in enumerate(verdicts)
        ],
        "edits": None,
    }


def _place_image(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def run_labeling(
    cfg: LabelJobConfig,
    progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> LabelJobResult:
    from ultralytics import YOLO

    from app.config import resolve_device

    device = resolve_device(cfg.device)

    models: list[tuple[str, "YOLO", dict[int, int]]] = []
    registry = ClassRegistry()
    for i, path in enumerate(cfg.model_paths):
        model_id = cfg.model_names[i] if cfg.model_names else path.stem
        model = YOLO(str(path))
        mapping = registry.add_model(model_id, dict(model.names))
        models.append((model_id, model, mapping))

    class_sources = registry.class_sources()
    images = list_images(cfg.images_dir)
    result = LabelJobResult(total=len(images), registry=registry.to_dict())

    confirmed_img = cfg.out_dir / "confirmed" / "images"
    confirmed_lbl = cfg.out_dir / "confirmed" / "labels"
    review_dir = cfg.out_dir / "review"
    for d in (confirmed_img, confirmed_lbl, review_dir):
        d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.out_dir / "classes.json", json.dumps(registry.to_dict(), indent=2))

    done = 0
    for start in range(0, len(images), cfg.batch_size):
        if cancel_check and cancel_check():
            if progress:
                progress({"phase": "cancelled", "done": done, "total": result.total})
            raise JobCancelled(f"cancelled at {done}/{result.total}")
        batch = images[start : start + cfg.batch_size]
        # detections per image index in this batch
        per_image: list[list[Detection]] = [[] for _ in batch]
        sizes: list[tuple[int, int]] = [(0, 0)] * len(batch)

        for model_id, model, mapping in models:
            results = model.predict(
                [str(p) for p in batch],
                conf=cfg.decision.conf_min,
                imgsz=cfg.imgsz,
                device=device,
                verbose=False,
            )
            for i, r in enumerate(results):
                sizes[i] = (int(r.orig_shape[1]), int(r.orig_shape[0]))  # (w, h)
                if r.boxes is None:
                    continue
                xyxyn = r.boxes.xyxyn.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for k in range(len(clss)):
                    per_image[i].append(
                        Detection(
                            cls=mapping[int(clss[k])],
                            xyxy=tuple(float(v) for v in xyxyn[k]),  # type: ignore[arg-type]
                            score=float(confs[k]),
                            model_id=model_id,
                        )
                    )

        for i, img_path in enumerate(batch):
            fused = merge_detections(per_image[i], class_sources, iou_thr=cfg.iou_wbf)
            verdicts = judge_boxes(fused, class_sources, cfg.decision)
            bucket = route_image(verdicts, cfg.decision)

            if bucket == "confirmed":
                write_label_file(
                    confirmed_lbl / f"{img_path.stem}.txt",
                    [(v.box.cls, v.box.xyxy) for v in verdicts],
                )
                _place_image(img_path, confirmed_img / img_path.name, cfg.copy_images)
                result.confirmed += 1
            elif bucket == "negative":
                write_label_file(confirmed_lbl / f"{img_path.stem}.txt", [])
                _place_image(img_path, confirmed_img / img_path.name, cfg.copy_images)
                result.negative += 1
            else:
                w, h = sizes[i]
                rel = str(img_path.resolve())
                try:
                    rel = str(img_path.resolve().relative_to(cfg.images_dir.resolve()))
                except ValueError:
                    pass
                payload = _review_item_payload(rel, w, h, verdicts, cfg.job_id)
                atomic_write_text(
                    review_dir / f"{img_path.stem}.json", json.dumps(payload, indent=2)
                )
                result.review += 1

            done += 1

        if progress:
            progress(
                {
                    "phase": "inference",
                    "done": done,
                    "total": result.total,
                    "confirmed": result.confirmed,
                    "review": result.review,
                    "negative": result.negative,
                }
            )

    if progress:
        progress({"phase": "done", "done": done, "total": result.total})
    return result
