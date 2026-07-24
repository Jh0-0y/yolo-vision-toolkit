"""Ensemble labeling pipeline: N models over an image folder.

Writes labels/{stem}.txt for every processed image (empty file = negative).
No confidence-based routing — every result awaits user review.

This module is the worker-process entrypoint — ultralytics/torch are imported
lazily so the API process never touches CUDA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.domain.class_registry import ClassRegistry
from app.ml.ensemble import Detection, merge_detections
from app.domain.yolo_io import atomic_write_text, write_box_meta, write_label_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Detection floor for auto-labeling: boxes are kept down to this confidence so
# low-confidence detections aren't silently dropped. `cfg.conf` is then a review
# boundary — boxes below it are flagged "needs_review" rather than discarded.
DETECT_FLOOR = 0.05

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
    conf: float = 0.4
    iou_wbf: float = 0.55
    device: str | None = None
    batch_size: int = 16
    imgsz: int = 640
    # restrict to these file names (None = all images in images_dir)
    image_names: list[str] | None = None
    job_id: str | None = None


@dataclass
class LabelJobResult:
    total: int = 0
    labeled: int = 0
    boxes: int = 0
    stems: list[str] = field(default_factory=list)
    registry: dict = field(default_factory=dict)


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def _load_registry(out_dir: Path) -> ClassRegistry:
    """Seed from the existing classes.json so re-labeling never renumbers ids."""
    path = out_dir / "classes.json"
    if not path.exists():
        return ClassRegistry()
    try:
        return ClassRegistry.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError, ValueError):
        return ClassRegistry()


def run_labeling(
    cfg: LabelJobConfig,
    progress: ProgressFn | None = None,
    cancel_check: CancelFn | None = None,
) -> LabelJobResult:
    from ultralytics import YOLO

    from app.core.config import resolve_device

    device = resolve_device(cfg.device)

    models: list[tuple[str, "YOLO", dict[int, int]]] = []
    registry = _load_registry(cfg.out_dir)
    for i, path in enumerate(cfg.model_paths):
        model_id = cfg.model_names[i] if cfg.model_names else path.stem
        model = YOLO(str(path))
        mapping = registry.add_model(model_id, dict(model.names))
        models.append((model_id, model, mapping))

    class_sources = registry.class_sources()
    images = list_images(cfg.images_dir)
    if cfg.image_names is not None:
        wanted = set(cfg.image_names)
        images = [p for p in images if p.name in wanted]
    result = LabelJobResult(total=len(images), registry=registry.to_dict())

    labels_dir = cfg.out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
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

        for model_id, model, mapping in models:
            results = model.predict(
                [str(p) for p in batch],
                conf=min(cfg.conf, DETECT_FLOOR),
                imgsz=cfg.imgsz,
                device=device,
                verbose=False,
            )
            for i, r in enumerate(results):
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
            label_file = labels_dir / f"{img_path.stem}.txt"
            write_label_file(label_file, [(fb.cls, fb.xyxy) for fb in fused])
            # Boxes below the user's conf are kept but flagged for review.
            write_box_meta(
                label_file,
                [
                    {
                        "score": fb.score,
                        "status": "needs_review" if fb.score < cfg.conf else None,
                    }
                    for fb in fused
                ],
            )
            result.labeled += 1
            result.boxes += len(fused)
            result.stems.append(img_path.stem)
            done += 1

        if progress:
            progress(
                {
                    "phase": "inference",
                    "done": done,
                    "total": result.total,
                    "labeled": result.labeled,
                    "boxes": result.boxes,
                }
            )

    if progress:
        progress({"phase": "done", "done": done, "total": result.total})
    return result
