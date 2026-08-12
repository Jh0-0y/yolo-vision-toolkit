"""Pure inference for the Test playground: N loaded models over one image.

Unlike `ml/labeling.py` (which writes label files for review), this returns
detections in-memory as plain dicts — nothing is committed to disk or DB.

Every box down to DETECT_FLOOR is returned so the UI's confidence slider can
filter client-side without re-running inference. `cfg.conf` is only carried
through as the slider's initial value; it does NOT drop boxes here.

Pure module: no process/DB concerns. `ultralytics` is imported lazily by the
worker that calls this; this file only touches loaded model objects it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.detect.ensemble import Detection, merge_detections
from lib.detect.labeling import DETECT_FLOOR
from lib.labels.registry import ClassRegistry


@dataclass
class PredictConfig:
    conf: float = 0.4  # review boundary / slider default (not a hard filter here)
    iou_wbf: float = 0.55
    imgsz: int = 640
    device: str | None = None


def predict_image(models: list[tuple[str, object]], image, cfg: PredictConfig) -> dict:
    """Run each loaded model over one image and fuse the results.

    `models` is a list of (display_model_id, loaded ultralytics YOLO) pairs —
    already warm in the caller's process. `image` is anything ultralytics
    accepts (path str, ndarray). Returns a JSON-ready dict.
    """
    registry = ClassRegistry()
    mappings: list[tuple[str, object, dict[int, int]]] = []
    task = "detect"
    for model_id, model in models:
        names = dict(getattr(model, "names", {}) or {})
        mapping = registry.add_model(model_id, names)
        task = getattr(model, "task", task) or task
        mappings.append((model_id, model, mapping))

    dets: list[Detection] = []
    for model_id, model, mapping in mappings:
        results = model.predict(
            image,
            conf=DETECT_FLOOR,
            iou=0.7,
            imgsz=cfg.imgsz,
            device=cfg.device,
            verbose=False,
        )
        for r in results:
            if r.boxes is None:
                continue
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for k in range(len(clss)):
                dets.append(
                    Detection(
                        cls=mapping[int(clss[k])],
                        xyxy=tuple(float(v) for v in xyxyn[k]),  # type: ignore[arg-type]
                        score=float(confs[k]),
                        model_id=model_id,
                    )
                )

    class_sources = registry.class_sources()
    fused = merge_detections(dets, class_sources, iou_thr=cfg.iou_wbf)
    names = registry.names

    boxes = [
        {
            "cls": fb.cls,
            "name": names.get(fb.cls, str(fb.cls)),
            "score": round(fb.score, 4),
            "xyxyn": [round(v, 5) for v in fb.xyxy],
            "model_ids": sorted({m for m, _ in fb.sources}),
            "agree": fb.agree_count,
        }
        for fb in fused
    ]
    # stable order: highest confidence first
    boxes.sort(key=lambda b: b["score"], reverse=True)
    return {"task": task, "names": names, "boxes": boxes, "floor": DETECT_FLOOR}
