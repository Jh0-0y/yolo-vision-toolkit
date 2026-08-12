"""Model-comparison worker: score one or more models against an UPLOADED YOLO
test set (images + labels + data.yaml), separately per model, so they can be
compared. Runs in a child process (torch/ultralytics). Progress →
jobs_dir/{job_id}/progress.jsonl; result → jobs_dir/{job_id}/result.json.

Metrics per model:
  - Precision / Recall / F1 + TP/FP/FN at the chosen display confidence
    (micro-averaged overall, and per class).
  - mAP@0.5 and mAP@0.5:0.95 + per-class AP, computed COCO-style over the FULL
    prediction set (every box down to the detector floor), independent of the
    display confidence.

Predictions are mapped to the dataset's class ids by NAME (normalized); a
predicted class that isn't in the dataset is kept as a false positive (id -1)
rather than dropped. GT is read from the dataset's YOLO label files.

Overlay images are served by index via jobs manifest — see
`images_manifest.json` next to result.json and the compare image route.
"""

from __future__ import annotations

import json
from pathlib import Path

from infra import jobs

# sentinel class id for predictions whose class isn't defined in the dataset;
# it never matches any GT box, so it is correctly counted as a false positive.
_OTHER = -1


def _read_yaml_names(dataset_dir: Path) -> dict[int, str]:
    """Class names from the dataset's data.yaml (list or dict form). Pure — the
    API layer already verified a data.yaml exists before dispatching."""
    import yaml

    for candidate in ("data.yaml", "data.yml"):
        hits = sorted(dataset_dir.rglob(candidate))
        if hits:
            data = yaml.safe_load(hits[0].read_text()) or {}
            raw = data.get("names")
            if isinstance(raw, dict):
                return {int(k): str(v) for k, v in raw.items()}
            if isinstance(raw, list):
                return {i: str(v) for i, v in enumerate(raw)}
            return {}
    return {}


def _label_for(img: Path) -> Path | None:
    """The YOLO label path for an image: swap the last `images` segment for
    `labels` and the suffix for `.txt`. Returns None if that file is absent."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return None
    label = Path(*parts).with_suffix(".txt")
    return label if label.exists() else None


def _gather_pairs(dataset_dir: Path, image_exts: set[str]) -> list[tuple[Path, Path]]:
    """(image, label) pairs for every labeled image under the dataset."""
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(dataset_dir.rglob("*")):
        if img.suffix.lower() not in image_exts or not img.is_file():
            continue
        label = _label_for(img)
        if label is not None:
            pairs.append((img, label))
    return pairs


def run_compare(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    from app.core.config import resolve_device, settings
    from app.ml.evaluate import (
        IOU_THRESHOLDS,
        accumulate,
        aggregate,
        map_from_accumulated,
        match_for_ap,
        match_frame,
    )
    from app.ml.labeling import IMAGE_EXTS
    from app.ml.predict import PredictConfig, predict_image
    from lib.labels.io import read_label_file
    from lib.labels.registry import normalize

    job = jobs.at(Path(jobs_dir), job_id)
    progress = job.progress_path

    dataset_dir = Path(cfg["dataset_dir"])
    conf_thr = float(cfg.get("conf", 0.4))
    iou_match = float(cfg.get("iou", 0.5))
    device = resolve_device(cfg.get("device"))

    names = _read_yaml_names(dataset_dir)
    names[_OTHER] = "(not in dataset)"
    ds_by_norm = {normalize(name): cid for cid, name in names.items() if cid != _OTHER}
    warning = None if ds_by_norm else "data.yaml has no class names — every prediction counts as a false positive."

    pairs = _gather_pairs(dataset_dir, set(IMAGE_EXTS))

    try:
        if not pairs:
            raise RuntimeError(
                "No labeled images found in the dataset (expected images/ and labels/ with data.yaml)."
            )

        from ultralytics import YOLO

        specs = cfg["specs"]  # [(model_id, pt)]
        model_names = cfg.get("model_names", {})  # {model_id: display name}
        models = []
        for model_id, pt in specs:
            model = YOLO(pt)
            try:
                model.to(device)
            except Exception:
                pass
            models.append((model_id, model))

        pcfg = PredictConfig(
            conf=conf_thr, iou_wbf=float(cfg.get("iou_wbf", 0.55)),
            imgsz=int(cfg.get("imgsz", 640)), device=device,
        )

        totals: dict[str, dict[int, dict[str, int]]] = {mid: {} for mid, _ in specs}
        det_counts: dict[str, int] = {mid: 0 for mid, _ in specs}
        # acc[model_id][iou_thr][cls] -> list of (score, is_tp) for AP
        ap_acc: dict[str, dict[float, dict[int, list]]] = {
            mid: {t: {} for t in IOU_THRESHOLDS} for mid, _ in specs
        }
        gt_by_cls: dict[int, int] = {}  # identical across models (same dataset)
        images: list[dict] = []
        manifest: dict[str, str] = {}  # image index -> absolute path (served by route)
        jobs.emit(progress, {"phase": "start", "total": len(pairs)})

        for i, (img_path, label_path) in enumerate(pairs):
            if job.cancelled():
                jobs.emit(progress, {"phase": "cancelled", "done": i, "total": len(pairs)})
                return {"status": "cancelled"}

            gt = [{"cls": cls, "xyxy_n": list(xyxy)} for cls, xyxy in read_label_file(label_path)]
            for g in gt:
                gt_by_cls[g["cls"]] = gt_by_cls.get(g["cls"], 0) + 1

            manifest[str(i)] = str(img_path)
            entry = {
                "stem": img_path.stem,
                "name": img_path.name,
                "url": f"{settings.api_prefix}/predict/compare/{job_id}/images/{i}",
                "gt_boxes": [
                    {"cls": g["cls"], "name": names.get(g["cls"], str(g["cls"])), "xyxyn": g["xyxy_n"]}
                    for g in gt
                ],
                "per_model": [],
            }
            for model_id, model in models:
                res = predict_image([(model_id, model)], str(img_path), pcfg)
                preds_full = []
                for b in res["boxes"]:
                    cid = ds_by_norm.get(normalize(b["name"]))
                    if cid is None:
                        cid = _OTHER  # not a dataset class → false positive
                    preds_full.append(
                        {"cls": cid, "name": b["name"], "xyxyn": b["xyxyn"], "score": b["score"]}
                    )
                # display-confidence subset drives P/R/F1 counts + the overlay boxes
                preds_disp = [p for p in preds_full if p["score"] >= conf_thr]
                m = match_frame(gt, preds_disp, iou_match)
                accumulate(totals[model_id], m["per_class"])
                det_counts[model_id] += len(preds_disp)
                # full prediction set drives mAP (independent of display conf)
                for t in IOU_THRESHOLDS:
                    bucket = ap_acc[model_id][t]
                    for cls, score, is_tp in match_for_ap(gt, preds_full, t):
                        bucket.setdefault(cls, []).append((score, is_tp))
                entry["per_model"].append({
                    "model_id": model_id,
                    "pred_boxes": [
                        {"cls": p["cls"], "name": p["name"], "score": round(p["score"], 4), "xyxyn": p["xyxyn"]}
                        for p in preds_disp
                    ],
                })
            images.append(entry)
            if (i + 1) % 3 == 0 or i + 1 == len(pairs):
                jobs.emit(progress, {"phase": "analyze", "done": i + 1, "total": len(pairs)})

        per_model = []
        for model_id, _ in specs:
            agg = aggregate(totals[model_id], names)
            ap = map_from_accumulated(ap_acc[model_id], gt_by_cls)
            for row in agg["per_class"]:
                cls_ap = ap["per_class"].get(row["cls"], {})
                row["ap50"] = cls_ap.get("ap50", 0.0)
                row["ap"] = cls_ap.get("ap", 0.0)
            per_model.append({
                "model_id": model_id,
                "name": model_names.get(model_id, model_id),
                "overall": agg["overall"],
                "per_class": agg["per_class"],
                "detections": det_counts[model_id],
                "map50": ap["map50"],
                "map": ap["map"],
            })

        (job.path / "images_manifest.json").write_text(json.dumps(manifest))
        result = {
            "per_model": per_model,
            "images": images,
            "image_count": len(images),
            "conf": conf_thr,
            "iou": iou_match,
            "warning": warning,
        }
        (job.path / "result.json").write_text(json.dumps(result))
        jobs.emit(progress, {"phase": "done", "done": len(pairs), "total": len(pairs)})
        return {"status": "done", "images": len(images)}
    except Exception as e:
        jobs.emit(progress, {"phase": "error", "msg": str(e)})
        raise
