"""Model-comparison worker: score one or more models against ground truth on a
chosen subset of a project's LABELED images. Runs in a child process
(torch/ultralytics). Progress → jobs_dir/{job_id}/progress.jsonl; result →
jobs_dir/{job_id}/result.json.

Each model is scored SEPARATELY (not ensembled) so results can be compared.
Predictions are mapped to project class ids by NAME; a predicted class that
isn't in the project is kept as a false positive (id -1) rather than dropped —
dropping was the bug that made metrics read all-zero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# sentinel class id for predictions whose class isn't defined in the project;
# it never matches any GT box, so it is correctly counted as a false positive.
_OTHER = -1


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def run_compare(job_id: str, cfg: dict, jobs_dir: str) -> dict:
    from app.core.config import resolve_device, settings
    from app.domain.class_registry import normalize
    from app.domain.classes import read_classes
    from app.domain.labels import read_boxes, read_reviewed
    from app.ml.evaluate import accumulate, aggregate, match_frame
    from app.ml.predict import PredictConfig, predict_image

    job_dir = Path(jobs_dir) / job_id
    progress = job_dir / "progress.jsonl"
    cancel = job_dir / "CANCEL"

    project_id = cfg["project_id"]
    pdir = settings.projects_dir / project_id
    conf_thr = float(cfg.get("conf", 0.4))
    iou_match = float(cfg.get("iou", 0.5))
    device = resolve_device(cfg.get("device"))

    classes = read_classes(pdir)
    names = {c["id"]: c["name"] for c in classes}
    names[_OTHER] = "(not in project)"
    proj_by_norm = {normalize(c["name"]): c["id"] for c in classes}
    warning = None if classes else "This project has no class definitions (classes.json is empty)."

    # resolve the image subset → label stems that actually have a label file
    labeled = {p.stem for p in (pdir / "labels").glob("*.txt")}
    requested = cfg.get("image_names")
    if requested:
        stems = [Path(n).stem for n in requested if Path(n).stem in labeled]
    else:
        stems = sorted(labeled)
        if cfg.get("reviewed_only"):
            rev = read_reviewed(pdir)
            stems = [s for s in stems if s in rev]

    def raw_name(stem: str) -> str | None:
        for p in (pdir / "raw").glob(f"{stem}.*"):
            if p.suffix.lower() != ".json":
                return p.name
        return None

    try:
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
        images: list[dict] = []
        _emit(progress, {"phase": "start", "total": len(stems)})

        for i, stem in enumerate(stems):
            if cancel.exists():
                _emit(progress, {"phase": "cancelled", "done": i, "total": len(stems)})
                return {"status": "cancelled"}
            name = raw_name(stem)
            if name is None:
                continue
            gt = read_boxes(pdir, stem)  # [{cls, xyxy_n}]
            entry = {
                "stem": stem,
                "name": name,
                "url": f"{settings.api_prefix}/files/projects/{project_id}/raw/{name}",
                "gt_boxes": [
                    {"cls": g["cls"], "name": names.get(g["cls"], str(g["cls"])), "xyxyn": g["xyxy_n"]}
                    for g in gt
                ],
                "per_model": [],
            }
            for model_id, model in models:
                res = predict_image([(model_id, model)], str(pdir / "raw" / name), pcfg)
                preds = []
                for b in res["boxes"]:
                    if b["score"] < conf_thr:
                        continue
                    cid = proj_by_norm.get(normalize(b["name"]))
                    if cid is None:
                        cid = _OTHER  # not a project class → false positive
                    preds.append({"cls": cid, "name": b["name"], "xyxyn": b["xyxyn"], "score": b["score"]})
                m = match_frame(gt, preds, iou_match)
                accumulate(totals[model_id], m["per_class"])
                det_counts[model_id] += len(preds)
                entry["per_model"].append({
                    "model_id": model_id,
                    "pred_boxes": [
                        {"cls": p["cls"], "name": p["name"], "score": round(p["score"], 4), "xyxyn": p["xyxyn"]}
                        for p in preds
                    ],
                })
            images.append(entry)
            if (i + 1) % 3 == 0 or i + 1 == len(stems):
                _emit(progress, {"phase": "analyze", "done": i + 1, "total": len(stems)})

        per_model = []
        for model_id, _ in specs:
            agg = aggregate(totals[model_id], names)
            per_model.append({
                "model_id": model_id,
                "name": model_names.get(model_id, model_id),
                "overall": agg["overall"],
                "per_class": agg["per_class"],
                "detections": det_counts[model_id],
            })

        result = {
            "per_model": per_model,
            "images": images,
            "image_count": len(images),
            "conf": conf_thr,
            "iou": iou_match,
            "warning": warning,
        }
        (job_dir / "result.json").write_text(json.dumps(result))
        _emit(progress, {"phase": "done", "done": len(stems), "total": len(stems)})
        return {"status": "done", "images": len(images)}
    except Exception as e:
        _emit(progress, {"phase": "error", "msg": str(e)})
        raise
