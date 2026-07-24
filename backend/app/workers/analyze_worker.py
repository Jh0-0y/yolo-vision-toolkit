"""Precise-analysis worker: run a model over a project's LABELED images and
compare predictions to ground truth. Runs in a child process (torch/ultralytics).
Progress → jobs_dir/{job_id}/progress.jsonl; result → jobs_dir/{job_id}/result.json.

Predictions are remapped to PROJECT class ids by class NAME (a model trained on
this project shares names, but ids could differ) so matching is robust.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _emit(progress_path: Path, event: dict) -> None:
    with open(progress_path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def run_analyze(job_id: str, cfg: dict, jobs_dir: str) -> dict:
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

    names = {c["id"]: c["name"] for c in read_classes(pdir)}
    proj_ids = {normalize(c["name"]): c["id"] for c in read_classes(pdir)}

    stems = sorted(p.stem for p in (pdir / "labels").glob("*.txt"))
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

        models = []
        for model_id, pt in cfg["specs"]:
            model = YOLO(pt)
            try:
                model.to(device)
            except Exception:
                pass
            models.append((model_id, model))

        pcfg = PredictConfig(conf=conf_thr, iou_wbf=float(cfg.get("iou_wbf", 0.55)),
                             imgsz=int(cfg.get("imgsz", 640)), device=device)

        total: dict[int, dict[str, int]] = {}
        worst: list[dict] = []
        _emit(progress, {"phase": "start", "total": len(stems)})

        for i, stem in enumerate(stems):
            if cancel.exists():
                _emit(progress, {"phase": "cancelled", "done": i, "total": len(stems)})
                return {"status": "cancelled"}
            name = raw_name(stem)
            if name is None:
                continue
            gt = read_boxes(pdir, stem)  # [{cls, xyxy_n}]
            res = predict_image(models, str(pdir / "raw" / name), pcfg)

            preds = []
            for b in res["boxes"]:
                if b["score"] < conf_thr:
                    continue
                cid = proj_ids.get(normalize(b["name"]))
                if cid is None:  # predicted a class not in this project → ignore
                    continue
                preds.append({"cls": cid, "name": b["name"], "xyxyn": b["xyxyn"], "score": b["score"]})

            m = match_frame(gt, preds, iou_match)
            accumulate(total, m["per_class"])
            if m["fp"] + m["fn"] > 0:
                worst.append({
                    "stem": stem, "name": name,
                    "url": f"{settings.api_prefix}/files/projects/{project_id}/raw/{name}",
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "gt_boxes": [
                        {"cls": g["cls"], "name": names.get(g["cls"], str(g["cls"])), "xyxyn": g["xyxy_n"]}
                        for g in gt
                    ],
                    "pred_boxes": [
                        {"cls": p["cls"], "name": p["name"], "score": round(p["score"], 4), "xyxyn": p["xyxyn"]}
                        for p in preds
                    ],
                })
            if (i + 1) % 5 == 0 or i + 1 == len(stems):
                _emit(progress, {"phase": "analyze", "done": i + 1, "total": len(stems)})

        agg = aggregate(total, names)
        worst.sort(key=lambda w: -(w["fp"] + w["fn"]))
        result = {
            **agg,
            "worst": worst[:50],
            "images": len(stems),
            "conf": conf_thr,
            "iou": iou_match,
        }
        (job_dir / "result.json").write_text(json.dumps(result))
        _emit(progress, {"phase": "done", "done": len(stems), "total": len(stems)})
        return {"status": "done", "images": len(stems)}
    except Exception as e:
        _emit(progress, {"phase": "error", "msg": str(e)})
        raise
