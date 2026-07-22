"""Labeling-job worker entrypoint. Runs in a separate process (spawn) so
torch/CUDA never load in the API process. Must stay picklable: module-level
function, plain-dict arguments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _append_progress(path: Path, event: dict) -> None:
    event = {"ts": time.time(), **event}
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def run_label_job(job_id: str, cfg_dict: dict, jobs_dir: str) -> dict:
    from app.core.decision import DecisionConfig
    from app.core.inference import JobCancelled, LabelJobConfig, run_labeling

    job_dir = Path(jobs_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"
    cancel_path = job_dir / "CANCEL"

    cfg = LabelJobConfig(
        model_paths=[Path(p) for p in cfg_dict["model_paths"]],
        model_names=cfg_dict.get("model_names"),
        images_dir=Path(cfg_dict["images_dir"]),
        out_dir=Path(cfg_dict["out_dir"]),
        decision=DecisionConfig(
            conf_min=cfg_dict.get("conf_min", 0.10),
            conf_confirm=cfg_dict.get("conf_confirm", 0.60),
            per_class_confirm={
                int(k): v for k, v in cfg_dict.get("per_class_confirm", {}).items()
            },
            iou_conflict=cfg_dict.get("iou_conflict", 0.50),
            empty_policy=cfg_dict.get("empty_policy", "review"),
        ),
        iou_wbf=cfg_dict.get("iou_wbf", 0.55),
        device=cfg_dict.get("device"),
        batch_size=cfg_dict.get("batch_size", 16),
        imgsz=cfg_dict.get("imgsz", 640),
        job_id=job_id,
    )

    try:
        result = run_labeling(
            cfg,
            progress=lambda ev: _append_progress(progress_path, ev),
            cancel_check=cancel_path.exists,
        )
    except JobCancelled:
        return {"status": "cancelled"}
    except Exception as e:  # surface the message to the progress stream + parent
        _append_progress(progress_path, {"phase": "error", "msg": str(e)})
        raise

    return {
        "status": "done",
        "total": result.total,
        "confirmed": result.confirmed,
        "review": result.review,
        "negative": result.negative,
        "registry": result.registry,
    }
