"""Labeling-job worker entrypoint. Runs in a separate process (spawn) so
torch/CUDA never load in the API process. Must stay picklable: module-level
function, plain-dict arguments.
"""

from __future__ import annotations

from pathlib import Path

from infra import jobs


def run_label_job(job_id: str, cfg_dict: dict, jobs_dir: str) -> dict:
    from app.core.config import resolve_device
    from lib.detect.labeling import (
        DetectorSpec,
        JobCancelled,
        LabelJobConfig,
        run_labeling,
    )

    job = jobs.at(Path(jobs_dir), job_id).ensure()
    progress_path = job.progress_path

    cfg = LabelJobConfig(
        detectors=[
            DetectorSpec(
                model_path=Path(d["model_path"]),
                model_id=d["model_id"],
                mode=d.get("mode", "full"),
                imgsz=d.get("imgsz", 640),
                tile_size=d.get("tile_size", 640),
                stride=d.get("stride", 480),
                merge_iou=d.get("merge_iou", 0.5),
                border_margin_px=d.get("border_margin_px", 4),
            )
            for d in cfg_dict["detectors"]
        ],
        images_dir=Path(cfg_dict["images_dir"]),
        out_dir=Path(cfg_dict["out_dir"]),
        conf=cfg_dict.get("conf", 0.4),
        iou_wbf=cfg_dict.get("iou_wbf", 0.55),
        # 순수 계산 쪽은 설정을 모른다 — 여기서 해석해 넘긴다
        device=resolve_device(cfg_dict.get("device")),
        batch_size=cfg_dict.get("batch_size", 16),
        image_names=cfg_dict.get("image_names"),
        max_boxes_per_class=cfg_dict.get("max_boxes_per_class"),
        job_id=job_id,
    )

    try:
        result = run_labeling(
            cfg,
            progress=lambda ev: jobs.emit(progress_path, ev),
            cancel_check=job.cancelled,
        )
    except JobCancelled:
        return {"status": "cancelled"}
    except Exception as e:  # surface the message to the progress stream + parent
        jobs.emit(progress_path, {"phase": "error", "msg": str(e)})
        raise

    return {
        "status": "done",
        "total": result.total,
        "labeled": result.labeled,
        "boxes": result.boxes,
        "stems": result.stems,
        "registry": result.registry,
    }
