"""Test playground: run trained models on an image (or dataset frame) and get
detections back — no DB writes. Also manages warm model residency for fast
repeated tests / A-B comparison.

The endpoint is thin: resolve model paths + image path (business), then hand the
raw compute to InferManager (which owns the warm worker). torch never loads here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session
from app.domain.video import VIDEO_EXTS
from app.ml.labeling import IMAGE_EXTS
from app.models import ModelEntry
from app.schemas.predict import PredictResponse, ResidentModel, TestJobStart
from app.services.infer_manager import infer_manager
from app.services.label_manager import read_progress
from app.services.test_jobs import sweep_old_annotations, test_job_manager

router = APIRouter(prefix="/predict", tags=["predict"])

_TERMINAL = {"done", "error", "cancelled"}


async def _job_event_stream(job_id: str):
    """SSE tail of jobs_dir/{job_id}/progress.jsonl until a terminal phase."""

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, job_id, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in _TERMINAL:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())


def _model_pt(session: Session, model_id: str, project_id: str | None) -> str:
    """Resolve a model id to its .pt path, enforcing project scope."""
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(422, f"Model not found: {model_id}")
    if entry.project_id is not None and project_id and entry.project_id != project_id:
        raise HTTPException(422, f"Model does not belong to this project: {model_id}")
    pt = settings.model_dir(entry.project_id, model_id) / "model.pt"
    if not pt.exists():
        raise HTTPException(422, f"Model file missing: {model_id}")
    return str(pt)


@router.post("", response_model=PredictResponse)
async def predict(
    model_ids: str = Form(...),  # comma-separated DB model ids
    conf: float = Form(0.4),
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    image_project_id: str | None = Form(None),
    image_name: str | None = Form(None),
    file: UploadFile | None = File(None),
    session: Session = Depends(get_session),
):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]

    tmp_path: Path | None = None
    if file is not None:
        ext = Path(file.filename or "img").suffix.lower() or ".jpg"
        if ext not in IMAGE_EXTS:
            raise HTTPException(422, f"Unsupported image type: {ext}")
        uploads = settings.test_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        tmp_path = uploads / f"{uuid.uuid4().hex}{ext}"
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        image_path = str(tmp_path)
    elif image_project_id and image_name:
        p = settings.projects_dir / image_project_id / "raw" / Path(image_name).name
        if not p.exists():
            raise HTTPException(404, "Image not found")
        image_path = str(p)
    else:
        raise HTTPException(422, "Provide an image file or a dataset image reference")

    cfg = {"conf": conf, "iou_wbf": iou_wbf, "imgsz": imgsz, "device": device}
    try:
        result = await run_in_threadpool(infer_manager.predict, specs, image_path, cfg)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return result


@router.get("/residents", response_model=list[ResidentModel])
async def list_residents():
    return await run_in_threadpool(infer_manager.residents)


@router.post("/residents/{model_id}", response_model=ResidentModel, status_code=201)
async def load_resident(
    model_id: str,
    project_id: str | None = None,
    device: str | None = None,
    session: Session = Depends(get_session),
):
    pt = _model_pt(session, model_id, project_id)
    return await run_in_threadpool(infer_manager.load, model_id, pt, device)


@router.delete("/residents/{model_id}", status_code=204)
async def unload_resident(model_id: str):
    await run_in_threadpool(infer_manager.unload, model_id)


# ---------- video annotation (batch: draw boxes onto the whole clip) ----------


@router.post("/annotate", response_model=TestJobStart, status_code=201)
async def start_annotate(
    model_ids: str = Form(...),
    conf: float = Form(0.4),
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    object_tracking: bool = Form(True),  # ByteTrack boxes + IDs + trails
    crop_tracking: bool = Form(True),  # trackcrop vertical crop-window overlay
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    ext = Path(file.filename or "v").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type: {ext}")
    if not object_tracking and not crop_tracking:
        raise HTTPException(422, "Enable object tracking, crop tracking, or both")
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]

    await run_in_threadpool(sweep_old_annotations)  # keep test_dir/annotate tidy
    job_id = uuid.uuid4().hex
    work = settings.test_dir / "annotate" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"source{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    cfg = {
        "source": str(src),
        "out": str(work / "out.mp4"),
        "specs": specs,
        "conf": conf,
        "iou_wbf": iou_wbf,
        "imgsz": imgsz,
        "device": device,
        "object_tracking": object_tracking,
        "crop_tracking": crop_tracking,
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/annotate/{job_id}/events")
async def annotate_events(job_id: str):
    return await _job_event_stream(job_id)


@router.get("/annotate/{job_id}/result")
def annotate_result(job_id: str):
    if not job_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid job id")
    out = settings.test_dir / "annotate" / job_id / "out.mp4"
    if not out.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(out, media_type="video/mp4")  # FileResponse handles Range


@router.get("/annotate/{job_id}/crop")
def annotate_crop(job_id: str):
    """trackcrop's computed crop-X coordinates (keyframes + 100ms samples) as JSON.
    Present only when the job ran with crop tracking enabled."""
    if not job_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid job id")
    crop = settings.test_dir / "annotate" / job_id / "crop.json"
    if not crop.exists():
        raise HTTPException(404, "Crop coordinates not available")
    return FileResponse(crop, media_type="application/json", filename="crop.json")


# ---------- model comparison (score models vs labeled ground truth) ----------


@router.post("/compare", response_model=TestJobStart, status_code=201)
async def start_compare(
    project_id: str = Form(...),
    model_ids: str = Form(...),
    image_names: str | None = Form(None),  # comma-separated filenames; empty = all labeled
    conf: float = Form(0.4),
    iou: float = Form(0.5),  # IoU match threshold (pred↔GT)
    imgsz: int = Form(640),
    device: str | None = Form(None),
    reviewed_only: bool = Form(False),
    session: Session = Depends(get_session),
):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs: list[tuple[str, str]] = []
    model_names: dict[str, str] = {}
    for mid in ids:
        specs.append((mid, _model_pt(session, mid, project_id)))
        entry = session.get(ModelEntry, mid)
        model_names[mid] = entry.name if entry else mid
    if not (settings.projects_dir / project_id / "labels").exists():
        raise HTTPException(422, "No labeled data to compare in this project")

    names_list = [n.strip() for n in image_names.split(",") if n.strip()] if image_names else None
    job_id = uuid.uuid4().hex
    cfg = {
        "project_id": project_id,
        "specs": specs,
        "model_names": model_names,
        "image_names": names_list,
        "conf": conf,
        "iou": iou,
        "iou_wbf": 0.55,
        "imgsz": imgsz,
        "device": device,
        "reviewed_only": reviewed_only,
    }
    await run_in_threadpool(test_job_manager.submit_compare, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/compare/{job_id}/events")
async def compare_events(job_id: str):
    return await _job_event_stream(job_id)


@router.get("/compare/{job_id}/result")
def compare_result(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(422, "Invalid job id")
    path = settings.jobs_dir / job_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "Comparison result not ready")
    return json.loads(path.read_text())
