"""Test playground: run trained models on an image (or dataset frame) and get
detections back — no DB writes. Also manages warm model residency for fast
repeated tests / A-B comparison.

The endpoint is thin: resolve model paths + image path (business), then hand the
raw compute to InferManager (which owns the warm worker). torch never loads here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.core.config import settings
from app.db import get_session
from app.domain.video import VIDEO_EXTS, extract_scrub_frames
from app.ml.labeling import IMAGE_EXTS
from app.models import ModelEntry
from app.schemas.predict import PredictResponse, ResidentModel, VideoUpload
from app.services.infer_manager import infer_manager

router = APIRouter(prefix="/predict", tags=["predict"])


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


# ---------- video frame scrubbing ----------


def _frame_path(video_id: str, idx: int) -> Path:
    if not video_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid video id")
    p = settings.test_dir / "videos" / video_id / "frames" / f"{idx}.jpg"
    if not p.exists():
        raise HTTPException(404, "Frame not found")
    return p


@router.post("/video", response_model=VideoUpload, status_code=201)
async def upload_video(file: UploadFile):
    ext = Path(file.filename or "v").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type: {ext}")
    video_id = uuid.uuid4().hex
    vdir = settings.test_dir / "videos" / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    src = vdir / f"source{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    count = await run_in_threadpool(extract_scrub_frames, src, vdir / "frames")
    if count == 0:
        raise HTTPException(422, "No frames could be extracted from the video")
    return VideoUpload(video_id=video_id, frame_count=count)


@router.get("/video/{video_id}/frame/{idx}")
def video_frame(video_id: str, idx: int):
    return FileResponse(_frame_path(video_id, idx), media_type="image/jpeg")


@router.post("/video/{video_id}/frame/{idx}", response_model=PredictResponse)
async def predict_video_frame(
    video_id: str,
    idx: int,
    model_ids: str = Form(...),
    conf: float = Form(0.4),
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    session: Session = Depends(get_session),
):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]
    frame = str(_frame_path(video_id, idx))
    cfg = {"conf": conf, "iou_wbf": iou_wbf, "imgsz": imgsz, "device": device}
    return await run_in_threadpool(infer_manager.predict, specs, frame, cfg)
