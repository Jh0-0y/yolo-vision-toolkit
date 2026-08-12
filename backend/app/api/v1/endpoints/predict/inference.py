"""단발 추론과 상주 모델 — 이미지 한 장을 모델 여러 개로 돌려 박스를 돌려준다.

DB 에 쓰지 않는다(플레이그라운드). 무거운 계산은 InferManager 가 소유한 따뜻한
워커로 넘기고 여기서는 모델 경로와 이미지 경로만 푼다 — torch 는 올라오지 않는다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import model_pt
from app.core.config import settings
from app.db import get_session
from app.schemas.predict import PredictResponse, ResidentModel
from app.services.infer_manager import infer_manager
from lib.formats import IMAGE_EXTS

router = APIRouter(prefix="/predict", tags=["predict"])


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
    specs = [(mid, model_pt(session, mid, project_id)) for mid in ids]

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
    pt = model_pt(session, model_id, project_id)
    return await run_in_threadpool(infer_manager.load, model_id, pt, device)


@router.delete("/residents/{model_id}", status_code=204)
async def unload_resident(model_id: str):
    await run_in_threadpool(infer_manager.unload, model_id)
