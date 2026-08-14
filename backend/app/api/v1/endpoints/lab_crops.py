"""연구실 크롭 런 — 시작 · 목록 · 상세 · 진행률 · 산출물 · 복제 · 삭제.

옛 크롭 랩은 시작(predict 계열)과 읽기(crops)가 갈라져 있었다. 여기서는 **한 곳**이다
— 런처가 보내는 것이 곧 런의 설정 스냅샷이고, 시작하면 `crop_id` 를 돌려주므로
화면은 바로 상세로 넘어가 진행률을 본다(학습이 TrainPage → TrainRunDetailPage 로
가는 것과 같다).

진행률은 **전역 잡 카드에 올리지 않는다.** 상세 페이지가 보여준다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.labs import the_lab
from app.api.v1.endpoints.predict.common import job_event_stream
from app.core.config import settings
from app.db import get_session
from app.models import ModelEntry
from app.schemas.lab_crop import (
    LabCropCreate,
    LabCropDuplicate,
    LabCropRename,
    LabCropRunOut,
)
from app.services import lab_crop_runs, lab_store
from app.services.test_jobs import test_job_manager

router = APIRouter(prefix="/lab/crops", tags=["lab"])


def _require_run(session: Session, crop_id: str) -> tuple[str, dict]:
    lab_id = the_lab(session).id
    if not lab_crop_runs.valid_id(crop_id):
        raise HTTPException(422, "Invalid crop id")
    meta = lab_crop_runs.read_meta(lab_id, crop_id)
    if meta is None:
        raise HTTPException(404, "Crop run not found")
    return lab_id, meta


def _resolve_models(session: Session, detectors: list) -> tuple[list[dict], list[dict]]:
    """검출기 엔트리 → (워커용 cfg, 런 메타에 남길 표시).

    연구실은 프로젝트에 속하지 않으므로 **전체 모델 풀**에서 고를 수 있다 —
    학습실 경로(`predict.common.model_pt`)의 프로젝트 스코프 검사를 쓰지 않는다.
    """
    if not detectors:
        raise HTTPException(422, "Pick at least one detection model")
    worker: list[dict] = []
    shown: list[dict] = []
    for d in detectors:
        entry = session.get(ModelEntry, d.model_id)
        if entry is None:
            raise HTTPException(422, f"Model not found: {d.model_id}")
        pt = settings.model_dir(entry.project_id, entry.id) / "model.pt"
        if not pt.exists():
            raise HTTPException(422, f"Model file missing: {d.model_id}")
        if d.mode not in ("full", "tiled"):
            raise HTTPException(422, "detector mode must be 'full' or 'tiled'")
        if d.mode == "tiled" and (d.stride <= 0 or d.stride > d.tile_size):
            raise HTTPException(422, "stride must be in (0, tile_size]")
        worker.append(
            {
                "pt": str(pt),
                "mode": d.mode,
                "conf": d.conf,
                "imgsz": d.imgsz,
                "tile_size": d.tile_size,
                "stride": d.stride,
                "merge_iou": d.merge_iou,
            }
        )
        shown.append(
            {"model_id": entry.id, "name": entry.name, "mode": d.mode, "conf": d.conf}
        )
    return worker, shown


def _start(session: Session, req: LabCropCreate) -> dict:
    """런 자리를 만들고 잡을 던진다. 목록·상세 응답 한 건을 돌려준다."""
    lab_id = the_lab(session).id
    if not lab_store.valid_id(req.source_video_id):
        raise HTTPException(422, "Invalid video id")
    video_meta = lab_store.read_video_meta(lab_id, req.source_video_id)
    if video_meta is None:
        raise HTTPException(404, "Video not found")
    source = lab_store.video_file(lab_id, req.source_video_id)
    if source is None:
        raise HTTPException(404, "Video file missing")

    worker_models, shown_models = _resolve_models(session, req.detectors)
    source_name = video_meta.get("name") or video_meta.get("filename", "")

    crop_id = lab_crop_runs.create(
        lab_id,
        (req.name or "").strip() or source_name,
        {
            "source_video_id": req.source_video_id,
            "source_name": source_name,
            "models": shown_models,
            "overrides": req.overrides,
            "crop_w": req.crop_w,
            "crop_h": req.crop_h,
            "toggles": req.toggles.model_dump(),
        },
    )
    test_job_manager.submit_lab_crop(
        crop_id,
        {
            "source": str(source),
            "work": str(lab_crop_runs.run_dir(lab_id, crop_id)),
            "detectors": worker_models,
            "overrides": req.overrides,
            "crop_w": req.crop_w,
            "crop_h": req.crop_h,
            "toggles": req.toggles.model_dump(),
            "device": req.device,
        },
    )
    return lab_crop_runs.to_out(lab_id, crop_id, lab_crop_runs.read_meta(lab_id, crop_id) or {})


@router.post("", response_model=LabCropRunOut, status_code=201)
def create_crop_run(req: LabCropCreate, session: Session = Depends(get_session)):
    return _start(session, req)


@router.get("", response_model=list[LabCropRunOut])
async def list_crop_runs(session: Session = Depends(get_session)):
    lab_id = the_lab(session).id
    return await run_in_threadpool(lab_crop_runs.list_runs, lab_id)


@router.get("/{crop_id}", response_model=LabCropRunOut)
def get_crop_run(crop_id: str, session: Session = Depends(get_session)):
    lab_id, meta = _require_run(session, crop_id)
    return lab_crop_runs.to_out(lab_id, crop_id, meta)


@router.get("/{crop_id}/events")
async def crop_run_events(crop_id: str, session: Session = Depends(get_session)):
    """진행률 SSE. progress.jsonl 을 처음부터 재생하므로 언제 붙어도 같은 그림이다 —
    탭을 옮기거나 새로고침해도 진행률이 이어진다."""
    _require_run(session, crop_id)
    return await job_event_stream(crop_id)


@router.post("/{crop_id}/cancel")
def cancel_crop_run(crop_id: str, session: Session = Depends(get_session)):
    _require_run(session, crop_id)
    test_job_manager.cancel(crop_id)
    return {"cancelled": True}


@router.post("/{crop_id}/duplicate", response_model=LabCropRunOut, status_code=201)
def duplicate_crop_run(
    crop_id: str,
    body: LabCropDuplicate,
    session: Session = Depends(get_session),
):
    """"이 설정 그대로 다시, 모델만 바꿔서" — 모델 업그레이드 전후를 즉시 비교한다."""
    lab_id, meta = _require_run(session, crop_id)
    detectors = body.detectors
    if detectors is None:
        # 원본이 쓴 모델을 그대로 — 메타에는 표시용만 남아 있으므로 다시 엮는다
        from app.schemas.lab_crop import CropDetector

        detectors = [
            CropDetector(model_id=m["model_id"], mode=m.get("mode", "full"), conf=m.get("conf"))
            for m in meta.get("models", [])
            if m.get("model_id")
        ]
    req = LabCropCreate(
        name=(body.name or "").strip() or f"{meta.get('name', crop_id)} (copy)",
        source_video_id=meta.get("source_video_id", ""),
        detectors=detectors,
        overrides=meta.get("overrides", {}),
        crop_w=meta.get("crop_w", 608),
        crop_h=meta.get("crop_h", 1080),
        toggles=meta.get("toggles", {}),
    )
    return _start(session, req)


def _artifact(lab_id: str, crop_id: str, name: str, media_type: str, filename: str):
    path = lab_crop_runs.artifact(lab_id, crop_id, name)
    if not path.exists():
        raise HTTPException(404, f"{name} not ready")
    # FileResponse 가 Range 를 처리한다 — 브라우저가 탐색하며 재생할 수 있어야 한다
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/{crop_id}/crop.json")
def download_crop_json(crop_id: str, session: Session = Depends(get_session)):
    lab_id, _ = _require_run(session, crop_id)
    return _artifact(lab_id, crop_id, lab_crop_runs.CROP_NAME, "application/json", "crop.json")


@router.get("/{crop_id}/wide.mp4")
def download_wide(crop_id: str, session: Session = Depends(get_session)):
    lab_id, _ = _require_run(session, crop_id)
    return _artifact(lab_id, crop_id, lab_crop_runs.WIDE_NAME, "video/mp4", "wide.mp4")


@router.get("/{crop_id}/crop.mp4")
def download_cut(crop_id: str, session: Session = Depends(get_session)):
    lab_id, _ = _require_run(session, crop_id)
    return _artifact(lab_id, crop_id, lab_crop_runs.CUT_NAME, "video/mp4", "crop.mp4")


@router.patch("/{crop_id}", response_model=LabCropRunOut)
def rename_crop_run(
    crop_id: str,
    body: LabCropRename,
    session: Session = Depends(get_session),
):
    lab_id, meta = _require_run(session, crop_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    meta["name"] = name
    lab_crop_runs.write_meta(lab_id, crop_id, meta)
    return lab_crop_runs.to_out(lab_id, crop_id, meta)


@router.delete("/{crop_id}", status_code=204)
def delete_crop_run(crop_id: str, session: Session = Depends(get_session)):
    lab_id, _ = _require_run(session, crop_id)
    test_job_manager.cancel(crop_id)  # 아직 돌고 있으면 멈추라고 알린다
    lab_crop_runs.delete(lab_id, crop_id)
