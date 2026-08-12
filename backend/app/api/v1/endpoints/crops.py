"""크롭 랩 산출물 — 목록 · 진행률 · 산출 파일 · 이름변경 · 삭제.

시작은 `predict/annotate` 계열이 맡고(업로드·모델 해석이 거기 있다), 시작한 뒤의
모든 것은 여기서 다룬다. 목록이 곧 진행 현황이라, 화면은 브라우저에 아무것도
기억해 두지 않고 이 엔드포인트만 물어보면 된다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import job_event_stream
from app.db import get_session
from app.models import Project
from app.schemas.crop import CropRunOut, CropRunRename
from app.services import crop_runs
from app.services.test_jobs import test_job_manager

router = APIRouter(prefix="/projects/{project_id}/crops", tags=["crops"])


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _require_run(session: Session, project_id: str, crop_id: str) -> dict:
    _require_project(session, project_id)
    if not crop_runs.valid_id(crop_id):
        raise HTTPException(422, "Invalid crop id")
    meta = crop_runs.read_meta(project_id, crop_id)
    if meta is None:
        raise HTTPException(404, "Crop run not found")
    return meta


@router.get("", response_model=list[CropRunOut])
async def list_crop_runs(project_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    await run_in_threadpool(crop_runs.sweep_expired)
    return await run_in_threadpool(crop_runs.list_runs, project_id)


@router.get("/{crop_id}", response_model=CropRunOut)
def get_crop_run(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    meta = _require_run(session, project_id, crop_id)
    return crop_runs.to_out(project_id, crop_id, meta)


@router.get("/{crop_id}/events")
async def crop_run_events(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    """진행률 SSE. progress.jsonl 을 처음부터 재생하므로 언제 붙어도 같은 그림이다."""
    _require_run(session, project_id, crop_id)
    return await job_event_stream(crop_id)


@router.post("/{crop_id}/cancel")
def cancel_crop_run(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    _require_run(session, project_id, crop_id)
    test_job_manager.cancel(crop_id)
    return {"cancelled": True}


@router.get("/{crop_id}/crop.json")
def download_crop_json(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    """adaptive-crop 이 계산한 크롭 X 좌표(keyframes + 100ms 샘플)."""
    _require_run(session, project_id, crop_id)
    path = crop_runs.run_dir(project_id, crop_id) / crop_runs.CROP_NAME
    if not path.exists():
        raise HTTPException(404, "Crop coordinates not available")
    return FileResponse(path, media_type="application/json", filename="crop.json")


@router.get("/{crop_id}/video")
def download_crop_video(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    meta = _require_run(session, project_id, crop_id)
    path = crop_runs.run_dir(project_id, crop_id) / crop_runs.VIDEO_NAME
    if not path.exists():
        raise HTTPException(
            410 if meta.get("video_expired") else 404,
            "Video expired" if meta.get("video_expired") else "Video not ready",
        )
    return FileResponse(path, media_type="video/mp4")  # FileResponse 가 Range 를 처리한다


@router.patch("/{crop_id}", response_model=CropRunOut)
def rename_crop_run(
    project_id: str,
    crop_id: str,
    body: CropRunRename,
    session: Session = Depends(get_session),
):
    meta = _require_run(session, project_id, crop_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    meta["name"] = name
    crop_runs.write_meta(project_id, crop_id, meta)
    return crop_runs.to_out(project_id, crop_id, meta)


@router.delete("/{crop_id}", status_code=204)
def delete_crop_run(project_id: str, crop_id: str, session: Session = Depends(get_session)):
    _require_run(session, project_id, crop_id)
    test_job_manager.cancel(crop_id)  # 아직 돌고 있으면 멈추라고 알린다
    crop_runs.delete(project_id, crop_id)
