"""Labeling jobs: launch, status, cancel, SSE progress stream."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session, session_scope
from app.jobs.runner import job_manager, read_progress
from app.models import Job, ModelEntry, Project

router = APIRouter(prefix="/api", tags=["jobs"])

TERMINAL_PHASES = {"done", "error", "cancelled"}
TERMINAL_STATUSES = {"done", "error", "cancelled"}


class JobCreate(BaseModel):
    model_ids: list[str]
    conf_min: float = 0.10
    conf_confirm: float = 0.60
    iou_wbf: float = 0.55
    iou_conflict: float = 0.50
    empty_policy: str = "review"
    imgsz: int = 640
    batch_size: int = 16


class JobOut(BaseModel):
    id: str
    project_id: str
    status: str
    config: dict
    result: dict | None
    error: str | None
    created_at: str


def _to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        config=json.loads(job.config_json),
        result=json.loads(job.result_json) if job.result_json else None,
        error=job.error,
        created_at=job.created_at.isoformat(),
    )


@router.post("/projects/{project_id}/jobs", response_model=JobOut)
def create_job(project_id: str, req: JobCreate, session: Session = Depends(get_session)):
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "프로젝트가 없습니다")
    if not req.model_ids:
        raise HTTPException(422, "모델을 1개 이상 선택하세요")

    model_paths = []
    model_names = []
    for mid in req.model_ids:
        entry = session.get(ModelEntry, mid)
        if entry is None:
            raise HTTPException(422, f"모델이 없습니다: {mid}")
        pt = settings.models_dir / mid / "model.pt"
        if not pt.exists():
            raise HTTPException(422, f"모델 파일이 없습니다: {mid}")
        model_paths.append(str(pt))
        # unique display id: name, disambiguated by registry id on collision
        name = entry.name if entry.name not in model_names else f"{entry.name}#{mid[-4:]}"
        model_names.append(name)

    pdir = settings.projects_dir / project_id
    cfg = {
        "model_paths": model_paths,
        "model_names": model_names,
        "images_dir": str(pdir / "raw"),
        "out_dir": str(pdir),
        "conf_min": req.conf_min,
        "conf_confirm": req.conf_confirm,
        "iou_wbf": req.iou_wbf,
        "iou_conflict": req.iou_conflict,
        "empty_policy": req.empty_policy,
        "imgsz": req.imgsz,
        "batch_size": req.batch_size,
        "device": settings.device,
    }
    job = Job(project_id=project_id, kind="label", config_json=json.dumps(req.model_dump()))
    session.add(job)
    session.commit()
    session.refresh(job)

    job_manager.submit_label_job(job.id, project_id, cfg)
    session.refresh(job)
    return _to_out(job)


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: str, session: Session = Depends(get_session)):
    jobs = session.exec(
        select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())
    ).all()
    return [_to_out(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "잡이 없습니다")
    return _to_out(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "잡이 없습니다")
    if job.status not in TERMINAL_STATUSES:
        job_manager.cancel(job_id)
        session.expire_all()
        job = session.get(Job, job_id)
    return _to_out(job)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, session: Session = Depends(get_session)):
    if session.get(Job, job_id) is None:
        raise HTTPException(404, "잡이 없습니다")

    async def stream():
        offset = 0
        idle_polls = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, job_id, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL_PHASES:
                    terminal = True
            if terminal:
                return
            if not events:
                idle_polls += 1
                # no new lines: check DB in case the job died without a final event
                if idle_polls % 10 == 0:
                    with session_scope() as s:
                        job = s.get(Job, job_id)
                        if job is not None and job.status in TERMINAL_STATUSES:
                            yield {
                                "event": "progress",
                                "data": json.dumps({"phase": job.status, "msg": job.error}),
                            }
                            return
            else:
                idle_polls = 0
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())
