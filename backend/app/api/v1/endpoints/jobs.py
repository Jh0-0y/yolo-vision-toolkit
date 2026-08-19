"""Labeling jobs: launch, status, cancel, SSE progress stream."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session, session_scope
from app.models import Job, ModelEntry, Project, iso_utc
from app.schemas.job import JobCreate, JobOut
from app.services import datasets
from app.services.label_manager import job_manager
from infra import jobs

router = APIRouter(prefix="", tags=["jobs"])

TERMINAL_PHASES = {"done", "error", "cancelled"}
TERMINAL_STATUSES = {"done", "error", "cancelled"}


def _to_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        config=json.loads(job.config_json),
        result=json.loads(job.result_json) if job.result_json else None,
        error=job.error,
        created_at=iso_utc(job.created_at),
    )


@router.post("/projects/{project_id}/jobs", response_model=JobOut, status_code=201)
def create_job(project_id: str, req: JobCreate, session: Session = Depends(get_session)):
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not req.model_ids:
        raise HTTPException(422, "Select at least one model")

    model_paths = []
    model_names = []
    for mid in req.model_ids:
        entry = session.get(ModelEntry, mid)
        if entry is None:
            raise HTTPException(422, f"Model not found: {mid}")
        # only this project's models or legacy/shared (project_id NULL) are usable
        if entry.project_id is not None and entry.project_id != project_id:
            raise HTTPException(422, f"Model does not belong to this project: {mid}")
        pt = settings.model_dir(entry.project_id, mid) / "model.pt"
        if not pt.exists():
            raise HTTPException(422, f"Model file missing: {mid}")
        model_paths.append(str(pt))
        # unique display id: name, disambiguated by registry id on collision
        name = entry.name if entry.name not in model_names else f"{entry.name}#{mid[-4:]}"
        model_names.append(name)

    # 오토라벨링의 대상은 **데이터셋**이다 — 이미지도 라벨도 클래스도 그 안에 있다.
    if not req.dataset_id:
        raise HTTPException(422, "dataset_id is required")
    if not datasets.valid_id(req.dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, req.dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    ddir = datasets.dataset_dir(project_id, req.dataset_id)

    cfg = {
        "model_paths": model_paths,
        "model_names": model_names,
        "images_dir": str(ddir / "raw"),
        "out_dir": str(ddir),
        "conf": req.conf,
        "iou_wbf": req.iou_wbf,
        "imgsz": req.imgsz,
        "batch_size": req.batch_size,
        "image_names": req.names,
        "max_boxes_per_class": req.max_boxes_per_class,
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
        raise HTTPException(404, "Job not found")
    return _to_out(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status not in TERMINAL_STATUSES:
        job_manager.cancel(job_id)
        session.expire_all()
        job = session.get(Job, job_id)
    return _to_out(job)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, session: Session = Depends(get_session)):
    if session.get(Job, job_id) is None:
        raise HTTPException(404, "Job not found")

    async def stream():
        offset = 0
        idle_polls = 0
        while True:
            events, offset = await asyncio.to_thread(jobs.at(settings.jobs_dir, job_id).read, offset)
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
