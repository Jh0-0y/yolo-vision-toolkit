"""Projects: 목록 · 생성 · 삭제.

프로젝트는 **껍데기**다 — 이미지·라벨·클래스·검수·분할은 전부 데이터셋 안에 있고
(`endpoints/datasets*.py`), 모델과 학습 기록만 프로젝트 스코프로 남는다.
"""

from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.core.config import settings
from app.db import get_session
from app.models import ModelEntry, Project, TrainRun, iso_utc
from app.schemas.project import ProjectCreate, ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])

# 새 프로젝트는 `datasets/` 하나만 갖는다.
PROJECT_SUBDIRS = ("datasets",)


def _project_dir(project_id: str):
    return settings.projects_dir / project_id


def _get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [ProjectOut(id=p.id, name=p.name, created_at=iso_utc(p.created_at)) for p in projects]


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(req: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=req.name)
    pdir = _project_dir(project.id)
    for sub in PROJECT_SUBDIRS:
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    (pdir / "project.json").write_text(
        json.dumps({"id": project.id, "name": project.name}, ensure_ascii=False)
    )
    session.add(project)
    session.commit()
    return ProjectOut(id=project.id, name=project.name, created_at=iso_utc(project.created_at))


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = _get_project(session, project_id)
    # cascade: this project's scoped models & training runs. Their files live
    # under the project dir (removed by the rmtree below); here we drop the DB
    # rows and each run's progress log in the shared jobs pool.
    for run in session.exec(select(TrainRun).where(TrainRun.project_id == project_id)).all():
        shutil.rmtree(settings.jobs_dir / run.id, ignore_errors=True)
        session.delete(run)
    for model in session.exec(select(ModelEntry).where(ModelEntry.project_id == project_id)).all():
        session.delete(model)
    session.delete(project)
    session.commit()
    shutil.rmtree(_project_dir(project_id), ignore_errors=True)
