"""데이터셋 — 목록 · 생성 · 이름변경 · 삭제.

데이터셋은 학습실의 작업 단위다. 이미지·라벨·클래스·검수 상태·분할을 **자기가 전부
갖는다.** 그래서 삭제는 그 안의 모든 것을 함께 지운다 — 다른 데이터셋과 공유하는 것이
없어 눈치 볼 곳이 없다.

가져오기·검수·분할·내보내기는 각자 다음 단계에서 자기 모듈로 붙는다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.db import get_session
from app.models import Project
from app.schemas.dataset import DatasetCreate, DatasetOut, DatasetPatch
from app.services import datasets

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> dict:
    _require_project(session, project_id)
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    meta = datasets.read_meta(project_id, dataset_id)
    if meta is None:
        raise HTTPException(404, "Dataset not found")
    return meta


@router.get("", response_model=list[DatasetOut])
async def list_datasets(project_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    return await run_in_threadpool(datasets.list_datasets, project_id)


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(
    project_id: str, req: DatasetCreate, session: Session = Depends(get_session)
):
    _require_project(session, project_id)
    name = req.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    dataset_id = datasets.create(project_id, name)
    return datasets.to_out(project_id, dataset_id, datasets.read_meta(project_id, dataset_id) or {})


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    meta = _require_dataset(session, project_id, dataset_id)
    return datasets.to_out(project_id, dataset_id, meta)


@router.patch("/{dataset_id}", response_model=DatasetOut)
def rename_dataset(
    project_id: str,
    dataset_id: str,
    body: DatasetPatch,
    session: Session = Depends(get_session),
):
    _require_dataset(session, project_id, dataset_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    meta = datasets.rename(project_id, dataset_id, name)
    return datasets.to_out(project_id, dataset_id, meta)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    """데이터셋과 그 안의 이미지·라벨·클래스를 함께 지운다."""
    _require_dataset(session, project_id, dataset_id)
    datasets.delete(project_id, dataset_id)
