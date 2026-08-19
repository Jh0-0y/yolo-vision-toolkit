"""데이터셋의 클래스 관리 (추가·이름변경·삭제).

클래스는 **데이터셋마다 다르다.** "공만" 데이터셋과 "선수만" 데이터셋이 서로 다른
목록을 갖는다. 그래서 자리도 그 데이터셋 안(`classes.json`)이다.

id 는 라벨 파일이 위치로 참조하므로 이어지는 번호로 두고, 삭제는 그 클래스의 박스를
버리고 뒤 번호를 당긴다 — 이 데이터셋의 라벨만 손대므로 다른 데이터셋은 안 흔들린다.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db import get_session
from app.models import Project
from app.schemas.class_ import ClassIn
from app.services import datasets
from lib.labels.classes import (
    add_class,
    count_boxes_with_class,
    delete_class,
    read_classes,
    rename_class,
)

router = APIRouter(
    prefix="/projects/{project_id}/datasets/{dataset_id}/classes", tags=["datasets"]
)


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


@router.get("")
def list_classes(project_id: str, dataset_id: str, session: Session = Depends(get_session)):
    return read_classes(_require_dataset(session, project_id, dataset_id))


@router.post("", status_code=201)
def create_class(
    project_id: str, dataset_id: str, body: ClassIn, session: Session = Depends(get_session)
):
    ddir = _require_dataset(session, project_id, dataset_id)
    try:
        return add_class(ddir, body.name)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.patch("/{class_id}")
def patch_class(
    project_id: str,
    dataset_id: str,
    class_id: int,
    body: ClassIn,
    session: Session = Depends(get_session),
):
    ddir = _require_dataset(session, project_id, dataset_id)
    try:
        return rename_class(ddir, class_id, body.name)
    except KeyError:
        raise HTTPException(404, "Class not found")
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.delete("/{class_id}")
def remove_class(
    project_id: str, dataset_id: str, class_id: int, session: Session = Depends(get_session)
):
    """이 클래스의 박스를 버리고 뒤 번호를 당긴다 — 이 데이터셋의 라벨만."""
    ddir = _require_dataset(session, project_id, dataset_id)
    removed_boxes = count_boxes_with_class(ddir, class_id)
    try:
        classes = delete_class(ddir, class_id)
    except KeyError:
        raise HTTPException(404, "Class not found")
    return {"ok": True, "removed_boxes": removed_boxes, "classes": classes}
