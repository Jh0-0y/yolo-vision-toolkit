"""Per-project class management (add/rename/delete)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.schemas.class_ import ClassIn
from lib.labels.classes import (
    add_class,
    count_boxes_with_class,
    delete_class,
    read_classes,
    rename_class,
)

router = APIRouter(prefix="/projects/{project_id}/classes", tags=["classes"])


def _require_project(session: Session, project_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    return settings.projects_dir / project_id


@router.get("")
def list_classes(project_id: str, session: Session = Depends(get_session)):
    pdir = _require_project(session, project_id)
    return read_classes(pdir)


@router.post("", status_code=201)
def create_class(project_id: str, body: ClassIn, session: Session = Depends(get_session)):
    pdir = _require_project(session, project_id)
    try:
        return add_class(pdir, body.name)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.patch("/{class_id}")
def patch_class(
    project_id: str, class_id: int, body: ClassIn, session: Session = Depends(get_session)
):
    pdir = _require_project(session, project_id)
    try:
        return rename_class(pdir, class_id, body.name)
    except KeyError:
        raise HTTPException(404, "Class not found")
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.delete("/{class_id}")
def remove_class(project_id: str, class_id: int, session: Session = Depends(get_session)):
    pdir = _require_project(session, project_id)
    removed_boxes = count_boxes_with_class(pdir, class_id)
    try:
        classes = delete_class(pdir, class_id)
    except KeyError:
        raise HTTPException(404, "Class not found")
    return {"ok": True, "removed_boxes": removed_boxes, "classes": classes}
