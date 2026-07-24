"""Per-image label editing (used by the label editor page)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from app.schemas.label import LabelsIn
from sqlmodel import Session

from app.core.config import settings
from app.domain.labels import read_boxes, read_reviewed, write_boxes
from app.db import get_session
from app.models import Project

router = APIRouter(prefix="/projects/{project_id}/labels", tags=["labels"])


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _safe_stem(stem: str) -> str:
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "Invalid filename")
    return stem


def _image_name(pdir: Path, stem: str) -> str | None:
    for p in (pdir / "raw").glob(f"{stem}.*"):
        if p.suffix.lower() != ".json":
            return p.name
    return None


def _classes(pdir: Path) -> list[dict]:
    path = pdir / "classes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("classes", [])


@router.get("/{stem}")
def get_labels(project_id: str, stem: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    stem = _safe_stem(stem)
    pdir = _project_dir(project_id)
    name = _image_name(pdir, stem)
    if name is None:
        raise HTTPException(404, "Image not found")
    return {
        "stem": stem,
        "name": name,
        "image_url": f"{settings.api_prefix}/files/projects/{project_id}/raw/{name}",
        "boxes": read_boxes(pdir, stem),
        "classes": _classes(pdir),
        "reviewed": stem in read_reviewed(pdir),
    }


@router.put("/{stem}")
def put_labels(
    project_id: str,
    stem: str,
    body: LabelsIn,
    session: Session = Depends(get_session),
):
    _require_project(session, project_id)
    stem = _safe_stem(stem)
    pdir = _project_dir(project_id)
    if _image_name(pdir, stem) is None:
        raise HTTPException(404, "Image not found")
    write_boxes(pdir, stem, [b.model_dump() for b in body.boxes])
    return {"ok": True, "count": len(body.boxes)}
