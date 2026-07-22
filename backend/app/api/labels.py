"""Per-image label editing for the confirmed/review buckets (used by the grid edit modal)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import settings
from app.core.labels import BUCKETS, read_boxes, write_boxes
from app.db import get_session
from app.models import Project

router = APIRouter(prefix="/api/projects/{project_id}/labels", tags=["labels"])


class BoxIn(BaseModel):
    id: str | None = None
    cls: int
    xyxy_n: list[float] = Field(min_length=4, max_length=4)
    score: float | None = None
    status: str | None = None
    reason: str | None = None
    sources: list[dict] | None = None


class LabelsIn(BaseModel):
    boxes: list[BoxIn]


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "프로젝트가 없습니다")


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _safe_stem(stem: str) -> str:
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "잘못된 파일명입니다")
    return stem


def _image_name(pdir: Path, bucket: str, stem: str) -> str | None:
    search = (pdir / "confirmed" / "images") if bucket == "confirmed" else (pdir / "raw")
    for p in search.glob(f"{stem}.*"):
        if p.suffix.lower() != ".json":
            return p.name
    return None


def _classes(pdir: Path) -> list[dict]:
    path = pdir / "classes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("classes", [])


@router.get("/{bucket}/{stem}")
def get_labels(
    project_id: str, bucket: str, stem: str, session: Session = Depends(get_session)
):
    _require_project(session, project_id)
    if bucket not in BUCKETS:
        raise HTTPException(422, f"알 수 없는 bucket: {bucket}")
    stem = _safe_stem(stem)
    pdir = _project_dir(project_id)
    name = _image_name(pdir, bucket, stem)
    if name is None:
        raise HTTPException(404, "이미지가 없습니다")
    return {
        "stem": stem,
        "bucket": bucket,
        "image_url": f"/api/files/projects/{project_id}/raw/{name}",
        "boxes": read_boxes(pdir, bucket, stem),
        "classes": _classes(pdir),
    }


@router.put("/{bucket}/{stem}")
def put_labels(
    project_id: str,
    bucket: str,
    stem: str,
    body: LabelsIn,
    session: Session = Depends(get_session),
):
    _require_project(session, project_id)
    if bucket not in BUCKETS:
        raise HTTPException(422, f"알 수 없는 bucket: {bucket}")
    stem = _safe_stem(stem)
    pdir = _project_dir(project_id)
    if _image_name(pdir, bucket, stem) is None:
        raise HTTPException(404, "이미지가 없습니다")
    write_boxes(pdir, bucket, stem, [b.model_dump() for b in body.boxes])
    return {"ok": True, "count": len(body.boxes)}
