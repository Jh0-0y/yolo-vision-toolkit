"""이미지 한 장의 라벨 편집 — 라벨 에디터가 쓴다.

라벨은 데이터셋 안에 있다. 클래스도 그 데이터셋 것이라, 같은 이미지를 다른
데이터셋에 넣었더라도 여기서 보는 클래스 목록은 이 데이터셋의 것이다.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.schemas.label import LabelsIn
from app.services import datasets
from lib.labels.store import read_boxes, write_boxes

router = APIRouter(
    prefix="/projects/{project_id}/datasets/{dataset_id}/labels", tags=["datasets"]
)


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


def _safe_stem(stem: str) -> str:
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "Invalid filename")
    return stem


def _image_name(ddir: Path, stem: str) -> str | None:
    for p in (ddir / "raw").glob(f"{stem}.*"):
        if p.suffix.lower() != ".json":
            return p.name
    return None


def _classes(ddir: Path) -> list[dict]:
    path = ddir / "classes.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("classes", [])
    except (json.JSONDecodeError, OSError):
        return []


@router.get("/{stem}")
def get_labels(
    project_id: str, dataset_id: str, stem: str, session: Session = Depends(get_session)
):
    ddir = _require_dataset(session, project_id, dataset_id)
    stem = _safe_stem(stem)
    name = _image_name(ddir, stem)
    if name is None:
        raise HTTPException(404, "Image not found")
    return {
        "stem": stem,
        "name": name,
        "image_url": (
            f"{settings.api_prefix}/files/projects/{project_id}"
            f"/datasets/{dataset_id}/raw/{name}"
        ),
        "boxes": read_boxes(ddir, stem),
        "classes": _classes(ddir),
        "reviewed": stem in datasets.read_reviewed(project_id, dataset_id),
    }


@router.put("/{stem}")
def put_labels(
    project_id: str,
    dataset_id: str,
    stem: str,
    body: LabelsIn,
    session: Session = Depends(get_session),
):
    ddir = _require_dataset(session, project_id, dataset_id)
    stem = _safe_stem(stem)
    if _image_name(ddir, stem) is None:
        raise HTTPException(404, "Image not found")
    write_boxes(ddir, stem, [b.model_dump() for b in body.boxes])
    return {"ok": True, "count": len(body.boxes)}
