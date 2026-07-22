"""Projects: CRUD, image ingest, unified image listing (filter/sort/search), stats."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.core.inference import IMAGE_EXTS
from app.core.labels import (
    label_classes,
    label_path,
    read_boxes,
    read_reviewed,
    set_reviewed,
    write_reviewed,
)
from app.db import get_session
from app.models import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])

PROJECT_SUBDIRS = ("raw", "thumbs", "videos", "labels", "exports")


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str


class StatsOut(BaseModel):
    images: int
    labeled: int
    reviewed: int
    classes: list[dict]


class ReviewedIn(BaseModel):
    reviewed: bool


class DeleteImagesIn(BaseModel):
    names: list[str]


def _project_dir(project_id: str):
    return settings.projects_dir / project_id


def _get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


def _raw_images(pdir: Path) -> list[Path]:
    raw = pdir / "raw"
    if not raw.exists():
        return []
    return [p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def _labeled_stems(pdir: Path) -> set[str]:
    labels = pdir / "labels"
    if not labels.exists():
        return set()
    return {p.stem for p in labels.glob("*.txt")}


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [ProjectOut(id=p.id, name=p.name, created_at=p.created_at.isoformat()) for p in projects]


@router.post("", response_model=ProjectOut)
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
    return ProjectOut(id=project.id, name=project.name, created_at=project.created_at.isoformat())


@router.delete("/{project_id}")
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = _get_project(session, project_id)
    session.delete(project)
    session.commit()
    shutil.rmtree(_project_dir(project_id), ignore_errors=True)
    return {"ok": True}


@router.post("/{project_id}/images")
async def upload_images(
    project_id: str,
    files: list[UploadFile],
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    raw_dir = _project_dir(project_id) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    for file in files:
        name = (file.filename or "").rsplit("/", 1)[-1]
        suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        if suffix == ".zip":
            tmp = raw_dir / f".upload_{name}"
            with open(tmp, "wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            try:
                with zipfile.ZipFile(tmp) as zf:
                    for info in zf.infolist():
                        base = info.filename.rsplit("/", 1)[-1]
                        ext = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
                        if info.is_dir() or ext not in IMAGE_EXTS or base.startswith("."):
                            skipped += 1
                            continue
                        with zf.open(info) as src, open(raw_dir / base, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        added += 1
            except zipfile.BadZipFile:
                raise HTTPException(422, f"Corrupted zip file: {name}")
            finally:
                tmp.unlink(missing_ok=True)
        elif suffix in IMAGE_EXTS:
            with open(raw_dir / name, "wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            added += 1
        else:
            skipped += 1
    return {"added": added, "skipped": skipped}


@router.get("/{project_id}/images")
def list_images(
    project_id: str,
    labeled: bool | None = None,
    reviewed: bool | None = None,
    cls: int | None = None,
    sort: str = "created",
    order: str = "desc",
    q: str = "",
    page: int = 1,
    size: int = 60,
    names_only: bool = False,
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)
    if sort not in ("created", "name"):
        raise HTTPException(422, f"Unknown sort: {sort}")

    labeled_stems = _labeled_stems(pdir)
    reviewed_stems = read_reviewed(pdir)

    images = _raw_images(pdir)
    if q:
        needle = q.lower()
        images = [p for p in images if needle in p.name.lower()]
    if labeled is not None:
        images = [p for p in images if (p.stem in labeled_stems) == labeled]
    if reviewed is not None:
        images = [p for p in images if (p.stem in reviewed_stems) == reviewed]
    if cls is not None:
        if cls == -1:  # "no classes": labeled but empty (negative)
            images = [
                p
                for p in images
                if p.stem in labeled_stems and not label_classes(pdir, p.stem)
            ]
        else:
            images = [
                p
                for p in images
                if p.stem in labeled_stems and cls in label_classes(pdir, p.stem)
            ]

    reverse = order == "desc"
    if sort == "created":
        entries = sorted(
            ((p, p.stat().st_mtime) for p in images),
            key=lambda e: (e[1], e[0].name),
            reverse=reverse,
        )
    else:
        entries = sorted(
            ((p, p.stat().st_mtime) for p in images),
            key=lambda e: e[0].name.lower(),
            reverse=reverse,
        )

    total = len(entries)
    if names_only:
        return {"total": total, "names": [p.name for p, _ in entries]}

    start = (page - 1) * size
    page_entries = entries[start : start + size]
    items = [
        {
            "name": p.name,
            "stem": p.stem,
            "thumb": f"/api/files/projects/{project_id}/thumbs/{p.name}",
            "url": f"/api/files/projects/{project_id}/raw/{p.name}",
            "labeled": p.stem in labeled_stems,
            "reviewed": p.stem in reviewed_stems,
            "boxes": read_boxes(pdir, p.stem) if p.stem in labeled_stems else [],
            "created_at": mtime,
        }
        for p, mtime in page_entries
    ]
    return {"total": total, "page": page, "size": size, "items": items}


@router.post("/{project_id}/images/delete")
def delete_images(
    project_id: str,
    body: DeleteImagesIn,
    session: Session = Depends(get_session),
):
    """Delete raw images and everything attached to them (thumb, label, reviewed flag)."""
    _get_project(session, project_id)
    pdir = _project_dir(project_id)
    reviewed = read_reviewed(pdir)
    deleted = 0
    for name in body.names:
        base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base.startswith(".") or Path(base).suffix.lower() not in IMAGE_EXTS:
            continue
        raw = pdir / "raw" / base
        if not raw.exists():
            continue
        stem = raw.stem
        raw.unlink()
        (pdir / "thumbs" / f"{stem}.jpg").unlink(missing_ok=True)
        label_path(pdir, stem).unlink(missing_ok=True)
        reviewed.discard(stem)
        deleted += 1
    write_reviewed(pdir, reviewed)
    return {"deleted": deleted}


@router.put("/{project_id}/images/{stem}/reviewed")
def put_reviewed(
    project_id: str,
    stem: str,
    body: ReviewedIn,
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    if "/" in stem or "\\" in stem or stem.startswith("."):
        raise HTTPException(422, "Invalid filename")
    pdir = _project_dir(project_id)
    if not any(p.stem == stem for p in _raw_images(pdir)):
        raise HTTPException(404, "Image not found")
    set_reviewed(pdir, stem, body.reviewed)
    return {"ok": True, "reviewed": body.reviewed}


@router.get("/{project_id}/stats", response_model=StatsOut)
def project_stats(project_id: str, session: Session = Depends(get_session)):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)

    classes: list[dict] = []
    classes_path = pdir / "classes.json"
    if classes_path.exists():
        classes = json.loads(classes_path.read_text()).get("classes", [])

    stems = {p.stem for p in _raw_images(pdir)}
    labeled = stems & _labeled_stems(pdir)
    reviewed = stems & read_reviewed(pdir)
    return StatsOut(
        images=len(stems),
        labeled=len(labeled),
        reviewed=len(reviewed),
        classes=classes,
    )
