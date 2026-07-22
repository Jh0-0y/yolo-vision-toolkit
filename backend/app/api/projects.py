"""Projects: CRUD, image ingest (files or zip), paginated listing, stats."""

from __future__ import annotations

import json
import shutil
import zipfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.core.inference import IMAGE_EXTS
from app.db import get_session
from app.models import Project, ReviewItem

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str


class StatsOut(BaseModel):
    raw: int
    confirmed: int
    review: int
    review_pending: int
    classes: list[dict]


def _project_dir(project_id: str):
    return settings.projects_dir / project_id


def _get_project(session: Session, project_id: str) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "프로젝트가 없습니다")
    return project


def _count_images(d) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [ProjectOut(id=p.id, name=p.name, created_at=p.created_at.isoformat()) for p in projects]


@router.post("", response_model=ProjectOut)
def create_project(req: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=req.name)
    pdir = _project_dir(project.id)
    for sub in ("raw", "thumbs", "confirmed/images", "confirmed/labels", "review", "rejected", "exports"):
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
    for item in session.exec(select(ReviewItem).where(ReviewItem.project_id == project_id)):
        session.delete(item)
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
                raise HTTPException(422, f"zip 파일이 손상되었습니다: {name}")
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
    bucket: str = "raw",
    page: int = 1,
    size: int = 100,
    session: Session = Depends(get_session),
):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)
    if bucket == "raw":
        files = sorted(p.name for p in (pdir / "raw").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif bucket == "confirmed":
        files = sorted(
            p.name for p in (pdir / "confirmed" / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
    elif bucket == "review":
        stems = {p.stem for p in (pdir / "review").glob("*.json")}
        files = sorted(
            p.name for p in (pdir / "raw").iterdir() if p.stem in stems
        )
    else:
        raise HTTPException(422, f"알 수 없는 bucket: {bucket}")

    total = len(files)
    start = (page - 1) * size
    items = [
        {
            "name": name,
            "thumb": f"/api/files/projects/{project_id}/thumbs/{name}",
            "url": f"/api/files/projects/{project_id}/raw/{name}",
        }
        for name in files[start : start + size]
    ]
    return {"total": total, "page": page, "size": size, "items": items}


@router.get("/{project_id}/stats", response_model=StatsOut)
def project_stats(project_id: str, session: Session = Depends(get_session)):
    _get_project(session, project_id)
    pdir = _project_dir(project_id)

    classes: list[dict] = []
    classes_path = pdir / "classes.json"
    if classes_path.exists():
        classes = json.loads(classes_path.read_text()).get("classes", [])

    pending = len(
        session.exec(
            select(ReviewItem).where(
                ReviewItem.project_id == project_id, ReviewItem.status == "pending"
            )
        ).all()
    )
    return StatsOut(
        raw=_count_images(pdir / "raw"),
        confirmed=_count_images(pdir / "confirmed" / "images"),
        review=len(list((pdir / "review").glob("*.json"))),
        review_pending=pending,
        classes=classes,
    )
