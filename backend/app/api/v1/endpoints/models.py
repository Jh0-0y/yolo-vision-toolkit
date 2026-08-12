"""Model registry: upload trained .pt files or download official pretrained models."""

from __future__ import annotations

import json
import shutil
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlmodel import Session, select

from app.core.config import settings
from app.db import get_session
from app.models import ModelEntry, iso_utc
from app.schemas.model import ModelOut, ModelPatch, OfficialRequest

router = APIRouter(prefix="/models", tags=["models"])

# Curated official detect models (baseline: yolo26n)
OFFICIAL_FAMILIES = ["yolo26", "yolo12", "yolo11"]
OFFICIAL_SIZES = ["n", "s", "m", "l", "x"]
DEFAULT_OFFICIAL = "yolo26n"


def _to_out(entry: ModelEntry) -> ModelOut:
    meta_path = settings.model_dir(entry.project_id, entry.id) / "meta.json"
    source = "upload"
    if meta_path.exists():
        source = json.loads(meta_path.read_text()).get("source", "upload")
    return ModelOut(
        id=entry.id,
        name=entry.name,
        classes={int(k): v for k, v in json.loads(entry.classes_json).items()},
        task=entry.task,
        created_at=iso_utc(entry.created_at),
        source=source,
    )


def _load_names(pt_path) -> tuple[dict[int, str], str]:
    """Load a .pt with ultralytics to validate it and read class names."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    return dict(model.names), getattr(model, "task", "detect") or "detect"


def _register(
    session: Session, name: str, pt_src, source: str, project_id: str | None = None
) -> ModelEntry:
    try:
        names, task = _load_names(pt_src)
    except Exception as e:
        raise HTTPException(422, f"Failed to load model: {e}")
    entry = ModelEntry(
        name=name, classes_json=json.dumps(names), task=task, project_id=project_id
    )
    model_dir = settings.model_dir(project_id, entry.id)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pt_src), model_dir / "model.pt")
    (model_dir / "meta.json").write_text(
        json.dumps({"name": name, "source": source, "classes": names}, ensure_ascii=False)
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


@router.get("", response_model=list[ModelOut])
def list_models(project_id: str | None = None, session: Session = Depends(get_session)):
    stmt = select(ModelEntry)
    if project_id is not None:
        # this project's models plus legacy/shared (project_id NULL)
        stmt = stmt.where(
            or_(ModelEntry.project_id == project_id, ModelEntry.project_id.is_(None))
        )
    entries = session.exec(stmt.order_by(ModelEntry.created_at.desc())).all()
    return [_to_out(e) for e in entries]


@router.get("/official")
def official_catalog():
    """Downloadable official pretrained detect models."""
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    catalog = []
    for family in OFFICIAL_FAMILIES:
        for size in OFFICIAL_SIZES:
            name = f"{family}{size}"
            if f"{name}.pt" in GITHUB_ASSETS_NAMES:
                catalog.append({"name": name, "default": name == DEFAULT_OFFICIAL})
    return catalog


@router.post("/official", response_model=ModelOut, status_code=201)
async def download_official(req: OfficialRequest, session: Session = Depends(get_session)):
    valid = {c["name"] for c in official_catalog()}
    if req.name not in valid:
        raise HTTPException(422, f"Unsupported model: {req.name}")

    def _download():
        from ultralytics.utils.downloads import attempt_download_asset

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        return attempt_download_asset(str(settings.models_dir / f"{req.name}.pt"))

    try:
        downloaded = await run_in_threadpool(_download)
    except Exception as e:
        raise HTTPException(502, f"Download failed: {e}")

    entry = _register(session, req.name, downloaded, source="official", project_id=req.project_id)
    return _to_out(entry)


@router.post("", response_model=ModelOut, status_code=201)
async def upload_model(
    file: UploadFile,
    project_id: str | None = None,
    session: Session = Depends(get_session),
):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(422, "Only .pt files can be uploaded")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.models_dir / f".upload_{file.filename}"
    with open(tmp, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    entry = _register(
        session, file.filename.removesuffix(".pt"), tmp, source="upload", project_id=project_id
    )
    return _to_out(entry)


@router.get("/{model_id}", response_model=ModelOut)
def get_model(model_id: str, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    return _to_out(entry)


@router.get("/{model_id}/download")
def download_model(model_id: str, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    pt = settings.model_dir(entry.project_id, model_id) / "model.pt"
    if not pt.exists():
        raise HTTPException(404, "Model file missing")
    # {date_time}-{name}.pt — timestamp keeps downloads from colliding on disk
    try:
        dt = entry.created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        stamp = dt.astimezone().strftime("%Y%m%d_%H%M%S")
    except Exception:
        stamp = entry.created_at.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in entry.name).strip("_") or model_id
    return FileResponse(pt, media_type="application/octet-stream", filename=f"{stamp}-{safe}.pt")


@router.patch("/{model_id}", response_model=ModelOut)
def rename_model(model_id: str, req: ModelPatch, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    name = req.name.strip()
    if not name:
        raise HTTPException(422, "Name is required")
    entry.name = name
    session.add(entry)
    session.commit()
    session.refresh(entry)

    meta_path = settings.model_dir(entry.project_id, model_id) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        meta["name"] = name
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    return _to_out(entry)


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: str, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    model_dir = settings.model_dir(entry.project_id, model_id)
    session.delete(entry)
    session.commit()
    shutil.rmtree(model_dir, ignore_errors=True)
