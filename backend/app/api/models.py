"""Model registry: upload trained .pt files or download official pretrained models."""

from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session
from app.models import ModelEntry

router = APIRouter(prefix="/api/models", tags=["models"])

# Curated official detect models (baseline: yolo26n)
OFFICIAL_FAMILIES = ["yolo26", "yolo12", "yolo11"]
OFFICIAL_SIZES = ["n", "s", "m", "l", "x"]
DEFAULT_OFFICIAL = "yolo26n"


class ModelOut(BaseModel):
    id: str
    name: str
    classes: dict[int, str]
    task: str
    created_at: str
    source: str = "upload"


class OfficialRequest(BaseModel):
    name: str  # e.g. "yolo26n"


class ModelPatch(BaseModel):
    name: str


def _to_out(entry: ModelEntry) -> ModelOut:
    meta_path = settings.models_dir / entry.id / "meta.json"
    source = "upload"
    if meta_path.exists():
        source = json.loads(meta_path.read_text()).get("source", "upload")
    return ModelOut(
        id=entry.id,
        name=entry.name,
        classes={int(k): v for k, v in json.loads(entry.classes_json).items()},
        task=entry.task,
        created_at=entry.created_at.isoformat(),
        source=source,
    )


def _load_names(pt_path) -> tuple[dict[int, str], str]:
    """Load a .pt with ultralytics to validate it and read class names."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    return dict(model.names), getattr(model, "task", "detect") or "detect"


def _register(session: Session, name: str, pt_src, source: str) -> ModelEntry:
    try:
        names, task = _load_names(pt_src)
    except Exception as e:
        raise HTTPException(422, f"Failed to load model: {e}")
    entry = ModelEntry(name=name, classes_json=json.dumps(names), task=task)
    model_dir = settings.models_dir / entry.id
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
def list_models(session: Session = Depends(get_session)):
    entries = session.exec(select(ModelEntry).order_by(ModelEntry.created_at.desc())).all()
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


@router.post("/official", response_model=ModelOut)
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

    entry = _register(session, req.name, downloaded, source="official")
    return _to_out(entry)


@router.post("", response_model=ModelOut)
async def upload_model(file: UploadFile, session: Session = Depends(get_session)):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(422, "Only .pt files can be uploaded")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.models_dir / f".upload_{file.filename}"
    with open(tmp, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    entry = _register(session, file.filename.removesuffix(".pt"), tmp, source="upload")
    return _to_out(entry)


@router.get("/{model_id}", response_model=ModelOut)
def get_model(model_id: str, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    return _to_out(entry)


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

    meta_path = settings.models_dir / model_id / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        meta["name"] = name
        meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    return _to_out(entry)


@router.delete("/{model_id}")
def delete_model(model_id: str, session: Session = Depends(get_session)):
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    session.delete(entry)
    session.commit()
    shutil.rmtree(settings.models_dir / model_id, ignore_errors=True)
    return {"ok": True}
