"""Dataset export: confirmed/ → train/val split + data.yaml + downloadable zip."""

from __future__ import annotations

import json
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import settings
from app.core.inference import IMAGE_EXTS
from app.core.yolo_io import atomic_write_text, write_data_yaml
from app.db import get_session
from app.models import Project

router = APIRouter(prefix="/api/projects/{project_id}/exports", tags=["exports"])


class ExportCreate(BaseModel):
    val_split: float = Field(default=0.2, ge=0.0, le=0.9)
    seed: int = 42


class ExportOut(BaseModel):
    id: str
    created_at: str
    val_split: float
    seed: int
    train: int
    val: int
    classes: int
    size_bytes: int


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "프로젝트가 없습니다")


def _export_meta_path(project_id: str, export_id: str) -> Path:
    return _project_dir(project_id) / "exports" / export_id / "export.json"


def _build_export(project_id: str, req: ExportCreate) -> dict:
    pdir = _project_dir(project_id)
    images_dir = pdir / "confirmed" / "images"
    labels_dir = pdir / "confirmed" / "labels"

    images = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    ) if images_dir.exists() else []
    if not images:
        raise HTTPException(422, "확정된 이미지가 없습니다. 먼저 라벨링/리뷰를 진행하세요.")

    rng = random.Random(req.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * req.val_split)
    if req.val_split > 0 and n_val == 0 and len(shuffled) >= 2:
        n_val = 1
    val_set = set(shuffled[:n_val])

    export_id = f"e_{uuid.uuid4().hex[:10]}"
    out = pdir / "exports" / export_id
    counts = {"train": 0, "val": 0}
    for img in images:
        split = "val" if img in val_set else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / split / img.name)
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, out / "labels" / split / label.name)
        else:
            (out / "labels" / split / f"{img.stem}.txt").write_text("")
        counts[split] += 1

    classes_path = pdir / "classes.json"
    names: dict[int, str] = {}
    if classes_path.exists():
        for c in json.loads(classes_path.read_text()).get("classes", []):
            names[int(c["id"])] = c["name"]

    write_data_yaml(
        out / "data.yaml",
        names,
        train="images/train",
        val="images/val" if counts["val"] else "images/train",
    )

    zip_base = pdir / "exports" / export_id
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out))

    meta = {
        "id": export_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "val_split": req.val_split,
        "seed": req.seed,
        "train": counts["train"],
        "val": counts["val"],
        "classes": len(names),
        "size_bytes": zip_path.stat().st_size,
    }
    atomic_write_text(_export_meta_path(project_id, export_id), json.dumps(meta))
    return meta


@router.post("", response_model=ExportOut)
async def create_export(
    project_id: str, req: ExportCreate, session: Session = Depends(get_session)
):
    _require_project(session, project_id)
    return await run_in_threadpool(_build_export, project_id, req)


@router.get("", response_model=list[ExportOut])
def list_exports(project_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    exports_dir = _project_dir(project_id) / "exports"
    metas = []
    if exports_dir.exists():
        for meta_path in exports_dir.glob("*/export.json"):
            try:
                metas.append(json.loads(meta_path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return metas


@router.get("/{export_id}/download")
def download_export(project_id: str, export_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    if "/" in export_id or export_id.startswith("."):
        raise HTTPException(422, "잘못된 id입니다")
    zip_path = _project_dir(project_id) / "exports" / f"{export_id}.zip"
    if not zip_path.exists():
        raise HTTPException(404, "내보내기가 없습니다")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"dataset_{project_id}_{export_id}.zip",
    )


@router.delete("/{export_id}")
def delete_export(project_id: str, export_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    if "/" in export_id or export_id.startswith("."):
        raise HTTPException(422, "잘못된 id입니다")
    exports_dir = _project_dir(project_id) / "exports"
    shutil.rmtree(exports_dir / export_id, ignore_errors=True)
    (exports_dir / f"{export_id}.zip").unlink(missing_ok=True)
    return {"ok": True}
