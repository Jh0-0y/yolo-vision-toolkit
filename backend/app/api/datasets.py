"""Dataset export: labeled images → train/val split + data.yaml zip, or a
plain zip of original images. Optionally restricted to a selected file list.
"""

from __future__ import annotations

import json
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
    kind: Literal["yolo", "images"] = "yolo"
    val_split: float = Field(default=0.2, ge=0.0, le=0.9)
    seed: int = 42
    # restrict to these file names (None = all eligible images)
    names: list[str] | None = None


class ExportOut(BaseModel):
    id: str
    kind: str = "yolo"
    created_at: str
    val_split: float = 0.0
    seed: int = 0
    train: int = 0
    val: int = 0
    count: int = 0
    classes: int = 0
    size_bytes: int = 0


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _export_meta_path(project_id: str, export_id: str) -> Path:
    return _project_dir(project_id) / "exports" / export_id / "export.json"


def _target_images(pdir: Path, req: ExportCreate) -> list[Path]:
    raw = pdir / "raw"
    images = sorted(
        p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS
    ) if raw.exists() else []
    if req.names is not None:
        wanted = set(req.names)
        images = [p for p in images if p.name in wanted]
    return images


def _build_export(project_id: str, req: ExportCreate) -> dict:
    pdir = _project_dir(project_id)
    labels_dir = pdir / "labels"
    images = _target_images(pdir, req)

    export_id = f"e_{uuid.uuid4().hex[:10]}"
    out = pdir / "exports" / export_id
    meta: dict = {
        "id": export_id,
        "kind": req.kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "val_split": 0.0,
        "seed": req.seed,
        "train": 0,
        "val": 0,
        "count": 0,
        "classes": 0,
    }

    if req.kind == "images":
        if not images:
            raise HTTPException(422, "No images to export")
        (out / "images").mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(img, out / "images" / img.name)
        meta["count"] = len(images)
    else:
        # yolo: only labeled images qualify
        images = [p for p in images if (labels_dir / f"{p.stem}.txt").exists()]
        if not images:
            raise HTTPException(422, "No labeled images. Label some images first.")

        rng = random.Random(req.seed)
        shuffled = images[:]
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * req.val_split)
        if req.val_split > 0 and n_val == 0 and len(shuffled) >= 2:
            n_val = 1
        val_set = set(shuffled[:n_val])

        counts = {"train": 0, "val": 0}
        for img in images:
            split = "val" if img in val_set else "train"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(labels_dir / f"{img.stem}.txt", out / "labels" / split / f"{img.stem}.txt")
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
        meta.update(
            val_split=req.val_split,
            train=counts["train"],
            val=counts["val"],
            count=len(images),
            classes=len(names),
        )

    zip_base = pdir / "exports" / export_id
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out))
    meta["size_bytes"] = zip_path.stat().st_size
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
        raise HTTPException(422, "Invalid id")
    zip_path = _project_dir(project_id) / "exports" / f"{export_id}.zip"
    if not zip_path.exists():
        raise HTTPException(404, "Export not found")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"dataset_{project_id}_{export_id}.zip",
    )


@router.delete("/{export_id}")
def delete_export(project_id: str, export_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    if "/" in export_id or export_id.startswith("."):
        raise HTTPException(422, "Invalid id")
    exports_dir = _project_dir(project_id) / "exports"
    shutil.rmtree(exports_dir / export_id, ignore_errors=True)
    (exports_dir / f"{export_id}.zip").unlink(missing_ok=True)
    return {"ok": True}
