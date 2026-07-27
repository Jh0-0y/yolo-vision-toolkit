"""Dataset export: labeled images → train/val split + data.yaml zip, or a
plain zip of original images. Optionally restricted to a selected file list.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from app.schemas.export import ExportCreate, ExportOut, ExportRename
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.domain.export_build import safe_token, target_images
from app.domain.yolo_io import atomic_write_text
from app.db import get_session
from app.models import Project
from app.services.export_manager import export_manager, task_dir as export_task_dir
from app.services.label_manager import read_progress

router = APIRouter(prefix="/projects/{project_id}/exports", tags=["exports"])

TERMINAL_PHASES = {"done", "error", "cancelled"}


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _export_meta_path(project_id: str, export_id: str) -> Path:
    return _project_dir(project_id) / "exports" / export_id / "export.json"


@router.post("", status_code=201)
def create_export(
    project_id: str, req: ExportCreate, session: Session = Depends(get_session)
):
    """Start an export as a background job. Validates synchronously (422 on no
    eligible images), then streams per-image progress via GET .../{id}/events.
    The finished export appears in GET /exports once its job reaches 'done'."""
    _require_project(session, project_id)
    project = session.get(Project, project_id)
    project_name = project.name if project else project_id
    pdir = _project_dir(project_id)

    images = target_images(pdir, req.names, req.kind)
    if not images:
        raise HTTPException(
            422,
            "No images to export"
            if req.kind == "images"
            else "No labeled images. Label some images first.",
        )

    export_id = f"e_{uuid.uuid4().hex[:10]}"
    export_manager.submit(
        export_id, pdir, project_name, req.kind, req.names, req.val_split, req.seed
    )
    return {"export_id": export_id, "status": "running"}


@router.get("/{export_id}/events")
async def export_events(
    project_id: str, export_id: str, session: Session = Depends(get_session)
):
    """Stream export build progress (start → copy → zip → done/error)."""
    _require_project(session, project_id)
    if not (export_task_dir(export_id) / "progress.jsonl").exists():
        raise HTTPException(404, "Export task not found")

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, export_id, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL_PHASES:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.3)

    return EventSourceResponse(stream())


@router.get("", response_model=list[ExportOut])
def list_exports(project_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    exports_dir = _project_dir(project_id) / "exports"
    metas = []
    if exports_dir.exists():
        for meta_path in exports_dir.glob("*/export.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            meta.setdefault("name", meta.get("id", ""))  # older exports had no name
            metas.append(meta)
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
    meta_path = _export_meta_path(project_id, export_id)
    name = export_id
    if meta_path.exists():
        try:
            name = json.loads(meta_path.read_text()).get("name", export_id)
        except (json.JSONDecodeError, OSError):
            pass
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{safe_token(name)}.zip",
    )


@router.patch("/{export_id}", response_model=ExportOut)
def rename_export(
    project_id: str,
    export_id: str,
    body: ExportRename,
    session: Session = Depends(get_session),
):
    _require_project(session, project_id)
    if "/" in export_id or export_id.startswith("."):
        raise HTTPException(422, "Invalid id")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    meta_path = _export_meta_path(project_id, export_id)
    if not meta_path.exists():
        raise HTTPException(404, "Export not found")
    meta = json.loads(meta_path.read_text())
    meta["name"] = name
    atomic_write_text(meta_path, json.dumps(meta))
    return meta


@router.delete("/{export_id}", status_code=204)
def delete_export(project_id: str, export_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    if "/" in export_id or export_id.startswith("."):
        raise HTTPException(422, "Invalid id")
    exports_dir = _project_dir(project_id) / "exports"
    shutil.rmtree(exports_dir / export_id, ignore_errors=True)
    (exports_dir / f"{export_id}.zip").unlink(missing_ok=True)
