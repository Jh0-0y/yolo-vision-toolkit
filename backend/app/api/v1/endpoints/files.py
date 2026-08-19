"""Image and thumbnail serving with path-traversal protection."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from lib.media.thumbnails import get_thumbnail

router = APIRouter(prefix="/files", tags=["files"])


def _safe_name(name: str) -> str:
    clean = Path(name).name
    if clean != name or not clean or clean.startswith("."):
        raise HTTPException(422, "Invalid filename")
    return clean


def _dataset_dir(project_id: str, dataset_id: str) -> Path:
    return (
        settings.projects_dir
        / _safe_name(project_id)
        / "datasets"
        / _safe_name(dataset_id)
    )


@router.get("/projects/{project_id}/datasets/{dataset_id}/raw/{name}")
def dataset_raw_image(project_id: str, dataset_id: str, name: str):
    path = _dataset_dir(project_id, dataset_id) / "raw" / _safe_name(name)
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@router.get("/projects/{project_id}/datasets/{dataset_id}/thumbs/{name}")
def dataset_thumbnail(project_id: str, dataset_id: str, name: str):
    """썸네일은 **요청할 때 만든다** — 가져오기가 미리 만들어 둘 필요가 없다."""
    ddir = _dataset_dir(project_id, dataset_id)
    src = ddir / "raw" / _safe_name(name)
    if not src.exists():
        raise HTTPException(404, "File not found")
    try:
        thumb = get_thumbnail(src, ddir / "thumbs")
    except OSError as e:
        raise HTTPException(422, f"Thumbnail generation failed: {e}")
    return FileResponse(thumb, media_type="image/jpeg")


