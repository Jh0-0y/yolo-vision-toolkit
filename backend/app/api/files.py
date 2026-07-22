"""Image and thumbnail serving with path-traversal protection."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.core.thumbnails import get_thumbnail

router = APIRouter(prefix="/api/files", tags=["files"])


def _safe_name(name: str) -> str:
    clean = Path(name).name
    if clean != name or not clean or clean.startswith("."):
        raise HTTPException(422, "잘못된 파일명입니다")
    return clean


@router.get("/projects/{project_id}/raw/{name}")
def raw_image(project_id: str, name: str):
    path = settings.projects_dir / _safe_name(project_id) / "raw" / _safe_name(name)
    if not path.exists():
        raise HTTPException(404, "파일이 없습니다")
    return FileResponse(path)


@router.get("/projects/{project_id}/thumbs/{name}")
def thumbnail(project_id: str, name: str):
    pdir = settings.projects_dir / _safe_name(project_id)
    src = pdir / "raw" / _safe_name(name)
    if not src.exists():
        raise HTTPException(404, "파일이 없습니다")
    try:
        thumb = get_thumbnail(src, pdir / "thumbs")
    except OSError as e:
        raise HTTPException(422, f"썸네일 생성 실패: {e}")
    return FileResponse(thumb, media_type="image/jpeg")
