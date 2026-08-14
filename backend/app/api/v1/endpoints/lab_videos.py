"""연구실 영상 아카이브 — 크롭을 돌릴 원본을 모아 두는 곳.

학습실 영상(`videos.py`)과 달리 **프레임 추출을 하지 않는다.** 여기 영상은 학습
재료가 아니라 크롭해서 내보낼 대상이라, 올려 두고 고르기만 하면 된다.

연구실이 하나뿐이라 경로에 id 가 없다 — 어느 연구실인지는 서버가 안다.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.labs import the_lab
from app.db import get_session
from app.schemas.lab import LabVideoOut, LabVideoPatch
from app.services import lab_store
from lib.formats import VIDEO_EXTS

router = APIRouter(prefix="/lab/videos", tags=["lab"])


def _out(lab_id: str, meta: dict) -> LabVideoOut:
    return LabVideoOut(
        id=meta["id"],
        name=meta.get("name") or meta.get("filename", meta["id"]),
        filename=meta.get("filename", ""),
        ext=meta.get("ext", ""),
        size_bytes=meta.get("size_bytes", 0),
        width=meta.get("width", 0),
        height=meta.get("height", 0),
        fps=meta.get("fps", 0.0),
        duration_ms=meta.get("duration_ms", 0),
        created_at=meta.get("created_at", ""),
        run_count=lab_store.count_runs_for_video(lab_id, meta["id"]),
    )


def _require_video(session: Session, video_id: str) -> tuple[str, dict]:
    lab_id = the_lab(session).id
    if not lab_store.valid_id(video_id):
        raise HTTPException(422, "Invalid video id")
    meta = lab_store.read_video_meta(lab_id, video_id)
    if meta is None:
        raise HTTPException(404, "Video not found")
    return lab_id, meta


@router.get("", response_model=list[LabVideoOut])
async def list_lab_videos(session: Session = Depends(get_session)):
    lab_id = the_lab(session).id
    metas = await run_in_threadpool(lab_store.list_videos, lab_id)
    return [_out(lab_id, m) for m in metas]


@router.post("", response_model=LabVideoOut, status_code=201)
async def upload_lab_video(file: UploadFile, session: Session = Depends(get_session)):
    lab_id = the_lab(session).id
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            422, f"Unsupported video format: {ext or 'none'} ({', '.join(sorted(VIDEO_EXTS))})"
        )

    lab_store.ensure_dirs(lab_id)
    video_id = lab_store.new_video_id()
    path = lab_store.videos_dir(lab_id) / f"{video_id}{ext}"
    try:
        with open(path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

        probed = await run_in_threadpool(lab_store.probe_meta, path)
        meta = lab_store.new_video_meta(video_id, filename, ext, probed)
        lab_store.write_video_meta(lab_id, video_id, meta)
    except Exception:
        # 사이드카가 없으면 목록에 뜨지 않는다 — 지우지 않으면 아무도 모르는 채로
        # 기가 단위 파일이 쌓인다. 실제로 그렇게 쌓였다.
        path.unlink(missing_ok=True)
        lab_store.video_meta_path(lab_id, video_id).unlink(missing_ok=True)
        raise
    meta["size_bytes"] = path.stat().st_size
    return _out(lab_id, meta)


@router.get("/{video_id}/file")
def download_lab_video(video_id: str, session: Session = Depends(get_session)):
    lab_id, meta = _require_video(session, video_id)
    path = lab_store.video_file(lab_id, video_id)
    if path is None:
        raise HTTPException(404, "Video file missing")
    # FileResponse 가 Range 를 처리한다 — 브라우저가 탐색하며 재생할 수 있어야 한다
    return FileResponse(path, media_type="video/mp4", filename=meta.get("filename"))


@router.patch("/{video_id}", response_model=LabVideoOut)
def rename_lab_video(
    video_id: str,
    body: LabVideoPatch,
    session: Session = Depends(get_session),
):
    lab_id, meta = _require_video(session, video_id)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Name cannot be empty")
    meta["name"] = name
    lab_store.write_video_meta(lab_id, video_id, meta)
    path = lab_store.video_file(lab_id, video_id)
    meta["size_bytes"] = path.stat().st_size if path else 0
    return _out(lab_id, meta)


@router.delete("/{video_id}", status_code=204)
def delete_lab_video(video_id: str, session: Session = Depends(get_session)):
    lab_id, _ = _require_video(session, video_id)
    lab_store.delete_video(lab_id, video_id)
