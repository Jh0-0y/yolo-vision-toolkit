"""Video ingestion: upload a video, sample frames into raw/, stream progress."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session

from app.core.config import settings
from app.domain.tiling import TilingParams
from app.domain.video import VIDEO_EXTS, ExtractParams
from app.db import get_session
from app.services.label_manager import read_progress
from app.services.video_manager import task_dir, video_manager
from app.models import Project

router = APIRouter(prefix="/projects/{project_id}/videos", tags=["videos"])

TERMINAL_PHASES = {"done", "error", "cancelled"}


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", stem).strip("_")
    return stem or "video"


def _params(
    target_fps: float,
    max_frames: int,
    start_sec: float,
    end_sec: float | None,
    dedup: bool,
    dedup_threshold: float,
    tile: bool = False,
    tile_size: int = 640,
    stride: int = 480,
) -> ExtractParams:
    if target_fps <= 0:
        raise HTTPException(422, "target_fps must be greater than 0")
    if max_frames <= 0:
        raise HTTPException(422, "max_frames must be greater than 0")
    if tile:
        errors = TilingParams(tile_size=tile_size, stride=stride).validate()
        if errors:
            raise HTTPException(422, f"Invalid tiling params: {errors}")
    return ExtractParams(
        target_fps=target_fps,
        max_frames=max_frames,
        start_sec=max(0.0, start_sec),
        end_sec=end_sec,
        dedup=dedup,
        dedup_threshold=dedup_threshold,
        tile=tile,
        tile_size=tile_size,
        stride=stride,
    )


@router.post("", status_code=201)
async def upload_video(
    project_id: str,
    file: UploadFile,
    target_fps: float = Form(2.0),
    max_frames: int = Form(2000),
    start_sec: float = Form(0.0),
    end_sec: float | None = Form(None),
    dedup: bool = Form(True),
    dedup_threshold: float = Form(0.92),
    tile: bool = Form(False),  # 프레임을 학습용 타일로 쪼개 저장
    tile_size: int = Form(640),
    stride: int = Form(480),  # 겹침 = tile_size - stride
    session: Session = Depends(get_session),
):
    _require_project(session, project_id)
    params = _params(
        target_fps, max_frames, start_sec, end_sec, dedup, dedup_threshold,
        tile, tile_size, stride,
    )

    name = (file.filename or "").rsplit("/", 1)[-1]
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            422, f"Unsupported video format: {ext or 'none'} ({', '.join(sorted(VIDEO_EXTS))})"
        )

    video_id = f"vid_{uuid.uuid4().hex[:10]}"
    videos_dir = _project_dir(project_id) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"{video_id}{ext}"
    with open(video_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    stem = _sanitize_stem(name)
    (videos_dir / f"{video_id}.json").write_text(
        json.dumps(
            {"id": video_id, "filename": name, "ext": ext, "stem": stem, "params": params.__dict__},
            ensure_ascii=False,
        )
    )

    video_manager.submit(
        video_id, video_path, _project_dir(project_id) / "raw", stem, params
    )
    return {"video_id": video_id, "filename": name, "status": "running"}


@router.post("/{video_id}/resample")
def resample_video(
    project_id: str,
    video_id: str,
    target_fps: float = Form(2.0),
    max_frames: int = Form(2000),
    start_sec: float = Form(0.0),
    end_sec: float | None = Form(None),
    dedup: bool = Form(True),
    dedup_threshold: float = Form(0.92),
    tile: bool = Form(False),  # 프레임을 학습용 타일로 쪼개 저장
    tile_size: int = Form(640),
    stride: int = Form(480),  # 겹침 = tile_size - stride
    session: Session = Depends(get_session),
):
    _require_project(session, project_id)
    params = _params(
        target_fps, max_frames, start_sec, end_sec, dedup, dedup_threshold,
        tile, tile_size, stride,
    )

    videos_dir = _project_dir(project_id) / "videos"
    meta_path = videos_dir / f"{video_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, "Video not found")
    meta = json.loads(meta_path.read_text())
    video_path = videos_dir / f"{video_id}{meta['ext']}"
    if not video_path.exists():
        raise HTTPException(404, "Video file missing")
    if video_manager.is_active(video_id):
        raise HTTPException(409, "Extraction already in progress")

    meta["params"] = params.__dict__
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))
    video_manager.submit(
        video_id, video_path, _project_dir(project_id) / "raw", meta["stem"], params
    )
    return {"video_id": video_id, "status": "running"}


@router.post("/{video_id}/cancel")
def cancel_video(project_id: str, video_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    ok = video_manager.cancel(video_id)
    return {"cancelled": ok}


@router.get("/{video_id}/events")
async def video_events(project_id: str, video_id: str, session: Session = Depends(get_session)):
    _require_project(session, project_id)
    if not (task_dir(video_id) / "progress.jsonl").exists():
        raise HTTPException(404, "Extraction task not found")

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, video_id, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL_PHASES:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())
