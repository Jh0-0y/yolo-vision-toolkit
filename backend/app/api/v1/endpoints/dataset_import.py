"""데이터셋으로 가져오기 — 동영상 · 기존 YOLO 데이터셋.

두 방식 모두 **미검수**로 들어온다. 검수는 그 다음 일이다.

동영상은 프레임을 얻는 수단일 뿐이라 **추출이 끝나면 지운다.** 그래서 업로드도
데이터셋 안 감춰진 자리(`.import/`)로 받는다 — 목록에 뜨는 자리가 아니다.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.services import datasets
from app.services.video_manager import video_manager
from infra import jobs
from lib.formats import VIDEO_EXTS
from lib.labels.import_yolo import YoloImportError, import_zip
from lib.media.extract import ExtractParams
from lib.media.tiling import TilingParams

router = APIRouter(
    prefix="/projects/{project_id}/datasets/{dataset_id}/import", tags=["datasets"]
)

TERMINAL_PHASES = {"done", "error", "cancelled"}


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


def _staging_dir(dataset: Path) -> Path:
    """업로드를 잠깐 받아 두는 자리. 이미지 목록에 섞이지 않게 감춰 둔다."""
    d = dataset / ".import"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize_stem(name: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", Path(name).stem).strip("_")
    return stem or "video"


def _extract_params(
    target_fps: float,
    max_frames: int,
    start_sec: float,
    end_sec: float | None,
    dedup: bool,
    dedup_threshold: float,
    tile: bool,
    tile_size: int,
    stride: int,
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


@router.post("/video", status_code=201)
async def import_video(
    project_id: str,
    dataset_id: str,
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
    """영상을 올려 프레임을 이 데이터셋으로 뽑는다. **영상은 끝나면 지운다.**"""
    dataset = _require_dataset(session, project_id, dataset_id)
    params = _extract_params(
        target_fps, max_frames, start_sec, end_sec, dedup, dedup_threshold,
        tile, tile_size, stride,
    )

    name = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            422, f"Unsupported video format: {ext or 'none'} ({', '.join(sorted(VIDEO_EXTS))})"
        )

    job_id = f"imp_{uuid.uuid4().hex[:10]}"
    source = _staging_dir(dataset) / f"{job_id}{ext}"
    try:
        with open(source, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
    except Exception:
        source.unlink(missing_ok=True)
        raise

    datasets.ensure_dirs(project_id, dataset_id)
    video_manager.submit(
        job_id,
        source,
        datasets.raw_dir(project_id, dataset_id),
        _sanitize_stem(name),
        params,
        delete_source=True,
    )
    return {"job_id": job_id, "filename": name, "status": "running"}


@router.post("/dataset", status_code=201)
async def import_yolo_dataset(
    project_id: str,
    dataset_id: str,
    file: UploadFile,
    tile: bool = Form(False),
    tile_size: int = Form(640),
    stride: int = Form(480),
    min_visibility: float = Form(0.3),
    drop_empty: bool = Form(True),
    session: Session = Depends(get_session),
):
    """YOLO zip(이미지+labels+data.yaml)을 이 데이터셋으로 들여온다.

    클래스는 이 데이터셋의 `classes.json` 에 병합되고 라벨의 클래스 id 는 그에 맞춰
    다시 매겨진다 — 데이터셋마다 클래스가 다르므로 저쪽 번호를 그대로 쓸 수 없다.
    """
    dataset = _require_dataset(session, project_id, dataset_id)
    tiling = None
    if tile:
        tiling = TilingParams(
            tile_size=tile_size,
            stride=stride,
            min_visibility=min_visibility,
            drop_empty=drop_empty,
        )
        errors = tiling.validate()
        if errors:
            raise HTTPException(422, f"Invalid tiling params: {errors}")

    staging = _staging_dir(dataset)
    tmp = staging / f"upload_{uuid.uuid4().hex[:8]}.zip"
    try:
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        result = await run_in_threadpool(import_zip, tmp, dataset, tiling)
    except YoloImportError as e:
        raise HTTPException(422, str(e)) from None
    finally:
        tmp.unlink(missing_ok=True)
    return result


@router.post("/{job_id}/cancel")
def cancel_import(
    project_id: str, dataset_id: str, job_id: str, session: Session = Depends(get_session)
):
    _require_dataset(session, project_id, dataset_id)
    return {"cancelled": video_manager.cancel(job_id)}


@router.get("/{job_id}/events")
async def import_events(
    project_id: str, dataset_id: str, job_id: str, session: Session = Depends(get_session)
):
    """진행률 SSE. progress.jsonl 을 처음부터 재생하므로 언제 붙어도 같은 그림이다."""
    _require_dataset(session, project_id, dataset_id)
    if not (settings.jobs_dir / job_id / "progress.jsonl").exists():
        raise HTTPException(404, "Import task not found")

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(
                jobs.at(settings.jobs_dir, job_id).read, offset
            )
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL_PHASES:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())
