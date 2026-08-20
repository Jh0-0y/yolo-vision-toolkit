"""데이터셋 타일링 — 검수완료 이미지를 타일로 쪼개 **새 데이터셋**을 만든다.

결과가 그냥 데이터셋이라 내보내기·학습·분할·라벨 에디터는 아무것도 바뀌지 않는다.
타일은 전부 미할당으로 나오고, 분할은 파생 데이터셋에서 따로 한다 —
흐름이 `검수 → 타일링 → 분할` 이기 때문이다.

진행률 SSE 를 여기에 따로 두는 이유: `/jobs/{id}/events` 는 DB `Job` 행을 요구하는데
타일링은 DB 행을 만들지 않는다(상태는 progress.jsonl 에서 파생한다). 가져오기가
자기 SSE 를 가진 것과 같은 이유다.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.schemas.dataset import (
    TileEstimateOut,
    TileIn,
    TileOut,
    TileParamsIn,
)
from app.services import datasets
from app.services.tile_manager import tile_manager
from infra import jobs
from lib.labels.dataset_tile import TileDatasetParams, TileError, estimate

router = APIRouter(
    prefix="/projects/{project_id}/datasets/{dataset_id}/tile", tags=["datasets"]
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


def _params(body: TileParamsIn) -> TileDatasetParams:
    params = TileDatasetParams(
        tile_size=body.tile_size,
        stride=body.stride,
        min_visibility=body.min_visibility,
        negative_ratio=body.negative_ratio,
        keep_all_negatives=body.keep_all_negatives,
        seed=body.seed,
    )
    errors = params.validate()
    if errors:
        raise HTTPException(422, f"Invalid tiling params: {'; '.join(errors)}")
    return params


def _derived_name(requested: str, source_name: str, tile_size: int) -> str:
    """파생 데이터셋 이름. 비우면 원본 이름에 타일 크기를 붙인다 —
    목록에서 `경기 1차` 와 `경기 1차 (tiled 640)` 이 나란히 보여야 한다."""
    return requested.strip() or f"{source_name} (tiled {tile_size})"


def _reviewed(project_id: str, dataset_id: str) -> set[str]:
    """검수완료이면서 **파일이 실제로 있는 것**만. 검수완료가 아닌 것은 나가지 않는다."""
    return datasets.read_reviewed(project_id, dataset_id) & datasets.image_stems(
        project_id, dataset_id
    )


@router.post("/estimate", response_model=TileEstimateOut)
async def estimate_tiling(
    project_id: str,
    dataset_id: str,
    body: TileParamsIn,
    session: Session = Depends(get_session),
):
    """만들기 전에 몇 장이 나오는지 센다. 픽셀을 디코딩하지 않아 빠르다."""
    ddir = _require_dataset(session, project_id, dataset_id)
    params = _params(body)
    try:
        result = await run_in_threadpool(
            estimate,
            dataset_dir=ddir,
            reviewed=_reviewed(project_id, dataset_id),
            params=params,
        )
    except TileError as e:
        raise HTTPException(422, str(e)) from None
    return TileEstimateOut(**result)


@router.post("", response_model=TileOut, status_code=201)
def create_tiled_dataset(
    project_id: str,
    dataset_id: str,
    body: TileIn,
    session: Session = Depends(get_session),
):
    """파생 데이터셋을 만들고 타일링 잡을 던진다.

    데이터셋을 **먼저** 만드는 이유: 잡 id 가 그 데이터셋 id 이고, 실패해도
    무엇을 만들려 했는지가 남아야 한다.
    """
    ddir = _require_dataset(session, project_id, dataset_id)
    params = _params(body)
    reviewed = _reviewed(project_id, dataset_id)
    if not reviewed:
        raise HTTPException(422, "Nothing to tile — review some images first")
    try:
        preview = estimate(dataset_dir=ddir, reviewed=reviewed, params=params)
    except TileError as e:
        raise HTTPException(422, str(e)) from None

    src_name = (datasets.read_meta(project_id, dataset_id) or {}).get("name") or dataset_id
    name = _derived_name(body.name, src_name, params.tile_size)
    tile_id = datasets.create(project_id, name)
    out_dir = datasets.dataset_dir(project_id, tile_id)

    # 클래스는 원본 것을 물려받는다 — 라벨의 클래스 id 가 그 목록을 가리킨다
    src_classes = ddir / "classes.json"
    if src_classes.exists():
        shutil.copyfile(src_classes, out_dir / "classes.json")

    # 기록은 **시작할 때** 쓴다 — 잡이 죽어도 이 데이터셋이 무엇이었는지 남는다.
    # `params` 를 박아 두는 이유: 타일 크기는 학습 imgsz · 타일 추론 tile_size 와
    # 같아야 하므로, 나중에 되읽을 수 있어야 한다.
    datasets.add_source(
        project_id,
        tile_id,
        {
            "id": tile_id,
            "kind": "tiling",
            "filename": src_name,
            "stem": None,
            "at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "source_dataset_id": dataset_id,
            "params": asdict(params),
            "images": preview["images"],
            "tiles": preview["total"],
            "positive": preview["positive"],
            "negative": preview["negative_kept"],
        },
    )

    def _record(result: dict | None) -> None:
        if result is None:
            datasets.update_source(project_id, tile_id, tile_id, status="error")
            return
        status = result.get("status", "done")
        datasets.update_source(
            project_id,
            tile_id,
            tile_id,
            status=status,
            images=result.get("images", 0),
            tiles=result.get("saved", 0),
            positive=result.get("positive", 0),
            negative=result.get("negative", 0),
        )
        # **끝까지 간 것만 검수완료로 올린다.** 중단된 런의 반쪽 타일이
        # 검수완료 딱지를 달면 그대로 학습·내보내기로 새어 나간다.
        if status == "done":
            datasets.write_reviewed(
                project_id, tile_id, datasets.image_stems(project_id, tile_id)
            )

    tile_manager.submit(tile_id, ddir, out_dir, reviewed, params, on_done=_record)
    return TileOut(dataset_id=tile_id, job_id=tile_id, name=name, status="running")


@router.post("/{tile_id}/cancel")
def cancel_tiling(
    project_id: str, dataset_id: str, tile_id: str, session: Session = Depends(get_session)
):
    _require_dataset(session, project_id, dataset_id)
    if not datasets.valid_id(tile_id):
        raise HTTPException(422, "Invalid dataset id")
    return {"cancelled": tile_manager.cancel(tile_id)}


@router.get("/{tile_id}/events")
async def tiling_events(
    project_id: str, dataset_id: str, tile_id: str, session: Session = Depends(get_session)
):
    """진행률 SSE. progress.jsonl 을 처음부터 재생하므로 언제 붙어도 같은 그림이다."""
    _require_dataset(session, project_id, dataset_id)
    if not datasets.valid_id(tile_id):
        raise HTTPException(422, "Invalid dataset id")
    if not (settings.jobs_dir / tile_id / "progress.jsonl").exists():
        raise HTTPException(404, "Tiling task not found")

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(
                jobs.at(settings.jobs_dir, tile_id).read, offset
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
