"""Test 플레이그라운드 라우터들이 함께 쓰는 조각.

네 계열(추론 · annotate · live · compare)이 모두 모델 경로를 풀고, 잡 진행률을
SSE 로 흘린다. 그 셋만 여기 둔다 — 계열별 로직은 각자 모듈에 있다.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.models import ModelEntry
from infra import jobs

# 이 페이즈가 보이면 스트림을 닫는다 — 워커가 더 쓸 것이 없다.
TERMINAL = {"done", "error", "cancelled"}


async def job_event_stream(job_id: str):
    """SSE tail of jobs_dir/{job_id}/progress.jsonl until a terminal phase."""

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(jobs.at(settings.jobs_dir, job_id).read, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())


def model_pt(session: Session, model_id: str, project_id: str | None) -> str:
    """Resolve a model id to its .pt path, enforcing project scope."""
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(422, f"Model not found: {model_id}")
    if entry.project_id is not None and project_id and entry.project_id != project_id:
        raise HTTPException(422, f"Model does not belong to this project: {model_id}")
    pt = settings.model_dir(entry.project_id, model_id) / "model.pt"
    if not pt.exists():
        raise HTTPException(422, f"Model file missing: {model_id}")
    return str(pt)


def detectors_cfg(
    session: Session, detectors: str, project_id: str | None
) -> list[dict]:
    """검출기 엔트리 목록 파싱 — [{model_id, mode, conf?, imgsz?, tile_size?,
    stride?, merge_iou?}]. 비어있으면 [] (호출측이 레거시 파라미터로 구성)."""
    try:
        entries = json.loads(detectors) if detectors else []
        if not isinstance(entries, list):
            raise ValueError
    except ValueError:
        raise HTTPException(422, "detectors must be a JSON array") from None

    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict) or not e.get("model_id"):
            raise HTTPException(422, "each detector needs a model_id")
        mode = e.get("mode", "full")
        if mode not in ("full", "tiled"):
            raise HTTPException(422, "detector mode must be 'full' or 'tiled'")
        tile_size = int(e.get("tile_size", 640))
        stride = int(e.get("stride", 480))
        if mode == "tiled" and (stride <= 0 or stride > tile_size):
            raise HTTPException(422, "stride must be in (0, tile_size]")
        out.append(
            {
                "pt": model_pt(session, e["model_id"], project_id),
                "mode": mode,
                "conf": e.get("conf"),
                "imgsz": int(e.get("imgsz", 1920)),
                "tile_size": tile_size,
                "stride": stride,
                "merge_iou": float(e.get("merge_iou", 0.5)),
            }
        )
    return out
