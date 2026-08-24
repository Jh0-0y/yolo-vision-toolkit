"""추론 라우터들이 함께 쓰는 조각.

두 계열(추론 · 벤치마크)이 모두 모델 경로를 풀고, 잡 진행률을 SSE 로 흘린다.
공유하는 것만 여기 둔다 — 계열별 로직은 각자 모듈에 있다. 진행률 스트림은
연구실 크롭 런(`endpoints/lab_crops.py`)도 여기서 빌려 쓴다.
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
