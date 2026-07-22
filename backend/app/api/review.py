"""Review queue: humans verify/edit uncertain boxes, then approve or reject.

Approve writes the final YOLO label into confirmed/ and removes the item;
reject records the stem in rejected/rejected.txt. Both return the next
pending item id so the UI can chain through the queue without extra queries.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.config import settings
from app.core.yolo_io import atomic_write_text, write_label_file
from app.db import get_session
from app.models import ReviewItem

router = APIRouter(prefix="/api", tags=["review"])

# guards concurrent approve/reject on the same project (e.g. two tabs)
_project_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class BoxIn(BaseModel):
    id: str | None = None
    cls: int
    xyxy_n: list[float] = Field(min_length=4, max_length=4)
    score: float | None = None
    status: str | None = None
    reason: str | None = None
    sources: list[dict] | None = None


class DraftIn(BaseModel):
    boxes: list[BoxIn]


class ApproveIn(BaseModel):
    boxes: list[BoxIn] | None = None  # None → approve stored boxes as-is


def _project_dir(project_id: str) -> Path:
    return settings.projects_dir / project_id


def _review_json_path(item: ReviewItem) -> Path:
    return _project_dir(item.project_id) / "review" / f"{item.stem}.json"


def _get_item(session: Session, item_id: str) -> ReviewItem:
    item = session.get(ReviewItem, item_id)
    if item is None:
        raise HTTPException(404, "리뷰 항목이 없습니다")
    return item


def _load_payload(item: ReviewItem) -> dict:
    path = _review_json_path(item)
    if not path.exists():
        raise HTTPException(410, "리뷰 파일이 없습니다 (이미 처리되었을 수 있습니다)")
    return json.loads(path.read_text())


def _find_raw_image(project_id: str, stem: str) -> Path | None:
    raw = _project_dir(project_id) / "raw"
    for p in raw.glob(f"{stem}.*"):
        return p
    return None


def _next_pending_id(session: Session, project_id: str, exclude: str) -> str | None:
    rows = session.exec(
        select(ReviewItem)
        .where(
            ReviewItem.project_id == project_id,
            ReviewItem.status == "pending",
            ReviewItem.id != exclude,
        )
        .order_by(ReviewItem.uncertainty.desc(), ReviewItem.stem)
    ).all()
    return rows[0].id if rows else None


def _classes(project_id: str) -> list[dict]:
    path = _project_dir(project_id) / "classes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("classes", [])


@router.get("/projects/{project_id}/review")
def review_queue(
    project_id: str,
    page: int = 1,
    size: int = 100,
    session: Session = Depends(get_session),
):
    rows = session.exec(
        select(ReviewItem)
        .where(ReviewItem.project_id == project_id, ReviewItem.status == "pending")
        .order_by(ReviewItem.uncertainty.desc(), ReviewItem.stem)
    ).all()
    total = len(rows)
    start = (page - 1) * size
    items = []
    for row in rows[start : start + size]:
        img = _find_raw_image(project_id, row.stem)
        items.append(
            {
                "id": row.id,
                "stem": row.stem,
                "uncertainty": row.uncertainty,
                "n_flagged": row.n_flagged,
                "thumb": f"/api/files/projects/{project_id}/thumbs/{img.name}" if img else None,
            }
        )
    return {"total": total, "page": page, "size": size, "items": items}


@router.get("/review/{item_id}")
def get_review_item(item_id: str, session: Session = Depends(get_session)):
    item = _get_item(session, item_id)
    payload = _load_payload(item)
    img = _find_raw_image(item.project_id, item.stem)
    boxes = payload.get("edits") or payload.get("boxes", [])
    return {
        "id": item.id,
        "project_id": item.project_id,
        "stem": item.stem,
        "status": item.status,
        "uncertainty": item.uncertainty,
        "image_url": f"/api/files/projects/{item.project_id}/raw/{img.name}" if img else None,
        "width": payload.get("width"),
        "height": payload.get("height"),
        "boxes": boxes,
        "original_boxes": payload.get("boxes", []),
        "classes": _classes(item.project_id),
    }


@router.put("/review/{item_id}")
def save_draft(item_id: str, draft: DraftIn, session: Session = Depends(get_session)):
    item = _get_item(session, item_id)
    payload = _load_payload(item)
    payload["edits"] = [b.model_dump() for b in draft.boxes]
    atomic_write_text(_review_json_path(item), json.dumps(payload, indent=2))
    return {"ok": True}


@router.post("/review/{item_id}/approve")
async def approve(item_id: str, req: ApproveIn, session: Session = Depends(get_session)):
    item = _get_item(session, item_id)
    async with _project_locks[item.project_id]:
        payload = _load_payload(item)
        if req.boxes is not None:
            boxes = [b.model_dump() for b in req.boxes]
        else:
            boxes = payload.get("edits") or payload.get("boxes", [])

        pdir = _project_dir(item.project_id)
        img = _find_raw_image(item.project_id, item.stem)
        if img is None:
            raise HTTPException(410, "원본 이미지가 없습니다")

        write_label_file(
            pdir / "confirmed" / "labels" / f"{item.stem}.txt",
            [(int(b["cls"]), tuple(b["xyxy_n"])) for b in boxes],
        )
        dst = pdir / "confirmed" / "images" / img.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)

        _review_json_path(item).unlink(missing_ok=True)
        item.status = "approved"
        session.add(item)
        session.commit()

    return {"ok": True, "next_id": _next_pending_id(session, item.project_id, item.id)}


@router.post("/review/{item_id}/reject")
async def reject(item_id: str, session: Session = Depends(get_session)):
    item = _get_item(session, item_id)
    async with _project_locks[item.project_id]:
        pdir = _project_dir(item.project_id)
        rejected = pdir / "rejected" / "rejected.txt"
        rejected.parent.mkdir(parents=True, exist_ok=True)
        with open(rejected, "a") as f:
            f.write(item.stem + "\n")

        _review_json_path(item).unlink(missing_ok=True)
        item.status = "rejected"
        session.add(item)
        session.commit()

    return {"ok": True, "next_id": _next_pending_id(session, item.project_id, item.id)}
