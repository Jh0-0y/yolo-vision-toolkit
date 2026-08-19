"""데이터셋의 분할과 내보내기.

분할은 **미할당만** 건드리는 것이 기본이다 — 이미 train/val/test 에 들어간 이미지를
다시 섞으면 어제 test 였던 것이 오늘 train 으로 가고, 이전 평가와 비교가 성립하지
않는다. 전부 다시 섞으려면 `reassign_all` 을 명시해야 한다.

내보내기는 **이력을 남기지 않는다.** 만들어서 받으면 끝이고, 새로 내보내면 이전
zip 은 지운다 — 목록이 필요하면 데이터셋 자체가 목록이다.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.db import get_session
from app.models import Project
from app.schemas.dataset import DatasetOut, ExportIn, ExportOut, SplitIn
from app.services import datasets
from lib.labels import split as split_lib
from lib.labels.dataset_export import ExportError, build

router = APIRouter(prefix="/projects/{project_id}/datasets/{dataset_id}", tags=["datasets"])


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


@router.post("/split", response_model=DatasetOut)
def split_dataset(
    project_id: str,
    dataset_id: str,
    body: SplitIn,
    session: Session = Depends(get_session),
):
    """검수완료 이미지를 비율로 나눈다. 기본 8:1:1, **미할당만**."""
    _require_dataset(session, project_id, dataset_id)
    stems = sorted(
        datasets.image_stems(project_id, dataset_id)
        & datasets.read_reviewed(project_id, dataset_id)
    )
    if not stems:
        raise HTTPException(422, "Nothing to split — review some images first")

    try:
        assigned = split_lib.assign(
            stems,
            datasets.read_splits(project_id, dataset_id),
            train=body.train,
            val=body.val,
            test=body.test,
            seed=body.seed,
            reassign_all=body.reassign_all,
        )
    except split_lib.SplitError as e:
        raise HTTPException(422, str(e)) from None

    datasets.write_splits(project_id, dataset_id, assigned)
    return datasets.to_out(
        project_id, dataset_id, datasets.read_meta(project_id, dataset_id) or {}
    )


def _export_dir(ddir: Path) -> Path:
    d = ddir / ".export"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/export", response_model=ExportOut)
async def export_dataset(
    project_id: str,
    dataset_id: str,
    body: ExportIn,
    session: Session = Depends(get_session),
):
    """세 가지 중 하나를 zip 으로 만든다. 이전에 만든 것은 지운다(이력 없음)."""
    ddir = _require_dataset(session, project_id, dataset_id)
    staging = _export_dir(ddir)
    # 이력을 남기지 않는다 — 새로 내보내면 이전 것은 사라진다
    for old in staging.iterdir():
        if old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)

    export_id = f"e_{uuid.uuid4().hex[:10]}"
    try:
        result = await run_in_threadpool(
            build,
            dataset_dir=ddir,
            out_dir=staging / export_id,
            kind=body.kind,
            reviewed=datasets.read_reviewed(project_id, dataset_id)
            & datasets.image_stems(project_id, dataset_id),
            splits=datasets.read_splits(project_id, dataset_id),
            class_ids=body.class_ids,
        )
    except ExportError as e:
        raise HTTPException(422, str(e)) from None

    return ExportOut(id=export_id, **{k: v for k, v in result.items() if k != "zip"})


@router.get("/export/{export_id}/download")
def download_export(
    project_id: str, dataset_id: str, export_id: str, session: Session = Depends(get_session)
):
    ddir = _require_dataset(session, project_id, dataset_id)
    if not datasets.valid_id(export_id):
        raise HTTPException(422, "Invalid export id")
    path = _export_dir(ddir) / f"{export_id}.zip"
    if not path.exists():
        raise HTTPException(404, "Export not found — build it again")
    name = (datasets.read_meta(project_id, dataset_id) or {}).get("name", dataset_id)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or dataset_id
    return FileResponse(path, media_type="application/zip", filename=f"{safe}.zip")
