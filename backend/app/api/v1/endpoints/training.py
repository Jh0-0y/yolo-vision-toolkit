"""Training runs: launch/monitor/stop ultralytics training, browse artifacts,
register trained weights back into the model registry (closing the loop)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import or_
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session, session_scope
from app.models import ModelEntry, Project, TrainRun, iso_utc
from app.schemas.training import RegisterIn, RunCreate, RunOut
from app.services import datasets
from app.services.train_manager import train_manager
from infra import jobs
from lib.labels import dataset_export
from lib.train import results as train_results

router = APIRouter(prefix="/training", tags=["training"])

TERMINAL = {"done", "error", "stopped"}
ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".csv"}


def _safe_part(s: str) -> str:
    """Filename-safe token (keeps unicode letters/digits, e.g. Korean)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s)).strip("_") or "x"


def _fdt(dt) -> str:
    """Local-time stamp for filenames: YYYYMMDD_HHMMSS (naive is assumed UTC)."""
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y%m%d_%H%M%S")
    except Exception:
        return dt.strftime("%Y%m%d_%H%M%S")


def _to_out(run: TrainRun, session: Session) -> RunOut:
    base = session.get(ModelEntry, run.base_model_id)
    return RunOut(
        id=run.id,
        name=run.name,
        status=run.status,
        dataset_path=run.dataset_path,
        base_model_id=run.base_model_id,
        base_model_name=base.name if base else None,
        params=json.loads(run.params_json),
        metrics=json.loads(run.metrics_json) if run.metrics_json else None,
        error=run.error,
        created_at=iso_utc(run.created_at),
        finished_at=iso_utc(run.finished_at),
    )



def _parse_dataset_token(token: str) -> tuple[str, str]:
    """`dataset:{project_id}:{dataset_id}` → `(project_id, dataset_id)`.

    학습은 데이터셋을 **직접** 먹는다 — 내보내기를 먼저 할 필요가 없다.
    """
    parts = token.split(":")
    if parts[0] != "dataset" or len(parts) != 3:
        raise HTTPException(422, f"Invalid dataset token: {token}")
    pid, dsid = parts[1], parts[2]
    if any(c in pid + dsid for c in "/\\.."):
        raise HTTPException(422, "Invalid dataset token")
    if datasets.read_meta(pid, dsid) is None:
        raise HTTPException(404, "Dataset not found")
    return pid, dsid


@router.post("/runs", response_model=RunOut, status_code=201)
def create_run(req: RunCreate, session: Session = Depends(get_session)):
    project_id, dataset_id = _parse_dataset_token(req.dataset)
    base = session.get(ModelEntry, req.base_model_id)
    if base is None:
        raise HTTPException(422, "Base model not found")
    base_pt = settings.model_dir(base.project_id, req.base_model_id) / "model.pt"
    if not base_pt.exists():
        raise HTTPException(422, "Base model file missing")
    if train_manager.has_active():
        raise HTTPException(409, "A training run is already in progress. Try again after it finishes.")

    ds_name = (datasets.read_meta(project_id, dataset_id) or {}).get("name", dataset_id)
    run = TrainRun(
        name=req.name or f"{base.name}-{ds_name}",
        project_id=project_id,
        dataset_path="",  # 아래에서 구체화한 자리로 채운다
        base_model_id=req.base_model_id,
        params_json=json.dumps(req.params.model_dump(exclude_none=True)),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    run_dir = settings.run_dir(run.project_id, run.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋의 train/val 을 이 런 아래에 **하드링크로** 펼친다. 이미지 바이트를
    # 복제하지 않으므로 사실상 공짜고, 런이 끝난 뒤에도 그때 무엇으로 학습했는지가
    # 그대로 남는다(데이터셋을 나중에 고쳐도 이 런의 기록은 안 흔들린다).
    dataset_dir = run_dir / "dataset"
    try:
        dataset_export.materialize(
            dataset_dir=datasets.dataset_dir(project_id, dataset_id),
            out_dir=dataset_dir,
            kind="train",
            reviewed=datasets.read_reviewed(project_id, dataset_id)
            & datasets.image_stems(project_id, dataset_id),
            splits=datasets.read_splits(project_id, dataset_id),
        )
    except dataset_export.ExportError as e:
        session.delete(run)
        session.commit()
        raise HTTPException(422, str(e)) from None
    run.dataset_path = str(dataset_dir)
    session.add(run)
    session.commit()
    session.refresh(run)
    (settings.jobs_dir / run.id).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_dir),
                "base_model_path": str(base_pt),
                "device": req.device or settings.device,
                "params": req.params.model_dump(exclude_none=True),
            }
        )
    )
    train_manager.start(run.id)
    session.refresh(run)
    return _to_out(run, session)


@router.get("/runs", response_model=list[RunOut])
def list_runs(project_id: str | None = None, session: Session = Depends(get_session)):
    stmt = select(TrainRun)
    if project_id is not None:
        # this project's runs plus legacy/shared (project_id NULL)
        stmt = stmt.where(
            or_(TrainRun.project_id == project_id, TrainRun.project_id.is_(None))
        )
    runs = session.exec(stmt.order_by(TrainRun.created_at.desc())).all()
    return [_to_out(r, session) for r in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    return _to_out(run, session)


@router.get("/runs/{run_id}/history")
def run_history(run_id: str, session: Session = Depends(get_session)):
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "Training run not found")
    events, _ = jobs.at(settings.jobs_dir, run_id).read()
    return [e for e in events if e.get("phase") == "epoch"]


@router.get("/runs/{run_id}/per-class")
def run_per_class(run_id: str, session: Session = Depends(get_session)):
    """Per-class metrics (P/R/mAP) captured at the end of training; [] if absent
    (older runs, or the extraction failed)."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "per_class.json"
    return train_results.read_json(path, default=[])


@router.get("/runs/{run_id}/per-class-history")
def run_per_class_history(run_id: str, session: Session = Depends(get_session)):
    """Per-class metrics per epoch (for 'class metric over epochs' charts); []
    for older runs that predate this capture."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "per_class_history.jsonl"
    return train_results.read_jsonl(path)


@router.get("/runs/{run_id}/results")
def run_results(run_id: str, session: Session = Depends(get_session)):
    """Full per-epoch metrics from ultralytics results.csv (train+val loss, mAP,
    precision/recall, lr, time). Works for completed and past runs; [] if none."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    csv_path = train_results.find(settings.run_dir(run.project_id, run_id), "results.csv")
    if csv_path is None:
        return []
    return train_results.read_results_csv(csv_path)


@router.get("/runs/{run_id}/results.csv")
def download_results_csv(run_id: str, session: Session = Depends(get_session)):
    """Download the raw ultralytics results.csv (per-epoch train/val loss, P/R,
    mAP50, mAP50-95, lr, time) as a CSV file named after the run."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    csv_path = train_results.find(settings.run_dir(run.project_id, run_id), "results.csv")
    if csv_path is None:
        raise HTTPException(404, "results.csv is not available for this run yet")
    filename = f"{_safe_part(run.name or run_id)}_results.csv"
    return FileResponse(csv_path, media_type="text/csv", filename=filename)


@router.get("/runs/{run_id}/args.yaml")
def download_args_yaml(run_id: str, session: Session = Depends(get_session)):
    """Download ultralytics args.yaml — the fully-resolved training config for this
    run (every hyperparameter actually applied, including all augmentation values)."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = train_results.find(settings.run_dir(run.project_id, run_id), "args.yaml")
    if path is None:
        raise HTTPException(404, "args.yaml is not available for this run yet")
    filename = f"{_safe_part(run.name or run_id)}_args.yaml"
    return FileResponse(path, media_type="text/yaml", filename=filename)


@router.get("/runs/{run_id}/log")
def run_log(run_id: str, tail_kb: int = 256, session: Session = Depends(get_session)):
    """The training worker's train.log (stdout+stderr, incl. full tracebacks).
    Only the last tail_kb KiB are returned so long runs stay cheap to fetch."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "train.log"
    if not path.exists():
        return {"text": "", "truncated": False}
    limit = max(1, tail_kb) * 1024
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    # strip ANSI color/escape codes — a plain code block can't render them
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return {"text": text, "truncated": size > limit}


@router.post("/runs/{run_id}/stop", response_model=RunOut)
def stop_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    if run.status not in TERMINAL:
        train_manager.stop(run_id)
    session.expire_all()
    return _to_out(session.get(TrainRun, run_id), session)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    if run.status not in TERMINAL:
        raise HTTPException(409, "Cannot delete a run that is still active. Stop it first.")
    run_dir = settings.run_dir(run.project_id, run_id)
    session.delete(run)
    session.commit()
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(settings.jobs_dir / run_id, ignore_errors=True)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, session: Session = Depends(get_session)):
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "Training run not found")

    async def stream():
        offset = 0
        idle = 0
        while True:
            events, offset = await asyncio.to_thread(jobs.at(settings.jobs_dir, run_id).read, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in {"done", "error", "cancelled"}:
                    terminal = True
            if terminal:
                return
            if not events:
                idle += 1
                if idle % 10 == 0:
                    with session_scope() as s:
                        run = s.get(TrainRun, run_id)
                        if run is not None and run.status in TERMINAL:
                            yield {
                                "event": "progress",
                                "data": json.dumps({"phase": run.status, "msg": run.error}),
                            }
                            return
            else:
                idle = 0
            await asyncio.sleep(1.0)

    return EventSourceResponse(stream())


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    run_dir = settings.run_dir(run.project_id, run_id)
    files = []
    if run_dir.exists():
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS:
                files.append({"name": p.name, "url": f"{settings.api_prefix}/training/runs/{run_id}/artifacts/{p.name}"})
    weights = []
    for w in ("best.pt", "last.pt"):
        if (run_dir / "weights" / w).exists():
            weights.append({"name": w, "url": f"{settings.api_prefix}/training/runs/{run_id}/weights/{w.split('.')[0]}"})
    return {"files": files, "weights": weights}


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str, session: Session = Depends(get_session)):
    if "/" in name or name.startswith("."):
        raise HTTPException(422, "Invalid filename")
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / name
    if not path.exists() or path.suffix.lower() not in ARTIFACT_EXTS:
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@router.get("/runs/{run_id}/weights/{which}")
def download_weights(run_id: str, which: str, session: Session = Depends(get_session)):
    if which not in ("best", "last"):
        raise HTTPException(422, "Only best or last is allowed")
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "weights" / f"{which}.pt"
    if not path.exists():
        raise HTTPException(404, "Weights not found")
    stamp = _fdt(run.created_at)
    fname = f"{stamp}-{_safe_part(run.name)}-{which}.pt"
    return FileResponse(path, filename=fname)


@router.post("/runs/{run_id}/register", status_code=201)
def register_weights(run_id: str, req: RegisterIn, session: Session = Depends(get_session)):
    from app.api.v1.endpoints.models import _register
    from app.api.v1.endpoints.models import _to_out as model_out

    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    if req.which not in ("best", "last"):
        raise HTTPException(422, "Only best or last is allowed")
    src = settings.run_dir(run.project_id, run_id) / "weights" / f"{req.which}.pt"
    if not src.exists():
        raise HTTPException(404, "Weights not found")

    # _register moves its input — hand it a copy so the run keeps its weights
    tmp = settings.models_dir / f".register_{run_id}_{req.which}.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    default_name = f"{run.name}-{req.which}"
    entry = _register(
        session, req.name or default_name, tmp, source="trained", project_id=run.project_id
    )
    return model_out(entry)
