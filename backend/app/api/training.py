"""Training runs: launch/monitor/stop ultralytics training, browse artifacts,
register trained weights back into the model registry (closing the loop)."""

from __future__ import annotations

import asyncio
import json
import shutil

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session, session_scope
from app.jobs.runner import read_progress
from app.jobs.train_manager import train_manager
from app.models import ModelEntry, Project, TrainRun

router = APIRouter(prefix="/api/train", tags=["training"])

TERMINAL = {"done", "error", "stopped"}
ARTIFACT_EXTS = {".png", ".jpg", ".jpeg", ".csv"}


class TrainParams(BaseModel):
    epochs: int = Field(default=100, ge=1, le=10000)
    imgsz: int = Field(default=640, ge=64, le=4096)
    batch: int = Field(default=16, ge=1, le=512)
    patience: int = Field(default=100, ge=0)
    lr0: float | None = None
    optimizer: str | None = None  # auto | SGD | Adam | AdamW ...
    workers: int = Field(default=2, ge=0, le=16)


class RunCreate(BaseModel):
    name: str | None = None
    project_id: str
    export_id: str
    base_model_id: str
    device: str | None = None
    params: TrainParams = Field(default_factory=TrainParams)


class RunOut(BaseModel):
    id: str
    name: str
    status: str
    dataset_path: str
    base_model_id: str
    base_model_name: str | None = None
    params: dict
    metrics: dict | None
    error: str | None
    created_at: str
    finished_at: str | None


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
        created_at=run.created_at.isoformat(),
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
    )


@router.get("/datasets")
def list_datasets(session: Session = Depends(get_session)):
    """Every export across all projects, newest first."""
    projects = {p.id: p.name for p in session.exec(select(Project)).all()}
    out = []
    for pid, pname in projects.items():
        exports_dir = settings.projects_dir / pid / "exports"
        if not exports_dir.exists():
            continue
        for meta_path in exports_dir.glob("*/export.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            out.append(
                {
                    "project_id": pid,
                    "project_name": pname,
                    "export_id": meta["id"],
                    "train": meta.get("train", 0),
                    "val": meta.get("val", 0),
                    "classes": meta.get("classes", 0),
                    "created_at": meta.get("created_at", ""),
                }
            )
    out.sort(key=lambda d: d["created_at"], reverse=True)
    return out


@router.post("/runs", response_model=RunOut)
def create_run(req: RunCreate, session: Session = Depends(get_session)):
    dataset_dir = settings.projects_dir / req.project_id / "exports" / req.export_id
    if not (dataset_dir / "data.yaml").exists():
        raise HTTPException(422, "데이터셋(내보내기)을 찾을 수 없습니다")
    base = session.get(ModelEntry, req.base_model_id)
    if base is None:
        raise HTTPException(422, "베이스 모델이 없습니다")
    base_pt = settings.models_dir / req.base_model_id / "model.pt"
    if not base_pt.exists():
        raise HTTPException(422, "베이스 모델 파일이 없습니다")
    if train_manager.has_active():
        raise HTTPException(409, "이미 실행 중인 학습이 있습니다. 완료 후 다시 시도하세요.")

    run = TrainRun(
        name=req.name or f"{base.name}-{req.export_id[-4:]}",
        dataset_path=str(dataset_dir),
        base_model_id=req.base_model_id,
        params_json=json.dumps(req.params.model_dump(exclude_none=True)),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    run_dir = settings.runs_dir / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
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
def list_runs(session: Session = Depends(get_session)):
    runs = session.exec(select(TrainRun).order_by(TrainRun.created_at.desc())).all()
    return [_to_out(r, session) for r in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "학습 런이 없습니다")
    return _to_out(run, session)


@router.get("/runs/{run_id}/history")
def run_history(run_id: str, session: Session = Depends(get_session)):
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "학습 런이 없습니다")
    events, _ = read_progress(run_id)
    return [e for e in events if e.get("phase") == "epoch"]


@router.post("/runs/{run_id}/stop", response_model=RunOut)
def stop_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "학습 런이 없습니다")
    if run.status not in TERMINAL:
        train_manager.stop(run_id)
    session.expire_all()
    return _to_out(session.get(TrainRun, run_id), session)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, session: Session = Depends(get_session)):
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "학습 런이 없습니다")

    async def stream():
        offset = 0
        idle = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, run_id, offset)
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
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "학습 런이 없습니다")
    run_dir = settings.runs_dir / run_id
    files = []
    if run_dir.exists():
        for p in sorted(run_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS:
                files.append({"name": p.name, "url": f"/api/train/runs/{run_id}/artifacts/{p.name}"})
    weights = []
    for w in ("best.pt", "last.pt"):
        if (run_dir / "weights" / w).exists():
            weights.append({"name": w, "url": f"/api/train/runs/{run_id}/weights/{w.split('.')[0]}"})
    return {"files": files, "weights": weights}


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str, session: Session = Depends(get_session)):
    if "/" in name or name.startswith("."):
        raise HTTPException(422, "잘못된 파일명입니다")
    path = settings.runs_dir / run_id / name
    if not path.exists() or path.suffix.lower() not in ARTIFACT_EXTS:
        raise HTTPException(404, "파일이 없습니다")
    return FileResponse(path)


@router.get("/runs/{run_id}/weights/{which}")
def download_weights(run_id: str, which: str, session: Session = Depends(get_session)):
    if which not in ("best", "last"):
        raise HTTPException(422, "best 또는 last만 가능합니다")
    path = settings.runs_dir / run_id / "weights" / f"{which}.pt"
    if not path.exists():
        raise HTTPException(404, "가중치가 없습니다")
    return FileResponse(path, filename=f"{run_id}_{which}.pt")


class RegisterIn(BaseModel):
    which: str = "best"
    name: str | None = None


@router.post("/runs/{run_id}/register")
def register_weights(run_id: str, req: RegisterIn, session: Session = Depends(get_session)):
    from app.api.models import _register, _to_out as model_out

    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "학습 런이 없습니다")
    if req.which not in ("best", "last"):
        raise HTTPException(422, "best 또는 last만 가능합니다")
    src = settings.runs_dir / run_id / "weights" / f"{req.which}.pt"
    if not src.exists():
        raise HTTPException(404, "가중치가 없습니다")

    # _register moves its input — hand it a copy so the run keeps its weights
    tmp = settings.models_dir / f".register_{run_id}_{req.which}.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    entry = _register(session, req.name or run.name, tmp, source="trained")
    return model_out(entry)
