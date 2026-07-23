"""Training runs: launch/monitor/stop ultralytics training, browse artifacts,
register trained weights back into the model registry (closing the loop)."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session, select

from app.config import settings
from app.core.inference import IMAGE_EXTS
from app.core.yolo_io import atomic_write_text
from app.db import get_session, session_scope
from app.jobs.runner import read_progress
from app.jobs.train_manager import train_manager
from app.models import ModelEntry, Project, TrainRun

router = APIRouter(prefix="/api/train", tags=["training"])

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
    project_id: str | None = None
    # dataset token: "export:{project_id}:{export_id}" | "upload:{dataset_id}"
    dataset: str
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
def list_datasets(project_id: str | None = None, session: Session = Depends(get_session)):
    """Trainable datasets: project exports + uploaded zips, newest first.

    Each item carries a `dataset` token consumed by POST /runs. When project_id
    is given, only that project's exports are listed (uploaded zips have no
    project and are always shown).
    """
    projects = {p.id: p.name for p in session.exec(select(Project)).all()}
    if project_id is not None:
        projects = {pid: name for pid, name in projects.items() if pid == project_id}
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
            if meta.get("kind", "yolo") != "yolo":
                continue  # image-only exports have no data.yaml
            out.append(
                {
                    "dataset": f"export:{pid}:{meta['id']}",
                    "name": meta.get("name") or meta["id"],
                    "source": "export",
                    "train": meta.get("train", 0),
                    "val": meta.get("val", 0),
                    "classes": meta.get("classes", 0),
                    "created_at": meta.get("created_at", ""),
                }
            )
    if settings.datasets_dir.exists():
        for meta_path in settings.datasets_dir.glob("*/dataset.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            out.append(
                {
                    "dataset": f"upload:{meta['id']}",
                    "name": meta.get("name", meta["id"]),
                    "source": "upload",
                    "train": meta.get("train", 0),
                    "val": meta.get("val", 0),
                    "classes": meta.get("classes", 0),
                    "created_at": meta.get("created_at", ""),
                }
            )
    out.sort(key=lambda d: d["created_at"], reverse=True)
    return out


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    if d.is_file():  # train may point at a txt list file
        return sum(1 for line in d.read_text().splitlines() if line.strip())
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _normalize_data_yaml(yaml_path: Path) -> dict:
    """Rewrite train/val to paths relative to the yaml, dropping any `path` key.

    Uploaded zips come from many tools — absolute paths and `path:` roots from
    other machines are re-resolved against what actually exists on disk.
    """
    root = yaml_path.parent
    data = yaml.safe_load(yaml_path.read_text()) or {}
    base = data.get("path")

    def _resolve(key: str) -> str | None:
        val = data.get(key)
        if val is None:
            return None
        p = Path(str(val))
        candidates: list[Path] = []
        if not p.is_absolute():
            candidates.append(root / p)
            if base and not Path(str(base)).is_absolute():
                candidates.append(root / str(base) / p)
        # absolute or unresolvable: try tail components against the extract root
        candidates.append(root / p.name)
        if len(p.parts) >= 2:
            candidates.append(root / Path(*p.parts[-2:]))
        for c in candidates:
            if c.exists():
                return c.relative_to(root).as_posix()
        return None

    train = _resolve("train")
    if train is None:
        raise HTTPException(422, "Cannot resolve the train path in data.yaml")
    val = _resolve("val") or train

    data.pop("path", None)
    data["train"] = train
    data["val"] = val
    atomic_write_text(yaml_path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    names = data.get("names") or {}
    n_classes = data.get("nc") or len(names)
    return {
        "train": _count_images(root / train),
        "val": _count_images(root / val),
        "classes": int(n_classes),
    }


def _extract_dataset(tmp_zip: Path, name: str) -> dict:
    dataset_id = f"u_{uuid.uuid4().hex[:10]}"
    dest = settings.datasets_dir / dataset_id
    dest.mkdir(parents=True, exist_ok=True)
    try:
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            raise HTTPException(422, "Corrupted zip file")

        # data.yaml at the root or one directory down
        yaml_path = None
        for candidate in ("data.yaml", "data.yml"):
            if (dest / candidate).exists():
                yaml_path = dest / candidate
                break
        if yaml_path is None:
            found = sorted(list(dest.glob("*/data.yaml")) + list(dest.glob("*/data.yml")))
            if found:
                yaml_path = found[0]
        if yaml_path is None:
            raise HTTPException(422, "data.yaml not found in the zip (a YOLO-format dataset is required)")

        # training_runner expects the file to be literally "data.yaml"
        if yaml_path.name != "data.yaml":
            renamed = yaml_path.with_name("data.yaml")
            yaml_path.rename(renamed)
            yaml_path = renamed

        counts = _normalize_data_yaml(yaml_path)
        if counts["train"] == 0:
            raise HTTPException(422, "No train images found")

        meta = {
            "id": dataset_id,
            "name": name,
            "yaml_dir": yaml_path.parent.relative_to(dest).as_posix(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **counts,
        }
        atomic_write_text(dest / "dataset.json", json.dumps(meta, ensure_ascii=False))
        return meta
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile, session: Session = Depends(get_session)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(422, "Only zip files can be uploaded")
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.datasets_dir / f".upload_{uuid.uuid4().hex[:8]}.zip"
    try:
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        name = file.filename.rsplit("/", 1)[-1].removesuffix(".zip")
        meta = await run_in_threadpool(_extract_dataset, tmp, name)
    finally:
        tmp.unlink(missing_ok=True)
    return {**meta, "dataset": f"upload:{meta['id']}", "source": "upload"}


def _resolve_dataset_dir(token: str) -> Path:
    parts = token.split(":")
    if parts[0] == "export" and len(parts) == 3:
        pid, eid = parts[1], parts[2]
        if any(c in pid + eid for c in "/\\.."):
            raise HTTPException(422, "Invalid dataset token")
        return settings.projects_dir / pid / "exports" / eid
    if parts[0] == "upload" and len(parts) == 2:
        uid = parts[1]
        if any(c in uid for c in "/\\.."):
            raise HTTPException(422, "Invalid dataset token")
        base = settings.datasets_dir / uid
        meta_path = base / "dataset.json"
        if meta_path.exists():
            try:
                yaml_dir = json.loads(meta_path.read_text()).get("yaml_dir", ".")
                return (base / yaml_dir).resolve()
            except (json.JSONDecodeError, OSError):
                pass
        return base
    raise HTTPException(422, f"Invalid dataset token: {token}")


@router.post("/runs", response_model=RunOut)
def create_run(req: RunCreate, session: Session = Depends(get_session)):
    dataset_dir = _resolve_dataset_dir(req.dataset)
    if not (dataset_dir / "data.yaml").exists():
        raise HTTPException(422, "Dataset not found")
    base = session.get(ModelEntry, req.base_model_id)
    if base is None:
        raise HTTPException(422, "Base model not found")
    base_pt = settings.model_dir(base.project_id, req.base_model_id) / "model.pt"
    if not base_pt.exists():
        raise HTTPException(422, "Base model file missing")
    if train_manager.has_active():
        raise HTTPException(409, "A training run is already in progress. Try again after it finishes.")

    # for export-sourced datasets the project id is embedded in the token
    project_id = req.project_id
    if project_id is None and req.dataset.startswith("export:"):
        parts = req.dataset.split(":")
        if len(parts) >= 2:
            project_id = parts[1]

    run = TrainRun(
        name=req.name or f"{base.name}-{req.dataset.rsplit(':', 1)[-1][-4:]}",
        project_id=project_id,
        dataset_path=str(dataset_dir),
        base_model_id=req.base_model_id,
        params_json=json.dumps(req.params.model_dump(exclude_none=True)),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    run_dir = settings.run_dir(run.project_id, run.id)
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
    events, _ = read_progress(run_id)
    return [e for e in events if e.get("phase") == "epoch"]


@router.get("/runs/{run_id}/per-class")
def run_per_class(run_id: str, session: Session = Depends(get_session)):
    """Per-class metrics (P/R/mAP) captured at the end of training; [] if absent
    (older runs, or the extraction failed)."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "per_class.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


@router.get("/runs/{run_id}/per-class-history")
def run_per_class_history(run_id: str, session: Session = Depends(get_session)):
    """Per-class metrics per epoch (for 'class metric over epochs' charts); []
    for older runs that predate this capture."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    path = settings.run_dir(run.project_id, run_id) / "per_class_history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@router.get("/runs/{run_id}/results")
def run_results(run_id: str, session: Session = Depends(get_session)):
    """Full per-epoch metrics from ultralytics results.csv (train+val loss, mAP,
    precision/recall, lr, time). Works for completed and past runs; [] if none."""
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    rdir = settings.run_dir(run.project_id, run_id)
    csv_path = rdir / "results.csv"
    if not csv_path.exists():
        found = sorted(rdir.glob("*/results.csv"))
        csv_path = found[0] if found else csv_path
    if not csv_path.exists():
        return []

    import csv as _csv

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for raw in _csv.DictReader(f):
            row: dict = {}
            for k, v in raw.items():
                key = (k or "").strip()
                if not key:
                    continue
                try:
                    row[key] = float(v)
                except (TypeError, ValueError):
                    row[key] = v
            rows.append(row)
    return rows


@router.post("/runs/{run_id}/stop", response_model=RunOut)
def stop_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    if run.status not in TERMINAL:
        train_manager.stop(run_id)
    session.expire_all()
    return _to_out(session.get(TrainRun, run_id), session)


@router.delete("/runs/{run_id}")
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
    return {"ok": True}


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, session: Session = Depends(get_session)):
    if session.get(TrainRun, run_id) is None:
        raise HTTPException(404, "Training run not found")

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
    run = session.get(TrainRun, run_id)
    if run is None:
        raise HTTPException(404, "Training run not found")
    run_dir = settings.run_dir(run.project_id, run_id)
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


class RegisterIn(BaseModel):
    which: str = "best"
    name: str | None = None


@router.post("/runs/{run_id}/register")
def register_weights(run_id: str, req: RegisterIn, session: Session = Depends(get_session)):
    from app.api.models import _register, _to_out as model_out

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
