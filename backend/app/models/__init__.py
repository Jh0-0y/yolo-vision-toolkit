import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ModelEntry(SQLModel, table=True):
    __tablename__ = "model_entry"

    id: str = Field(default_factory=lambda: _uid("m"), primary_key=True)
    name: str
    # JSON: {"0": "person", "1": "car", ...} as stored in model.names
    classes_json: str = "{}"
    task: str = "detect"
    created_at: datetime = Field(default_factory=_now)


class Project(SQLModel, table=True):
    __tablename__ = "project"

    id: str = Field(default_factory=lambda: _uid("p"), primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    __tablename__ = "job"

    id: str = Field(default_factory=lambda: _uid("j"), primary_key=True)
    project_id: str = Field(index=True)
    kind: str = "label"  # label | train
    status: str = "queued"  # queued | running | done | error | cancelled
    config_json: str = "{}"
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    error: str | None = None
    result_json: str | None = None


class TrainRun(SQLModel, table=True):
    __tablename__ = "train_run"

    id: str = Field(default_factory=lambda: _uid("t"), primary_key=True)
    name: str
    dataset_path: str  # directory containing data.yaml
    base_model_id: str
    params_json: str = "{}"
    status: str = "queued"  # queued | running | done | error | stopped
    pid: int | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    error: str | None = None
    metrics_json: str | None = None  # final epoch metrics
