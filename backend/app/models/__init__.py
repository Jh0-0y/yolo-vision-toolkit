import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | None) -> str | None:
    """Serialize a stored datetime as UTC ISO for the client.

    SQLite drops tzinfo on round-trip, so timestamps read back from the DB are
    naive even though _now() always writes UTC. We label them UTC (append the
    offset) so the frontend's `new Date(...)` converts to local time correctly
    instead of misreading a UTC value as local."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ModelEntry(SQLModel, table=True):
    __tablename__ = "model_entry"

    id: str = Field(default_factory=lambda: _uid("m"), primary_key=True)
    name: str
    # project scope; None = legacy/shared (visible in every project)
    project_id: str | None = Field(default=None, index=True)
    # JSON: {"0": "person", "1": "car", ...} as stored in model.names
    classes_json: str = "{}"
    task: str = "detect"
    created_at: datetime = Field(default_factory=_now)


class Project(SQLModel, table=True):
    __tablename__ = "project"

    id: str = Field(default_factory=lambda: _uid("p"), primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=_now)


class LabProject(SQLModel, table=True):
    """연구실 하나. 학습실 `Project` 와 나란한 최상위 단위다 — 종목·연구 주제마다
    하나씩 둔다(`basketball-adaptive-crop`). 라벨·클래스·데이터셋을 갖지 않고
    영상과 크롭 런만 갖는다. 모델은 공용 풀에서 고른다."""

    __tablename__ = "lab_project"

    id: str = Field(default_factory=lambda: _uid("lab"), primary_key=True)
    name: str
    # 새 런이 출발할 기본값 — 크롭 노브 · 크롭 비율 · 기본 검출 모델.
    # 종목마다 다르기 때문에 연구실이 갖는다.
    preset_json: str = "{}"
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
    # project scope; None = legacy/shared (visible in every project)
    project_id: str | None = Field(default=None, index=True)
    dataset_path: str  # directory containing data.yaml
    base_model_id: str
    params_json: str = "{}"
    status: str = "queued"  # queued | running | done | error | stopped
    pid: int | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    error: str | None = None
    metrics_json: str | None = None  # final epoch metrics
