"""Training request/response DTOs."""

from pydantic import BaseModel, Field


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


class RegisterIn(BaseModel):
    which: str = "best"
    name: str | None = None
