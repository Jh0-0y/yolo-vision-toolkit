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
    # augmentation — unset (None) falls back to the ultralytics default
    fliplr: float | None = Field(default=None, ge=0, le=1)
    flipud: float | None = Field(default=None, ge=0, le=1)
    degrees: float | None = Field(default=None, ge=0, le=180)
    scale: float | None = Field(default=None, ge=0, le=1)
    mosaic: float | None = Field(default=None, ge=0, le=1)
    mixup: float | None = Field(default=None, ge=0, le=1)
    copy_paste: float | None = Field(default=None, ge=0, le=1)


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


class DatasetPatch(BaseModel):
    # toggle self-delete-after-training on an uploaded dataset
    auto_delete: bool


class RegisterIn(BaseModel):
    which: str = "best"
    name: str | None = None
