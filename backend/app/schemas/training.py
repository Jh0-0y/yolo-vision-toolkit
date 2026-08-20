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


class TilingIn(BaseModel):
    """학습 전처리 타일링. 끄면(`enabled=False`) 지금까지와 완전히 같다."""

    enabled: bool = False
    tile_size: int = Field(default=640, ge=64, le=4096)
    stride: int = Field(default=480, ge=32, le=4096)
    min_visibility: float = Field(default=0.6, ge=0, le=1)
    # 포지티브 타일 수 대비 비율 (0.1 = 10%). 분할마다 따로 적용된다.
    negative_ratio: float = Field(default=0.1, ge=0)
    keep_all_negatives: bool = False
    seed: int = 0


class SplitPreviewOut(BaseModel):
    split: str
    positive: int = 0
    hard: int = 0
    incidental: int = 0
    negative_kept: int = 0
    excluded: int = 0
    total: int = 0
    overshoot: bool = False


class TilingPreviewOut(BaseModel):
    splits: list[SplitPreviewOut] = []


class TilingPreviewIn(BaseModel):
    dataset: str  # "dataset:{project_id}:{dataset_id}"
    tiling: TilingIn


class RunCreate(BaseModel):
    name: str | None = None
    project_id: str | None = None
    # dataset token: "dataset:{project_id}:{dataset_id}"
    dataset: str
    base_model_id: str
    device: str | None = None
    params: TrainParams = Field(default_factory=TrainParams)
    tiling: TilingIn = Field(default_factory=TilingIn)


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
