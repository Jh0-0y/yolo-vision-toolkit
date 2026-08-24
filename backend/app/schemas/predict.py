"""Test-playground (inference) DTOs."""

from pydantic import BaseModel


class PredictBox(BaseModel):
    cls: int
    name: str
    score: float
    xyxyn: list[float]  # [x1, y1, x2, y2] normalized [0,1]
    model_ids: list[str]  # which models contributed (DB ids)
    agree: int  # distinct models that agreed on this box


class PredictResponse(BaseModel):
    task: str
    names: dict[int, str]  # global class id -> name
    boxes: list[PredictBox]
    device: str  # device actually used
    floor: float  # detections returned down to this conf (slider filters above it)


class TestJobStart(BaseModel):
    job_id: str


class ResidentModel(BaseModel):
    model_id: str
    device: str | None = None
    task: str | None = None
    classes: dict[int, str] | None = None
    weight_mb: float | None = None
    loaded_at: float | None = None


class BenchmarkEntryIn(BaseModel):
    """검출기 엔트리 하나 — 모델과 **그 모델의 추론 방식**.

    `conf` 는 여기 없다. 벤치마크는 모든 엔트리를 **하나의 전역 동작점**에서 채점한다 —
    엔트리마다 다르면 P/R 을 나란히 놓을 수 없다.
    """

    model_id: str
    mode: str = "full"  # full | tiled
    imgsz: int = 1920  # full — 원본 프레임이 1920 이라 줄이지 않는 것이 기본이다
    tile_size: int = 640
    stride: int = 480
    merge_iou: float = 0.5
    border_margin_px: int = 4


class BenchmarkStart(BaseModel):
    dataset: str  # "dataset:{project_id}:{dataset_id}"
    entries: list[BenchmarkEntryIn]
    conf: float = 0.4
    iou: float = 0.5  # pred↔GT 매칭 IoU
    device: str | None = None


class BenchmarkOut(BaseModel):
    id: str
    created_at: str
    dataset_name: str = ""
    dataset: str = ""
    entries: int = 0
    conf: float = 0.0
    iou: float = 0.0
    status: str = "running"
    error: str | None = None
