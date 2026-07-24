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


class VideoUpload(BaseModel):
    video_id: str
    frame_count: int


class ResidentModel(BaseModel):
    model_id: str
    device: str | None = None
    task: str | None = None
    classes: dict[int, str] | None = None
    weight_mb: float | None = None
    loaded_at: float | None = None
