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


class LiveStatus(BaseModel):
    """라이브 세션이 아직 살아 있는지. 클라이언트가 기억한 detect_id 가 여전히
    쓸 수 있는지 판단하는 유일한 근거다 — 캐시는 TTL 로 쓸려 나간다."""

    status: str  # running | done | error | cancelled | expired
    msg: str | None = None
    has_render: bool = False  # 구워 둔 오버레이 영상이 아직 있나


class ResidentModel(BaseModel):
    model_id: str
    device: str | None = None
    task: str | None = None
    classes: dict[int, str] | None = None
    weight_mb: float | None = None
    loaded_at: float | None = None
