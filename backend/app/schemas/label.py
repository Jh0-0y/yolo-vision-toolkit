"""Label-editing request DTOs."""

from pydantic import BaseModel, Field


class BoxIn(BaseModel):
    id: str | None = None
    cls: int
    xyxy_n: list[float] = Field(min_length=4, max_length=4)
    score: float | None = None
    status: str | None = None
    reason: str | None = None
    sources: list[dict] | None = None


class LabelsIn(BaseModel):
    boxes: list[BoxIn]
