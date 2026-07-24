"""Model-registry request/response DTOs."""

from pydantic import BaseModel


class ModelOut(BaseModel):
    id: str
    name: str
    classes: dict[int, str]
    task: str
    created_at: str
    source: str = "upload"


class OfficialRequest(BaseModel):
    name: str  # e.g. "yolo26n"
    project_id: str | None = None


class ModelPatch(BaseModel):
    name: str
