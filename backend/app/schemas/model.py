"""Model-registry request/response DTOs."""

from pydantic import BaseModel


class ModelOut(BaseModel):
    id: str
    name: str
    classes: dict[int, str]
    task: str
    created_at: str
    source: str = "upload"
    # 어느 프로젝트에서 나온 모델인지 — None 은 공유 풀이다. 연구실은 프로젝트가
    # 없어 전체 목록에서 고르므로, 출처를 묶어 보여주려면 이 둘이 필요하다.
    project_id: str | None = None
    project_name: str | None = None


class OfficialRequest(BaseModel):
    name: str  # e.g. "yolo26n"
    project_id: str | None = None


class ModelPatch(BaseModel):
    name: str
