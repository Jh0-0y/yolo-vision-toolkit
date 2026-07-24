"""Project request/response DTOs."""

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str


class StatsOut(BaseModel):
    images: int
    labeled: int  # label file exists (may be empty = a "no class" negative)
    reviewed: int
    classes: list[dict]


class ReviewedIn(BaseModel):
    reviewed: bool


class DeleteImagesIn(BaseModel):
    names: list[str]
