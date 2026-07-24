"""Dataset-export request/response DTOs."""

from typing import Literal

from pydantic import BaseModel, Field


class ExportCreate(BaseModel):
    kind: Literal["yolo", "images"] = "yolo"
    val_split: float = Field(default=0.2, ge=0.0, le=0.9)
    seed: int = 42
    names: list[str] | None = None


class ExportOut(BaseModel):
    id: str
    name: str = ""
    kind: str = "yolo"
    created_at: str
    val_split: float = 0.0
    seed: int = 0
    train: int = 0
    val: int = 0
    count: int = 0
    classes: int = 0
    size_bytes: int = 0


class ExportRename(BaseModel):
    name: str
