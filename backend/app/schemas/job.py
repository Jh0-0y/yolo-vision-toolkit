"""Auto-labeling job request/response DTOs."""

from pydantic import BaseModel


class JobCreate(BaseModel):
    model_ids: list[str]
    conf: float = 0.4
    iou_wbf: float = 0.55
    imgsz: int = 640
    batch_size: int = 16
    names: list[str] | None = None
    # class name -> max boxes per image (unspecified classes are uncapped/300)
    max_boxes_per_class: dict[str, int] | None = None


class JobOut(BaseModel):
    id: str
    project_id: str
    status: str
    config: dict
    result: dict | None
    error: str | None
    created_at: str
