"""Auto-labeling job request/response DTOs."""

from pydantic import BaseModel


class AutoLabelDetector(BaseModel):
    """모델 하나와 그 추론 방식. 여러 개를 고를 수 있다.

    연구실의 `CropDetector` 와 필드가 닮았지만 **일부러 따로 둔다** — 학습실은
    연구실 DTO 에 묶이지 않는다(`docs/agents/architecture.md`).
    """

    model_id: str
    mode: str = "full"  # full | tiled
    imgsz: int = 1920  # full — 원본 프레임이 1920 이라 줄이지 않는 것이 기본이다
    tile_size: int = 640  # tiled
    stride: int = 480
    merge_iou: float = 0.5  # 타일 경계 병합 — 모델 간 병합(iou_wbf)과 별개
    border_margin_px: int = 4


class JobCreate(BaseModel):
    # 오토라벨링의 대상 — 이미지도 라벨도 클래스도 이 데이터셋 안에 있다
    dataset_id: str
    detectors: list[AutoLabelDetector]
    conf: float = 0.4
    iou_wbf: float = 0.55
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
