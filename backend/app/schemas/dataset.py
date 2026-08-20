"""데이터셋 요청·응답 DTO."""

from typing import Any

from pydantic import BaseModel


class DatasetCreate(BaseModel):
    name: str


class DatasetPatch(BaseModel):
    name: str


class DatasetOut(BaseModel):
    id: str
    name: str
    created_at: str
    # 수치는 저장하지 않고 파일에서 그때 센다 — 목록과 실제가 어긋나지 않게
    images: int = 0
    unreviewed: int = 0
    reviewed: int = 0
    unassigned: int = 0  # 검수완료인데 아직 train/val/test 어디에도 없는 것
    train: int = 0
    val: int = 0
    test: int = 0


class DatasetImageOut(BaseModel):
    name: str
    stem: str
    thumb: str
    url: str
    labeled: bool
    reviewed: bool
    split: str | None = None  # 검수완료가 아니면 언제나 None
    boxes: list[dict] = []
    created_at: float = 0.0


class DatasetImagesOut(BaseModel):
    total: int
    page: int = 1
    size: int = 60
    items: list[DatasetImageOut] = []
    names: list[str] | None = None  # names_only=true 일 때만
    model_config = {"extra": "allow"}


class ReviewedIn(BaseModel):
    reviewed: bool


class DeleteImagesIn(BaseModel):
    names: list[str]


class SplitIn(BaseModel):
    """비율은 `8:1:1` 처럼 정수로 줘도 된다 — 합이 1이 되도록 맞춘다."""

    train: float = 8
    val: float = 1
    test: float = 1
    seed: int = 0
    # 기본은 미할당만 나눈다. 전부 다시 섞으려면 **명시**해야 한다 —
    # 그러지 않으면 test 가 train 으로 새어 이전 평가와 비교가 안 된다.
    reassign_all: bool = False


class ExportIn(BaseModel):
    kind: str = "train"  # train(train+val) | test | all(검수완료 전부)
    class_ids: list[int] | None = None  # None 이면 이 데이터셋의 클래스 전부


class ExportOut(BaseModel):
    id: str
    kind: str
    count: int = 0
    train: int = 0
    val: int = 0
    test: int = 0
    classes: int = 0
    linked: int = 0  # 하드링크로 들어간 이미지 수 — 복사가 아니라는 증거
    copied: int = 0
    size_bytes: int = 0


class AutoLabelIn(BaseModel):
    """오토라벨링 — 모델 1~N개 앙상블. 대상은 이 데이터셋의 미검수 이미지다."""

    model_ids: list[str]
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 1280
    only_unlabeled: bool = False
    params: dict[str, Any] = {}
