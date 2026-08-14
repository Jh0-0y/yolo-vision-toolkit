"""연구실 크롭 런 요청·응답 DTO."""

from pydantic import BaseModel, Field


class CropDetector(BaseModel):
    """모델 하나와 그 추론 방식. 런처에서 여러 개를 고를 수 있다."""

    model_id: str
    mode: str = "full"  # full | tiled
    conf: float | None = None
    imgsz: int = 1920
    tile_size: int = 640
    stride: int = 480
    merge_iou: float = 0.5


class CropToggles(BaseModel):
    """그리기 도구 — 개별 토글. **하나도 켜지 않으면** wide.mp4 는 원본 사본이다."""

    obj_boxes: bool = False  # 추론 박스
    player_trails: bool = False  # 선수 궤적
    ball_trail: bool = False  # 공 궤적
    target_highlight: bool = False  # 타깃 (선택된 공·소유선수)
    crop_box: bool = True  # 크롭 박스
    dead_zone: bool = False  # 데드존
    center_line: bool = True  # 중앙선


class LabCropCreate(BaseModel):
    """런처가 보내는 것 전부. 연구실 프리셋에서 출발하되 런마다 바꿀 수 있다."""

    name: str = ""  # 테스트 이름 — 비우면 원본 영상 이름
    source_video_id: str
    detectors: list[CropDetector] = Field(default_factory=list)
    overrides: dict = Field(default_factory=dict)  # adaptive-crop 튜닝 노브
    crop_w: int = Field(default=608, ge=2)  # 크롭 창 가로 px
    crop_h: int = Field(default=1080, ge=2)  # 크롭 창 세로 px
    toggles: CropToggles = CropToggles()
    device: str | None = None


class LabCropRunOut(BaseModel):
    id: str
    name: str
    created_at: str
    source_video_id: str
    source_name: str
    models: list[dict] = []  # 런이 실제로 쓴 모델 — 이름과 추론 방식
    overrides: dict = {}
    crop_w: int = 608
    crop_h: int = 1080
    # 소스에 맞춰 실제로 쓰인 창 — 요청값이 소스보다 크면 clamp 된 값이 여기 온다
    applied_w: int = 0
    applied_h: int = 0
    toggles: dict = {}
    wide_kind: str = ""  # link | copy | render
    status: str
    error: str | None = None
    summary: dict | None = None
    has_crop_json: bool = False
    has_wide: bool = False
    has_cut: bool = False
    size_bytes: int = 0


class LabCropRename(BaseModel):
    name: str


class LabCropDuplicate(BaseModel):
    """"이 설정 그대로 다시, 모델만 바꿔서" — 모델 업그레이드 전후 비교용.

    준 것만 바꾸고 나머지는 원본 런의 설정을 그대로 쓴다.
    """

    name: str = ""
    detectors: list[CropDetector] | None = None
