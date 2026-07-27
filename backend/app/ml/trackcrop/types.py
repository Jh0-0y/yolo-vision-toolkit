"""파이프라인 데이터 타입.

원본에서 LivePick 전용 pydantic 스키마(CropPlan 등)를 걷어내고
순수 dataclass로 재정의했다. Detection 상세는 좌표 계산에만 쓰고 저장하지 않는다.
"""

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class TargetType(StrEnum):
    BALL = "ball"
    BALL_PLAYER = "ball_player"
    PLAYER_GROUP = "player_group"
    CENTER = "center"


@dataclass(slots=True)
class Detection:
    """단일 Sample의 객체 탐지 결과."""

    object_type: str  # "ball" | "player"
    track_id: int | None
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    video_offset_ms: int

    @property
    def center_x(self) -> float:
        return self.bbox_x + self.bbox_width / 2


@dataclass(slots=True)
class TargetSample:
    """Sample 시점의 Target 결정 결과."""

    video_offset_ms: int
    target_center_x: float
    target_type: str  # TargetType 값
    confidence: float


@dataclass(slots=True)
class Keyframe:
    """Crop 궤적의 Keyframe — 사이 구간은 소비 측에서 선형 보간한다."""

    video_offset_ms: int  # Clip 시작 = 0 기준 상대 시간
    x: int  # Crop 영역 왼쪽 X (Source Pixel, 0..CROP_X_MAX)
    target_type: str
    confidence: float


@dataclass(slots=True)
class CropResult:
    """추적·크롭 좌표 계산 결과.

    keyframes  — RDP로 축약된 Crop X 궤적 (소비 측 선형 보간용)
    samples    — 100ms 격자 전 구간의 Target/Crop 좌표 (프레임별 크롭이 필요할 때)
    summary    — keyframe 수·추적 커버리지 통계
    """

    source_width: int
    source_height: int
    duration_ms: int
    crop_width: int
    crop_height: int
    crop_y: int
    keyframes: list[Keyframe]
    samples: list[TargetSample] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
