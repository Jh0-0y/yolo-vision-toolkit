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

    @property
    def center_y(self) -> float:
        return self.bbox_y + self.bbox_height / 2


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


# 스펙 target_type 표기 — 내부 소문자 → 계약(camelCase JSON)의 UPPER_SNAKE
_SPEC_TARGET_TYPE = {
    "ball": "BALL",
    "ball_player": "BALL_PLAYER",
    "player_group": "PLAYER_GROUP",
    "center": "CENTER",
}

# 이 툴킷에는 게임/잡 식별자 체계가 없다 — 스펙 필수 컬럼은 플레이스홀더로 채운다.
SPEC_PLACEHOLDER_IDS = {
    "jobId": "job_local",
    "gameId": "game_local",
    "clipCandidateId": "clip_local",
    "sourceContentOutputId": "cnto_local",
    "sourceMediaAssetId": "mast_local",
}


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
    debug: list = field(default_factory=list)  # 대상 하이라이트용 시점별 선택 공/소유선수 bbox (opt-in)

    def to_dict(self, include_internal: bool = False) -> dict:
        """계약 스키마(camelCase)로 직렬화한다.

        기본은 스펙 그대로(schemaVersion·id·source/crop 중첩·keyframes·summary).
        include_internal=True면 라이브 오버레이용 내부 확장 필드(samples·debug —
        스펙 외)를 함께 담는다. 파일로 내보내는 crop.json은 스펙만 담아야 한다.
        """
        out: dict = {
            "schemaVersion": 1,
            **SPEC_PLACEHOLDER_IDS,
            "source": {
                "width": self.source_width,
                "height": self.source_height,
                "durationMs": self.duration_ms,
            },
            "crop": {
                "width": self.crop_width,
                "height": self.crop_height,
                "y": self.crop_y,
                "interpolation": "LINEAR",
            },
            "keyframes": [
                {
                    "videoOffsetMs": k.video_offset_ms,
                    "x": k.x,
                    "targetType": _SPEC_TARGET_TYPE.get(k.target_type, k.target_type.upper()),
                    "confidence": round(k.confidence, 4),
                }
                for k in self.keyframes
            ],
            "summary": {
                "keyframeCount": self.summary.get("keyframe_count", len(self.keyframes)),
                "ballTrackingCoverage": self.summary.get("ball_tracking_coverage", 0.0),
                "playerTrackingCoverage": self.summary.get("player_tracking_coverage", 0.0),
                "fallbackCoverage": self.summary.get("fallback_coverage", 0.0),
            },
        }
        if include_internal:
            out["samples"] = [asdict(s) for s in self.samples]
            out["debug"] = list(self.debug)
        return out

    def to_json(self, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
