"""Crop X 계산과 Keyframe 축약.

  raw_x = target_center_x - CROP_WIDTH / 2
  x = clamp(raw_x, 0, CROP_X_MAX), 정수 반올림

Keyframe 규칙:
- 첫·마지막 Frame 필수 포함
- 방향/속도 변화 지점 유지, 동일 X 구간 제거
- 소비 측(렌더러)이 Keyframe 사이를 선형 보간
"""

from . import constants
from .types import Keyframe, TargetSample


def to_crop_x(target_center_x: float) -> int:
    """Target 중심 → Crop 왼쪽 X (Clamp + 정수 반올림)."""
    raw_x = target_center_x - constants.CROP_WIDTH / 2
    return round(min(max(raw_x, constants.CROP_X_MIN), constants.CROP_X_MAX))


# Keyframe 사이 선형 보간이 원 궤적을 재현해야 하는 허용 오차 (구현 세부)
_RDP_EPSILON_PX = 8.0


def reduce_keyframes(samples: list[TargetSample]) -> list[Keyframe]:
    """안정화된 Sample 시퀀스를 Keyframe으로 축약.

    Ramer-Douglas-Peucker 단순화 — "Keyframe 사이 선형 보간이 원 궤적을
    epsilon(8px) 이내로 재현"을 보장하므로 소비 측의 선형 보간 규칙과
    부합한다. 첫·마지막 포함, 동일 X 구간·직선 구간은 자동 축약,
    방향·속도 변화 지점은 자동 보존된다.
    """
    if not samples:
        return []

    points = [(s.video_offset_ms, to_crop_x(s.target_center_x)) for s in samples]
    keep_indices = sorted(_rdp_indices(points, 0, len(points) - 1))

    return [
        Keyframe(
            video_offset_ms=samples[i].video_offset_ms,
            x=points[i][1],
            target_type=samples[i].target_type,
            confidence=round(samples[i].confidence, 4),
        )
        for i in keep_indices
    ]


def _rdp_indices(points: list[tuple[int, int]], start: int, end: int) -> set[int]:
    """(t, x) 폴리라인에서 유지할 인덱스 집합 — 재귀 RDP."""
    if end <= start + 1:
        return {start, end}

    t0, x0 = points[start]
    t1, x1 = points[end]
    max_error, max_index = 0.0, start
    for i in range(start + 1, end):
        t, x = points[i]
        interpolated = x0 + (x1 - x0) * (t - t0) / (t1 - t0)
        error = abs(x - interpolated)
        if error > max_error:
            max_error, max_index = error, i

    if max_error <= _RDP_EPSILON_PX:
        return {start, end}
    return _rdp_indices(points, start, max_index) | _rdp_indices(points, max_index, end)
