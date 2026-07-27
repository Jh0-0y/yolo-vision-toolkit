"""공 추적·선수 군집 정책.

- BallTracker: 위치·속도 예측, 최대 1000ms 가림 유지, 재검출 시 점진 복귀
- select_ball: 후보 다수일 때 Track 연속성·Confidence·선수 근접도로 판별
- player_group_center: 선수 군집 중심 (중앙값 — 심판·벤치 등 outlier에 강건)
"""

import statistics

from .constants import BALL_LOST_HOLD_MS, SOURCE_WIDTH
from .types import Detection

# select_ball 점수 가중치 (구현 세부)
_W_CONTINUITY = 0.5
_W_CONFIDENCE = 0.3
_W_PLAYER_PROXIMITY = 0.2
# 재검출 시 예측→실측 점진 복귀 블렌드 비율
_REACQUIRE_BLEND = 0.5


class BallTracker:
    """공 위치·속도 상태 기계."""

    def __init__(self) -> None:
        self._x: float | None = None
        self._velocity: float = 0.0  # px/ms
        self._last_seen_ms: int | None = None
        self._last_update_ms: int | None = None
        self._was_predicting = False

    def update(self, ball: Detection | None, offset_ms: int) -> float | None:
        """이번 Sample의 유효한 공 중심 X를 반환한다. 유지 시간 초과 시 None."""
        if ball is not None:
            measured = ball.center_x
            if self._x is not None and self._last_update_ms is not None:
                dt = offset_ms - self._last_update_ms
                if self._was_predicting:
                    # 가림에서 복귀 — 예측 위치에서 실측으로 점진 복귀
                    measured = self._x * _REACQUIRE_BLEND + measured * (1 - _REACQUIRE_BLEND)
                if dt > 0:
                    self._velocity = (measured - self._x) / dt
            self._x = measured
            self._last_seen_ms = offset_ms
            self._last_update_ms = offset_ms
            self._was_predicting = False
            return self._x

        # 미검출 — 예측 유지 구간 판정
        if self._x is None or self._last_seen_ms is None:
            return None
        if offset_ms - self._last_seen_ms > BALL_LOST_HOLD_MS:
            return None

        dt = offset_ms - (self._last_update_ms or offset_ms)
        predicted = self._x + self._velocity * dt
        self._x = min(max(predicted, 0.0), float(SOURCE_WIDTH))
        self._last_update_ms = offset_ms
        self._was_predicting = True
        return self._x

    @property
    def predicted_x(self) -> float | None:
        return self._x


def select_ball(
    candidates: list[Detection],
    predicted_x: float | None,
    player_centers: list[float],
) -> Detection | None:
    """공 후보 판별 — 연속성·Confidence·선수 근접도 점수 합산."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def score(det: Detection) -> float:
        continuity = 1.0
        if predicted_x is not None:
            continuity = 1.0 - min(abs(det.center_x - predicted_x) / SOURCE_WIDTH, 1.0)
        proximity = 0.0
        if player_centers:
            nearest = min(abs(det.center_x - p) for p in player_centers)
            proximity = 1.0 - min(nearest / SOURCE_WIDTH, 1.0)
        return (
            _W_CONTINUITY * continuity
            + _W_CONFIDENCE * det.confidence
            + _W_PLAYER_PROXIMITY * proximity
        )

    return max(candidates, key=score)


def player_group_center(detections: list[Detection]) -> float | None:
    """선수 군집 중심 X — 중앙값. 선수가 없으면 None."""
    centers = [d.center_x for d in detections if d.object_type == "player"]
    if not centers:
        return None
    return statistics.median(centers)
