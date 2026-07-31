"""공 추적·선수 군집 정책.

- BallTracker: 위치·속도 예측, 최대 1000ms 가림 유지, 재검출 시 점진 복귀
- select_ball: 후보 다수일 때 Track 연속성·Confidence·선수 근접도로 판별
- player_group_center: 선수 군집 중심 (중앙값 — 심판·벤치 등 outlier에 강건)
"""

import statistics

from .config import TrackcropConfig
from .constants import SOURCE_WIDTH
from .types import Detection


class BallTracker:
    """공 위치·속도 상태 기계."""

    def __init__(self, cfg: TrackcropConfig | None = None) -> None:
        self._cfg = cfg or TrackcropConfig()
        self._x: float | None = None  # 외삽 예측 — 크롭 타깃 부드럽게 하는 용도
        self._velocity: float = 0.0  # px/ms
        self._last_seen_x: float | None = None  # 마지막 실측 위치 — 공 선정 앵커(드리프트 안 함)
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
                    blend = self._cfg.reacquire_blend
                    measured = self._x * blend + measured * (1 - blend)
                if dt > 0:
                    self._velocity = (measured - self._x) / dt
            self._x = measured
            self._last_seen_x = ball.center_x  # 게이트·연속성 앵커 = 실측
            self._last_seen_ms = offset_ms
            self._last_update_ms = offset_ms
            self._was_predicting = False
            return self._x

        # 미검출 — 예측 유지 구간 판정
        if self._x is None or self._last_seen_ms is None:
            return None
        if offset_ms - self._last_seen_ms > self._cfg.ball_lost_hold_ms:
            self._x = None
            self._velocity = 0.0
            self._was_predicting = False
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

    @property
    def last_seen_x(self) -> float | None:
        """마지막으로 실제 검출된 공 위치 — 공 선정(게이트·연속성)의 기준."""
        return self._last_seen_x

    @property
    def velocity(self) -> float:
        """현재 추정 공 속도 (px/ms) — DET-05의 '공과 함께 이동하는 선수' 판정용."""
        return self._velocity

    def gate_radius(self, offset_ms: int) -> float | None:
        """공 선정 게이트 반경 — base + 최대속도×놓친시간. match_max_jump_px<=0이면 None."""
        if self._cfg.match_max_jump_px <= 0 or self._last_seen_ms is None:
            return None
        missed = max(0, offset_ms - self._last_seen_ms)
        return self._cfg.match_max_jump_px + self._cfg.max_move_px_per_second * missed / 1000


def select_ball(
    candidates: list[Detection],
    expected_x: float | None,
    player_centers: list[float],
    prev_ball_id: int | None = None,
    cfg: TrackcropConfig | None = None,
    gate_radius: float | None = None,
) -> Detection | None:
    """공 후보 판별 — Track 연속성·Confidence 점수 합산 (DET-04).

    expected_x = 마지막 실측 위치(공 선정 기준). gate_radius가 주어지면 그 밖의
    후보를 배제한다. 선수 근접도는 '기준 위치가 없는 첫 획득'에서만 프라이어로 쓴다.
    """
    if not candidates:
        return None

    cfg = cfg or TrackcropConfig()

    if gate_radius is not None and expected_x is not None:
        gated = [c for c in candidates if abs(c.center_x - expected_x) <= gate_radius]
        if not gated:
            return None
        candidates = gated

    if len(candidates) == 1:
        return candidates[0]

    def score(det: Detection) -> float:
        if prev_ball_id is not None and det.track_id == prev_ball_id:
            continuity = 1.0  # 같은 track_id — 최우선
        elif expected_x is not None:
            continuity = 1.0 - min(abs(det.center_x - expected_x) / SOURCE_WIDTH, 1.0)
        else:
            continuity = 1.0
        s = cfg.w_continuity * continuity + cfg.w_confidence * det.confidence
        if expected_x is None and player_centers:
            nearest = min(abs(det.center_x - p) for p in player_centers)
            s += cfg.w_player_proximity * (1.0 - min(nearest / SOURCE_WIDTH, 1.0))
        return s

    return max(candidates, key=score)


def player_group_center(detections: list[Detection]) -> float | None:
    """선수 군집 중심 X — 중앙값. 선수가 없으면 None."""
    centers = [d.center_x for d in detections if d.object_type == "player"]
    if not centers:
        return None
    return statistics.median(centers)
