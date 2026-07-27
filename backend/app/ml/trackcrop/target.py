"""Target 선정 및 Target X 계산.

우선순위: 공 → 공+선수 군집 가중 중심 → 선수 군집 → 직전 Target 유지 → 화면 중앙.

  공 유효:            target_center_x = ball_center_x
  공+선수 군집 유효:   target_center_x = ball×0.7 + player_group×0.3
  선수 군집만 유효:    target_center_x = player_group_center_x
  전체 미검출:        previous_target_center_x, 장시간(1500ms 초과) 미검출이면 960
"""

from .constants import (
    BALL_WEIGHT,
    CENTER_FALLBACK_X,
    CROP_WIDTH,
    PLAYER_GROUP_WEIGHT,
    PLAYER_LOST_HOLD_MS,
)
from .tracking import BallTracker, player_group_center, select_ball
from .types import Detection, TargetSample, TargetType

# 유지·Fallback 구간의 confidence 감쇠 비율 (구현 세부)
_CONFIDENCE_DECAY = 0.9


def resolve_targets(samples: list[tuple[int, list[Detection]]]) -> list[TargetSample]:
    """Sample 시퀀스에서 시점별 Target X를 결정한다."""
    tracker = BallTracker()
    results: list[TargetSample] = []

    last_valid_x = float(CENTER_FALLBACK_X)
    last_confidence = 0.0
    last_observed_ms: int | None = None  # 마지막으로 공 또는 선수를 관측한 시각

    for offset_ms, detections in samples:
        balls = [d for d in detections if d.object_type == "ball"]
        player_centers = [d.center_x for d in detections if d.object_type == "player"]
        group_x = player_group_center(detections)

        ball = select_ball(balls, tracker.predicted_x, player_centers)
        ball_x = tracker.update(ball, offset_ms)

        if ball is not None or group_x is not None:
            last_observed_ms = offset_ms

        if ball_x is not None:
            if ball is not None:
                confidence = ball.confidence
            else:
                confidence = last_confidence * _CONFIDENCE_DECAY
            if group_x is not None and abs(ball_x - group_x) <= CROP_WIDTH:
                # 공과 선수 군집을 한 화면에 담을 수 있으면 가중 중심
                target_x = ball_x * BALL_WEIGHT + group_x * PLAYER_GROUP_WEIGHT
                target_type = TargetType.BALL_PLAYER
            else:
                target_x = ball_x
                target_type = TargetType.BALL
        elif group_x is not None:
            # 공 미검출 — 선수 군집 Fallback
            target_x = group_x
            target_type = TargetType.PLAYER_GROUP
            confidence = max(d.confidence for d in detections if d.object_type == "player")
        elif last_observed_ms is not None and offset_ms - last_observed_ms <= PLAYER_LOST_HOLD_MS:
            # 짧은 전체 미검출 — 직전 Target 유지
            target_x = last_valid_x
            target_type = TargetType.PLAYER_GROUP
            confidence = last_confidence * _CONFIDENCE_DECAY
        else:
            # 장시간 미검출 — 화면 중앙
            target_x = float(CENTER_FALLBACK_X)
            target_type = TargetType.CENTER
            confidence = 0.0

        last_valid_x = target_x
        last_confidence = confidence
        results.append(
            TargetSample(
                video_offset_ms=offset_ms,
                target_center_x=target_x,
                target_type=target_type.value,
                confidence=confidence,
            )
        )

    return results
