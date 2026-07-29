"""Target 선정 및 Target X 계산.

우선순위: 공 → 공+선수 군집 가중 중심 → 공 든 선수(DET-05) → 선수 군집 → 직전 유지 → 화면 중앙.

  공 유효:            target_center_x = ball_center_x
  공+선수 군집 유효:   target_center_x = ball×0.7 + player_group×0.3
  공 미검출:          공과 가장 가깝고 함께 이동하던 선수(track_id)를 추적 (DET-05)
  공·해당 선수 미검출:  player_group_center_x (선수 군집 중심)
  전체 미검출:        previous_target_center_x, 장시간(1500ms 초과) 미검출이면 960
"""

from .constants import (
    BALL_WEIGHT,
    CENTER_FALLBACK_X,
    CROP_WIDTH,
    PLAYER_GROUP_WEIGHT,
    PLAYER_LOST_HOLD_MS,
    SOURCE_WIDTH,
)
from .tracking import BallTracker, player_group_center, select_ball
from .types import Detection, TargetSample, TargetType

# 유지·Fallback 구간의 confidence 감쇠 비율 (구현 세부)
_CONFIDENCE_DECAY = 0.9

# DET-05 carrier(공 든 선수) 선정 가중치·정규화 기준 (구현 세부)
_W_CARRIER_PROXIMITY = 0.6  # 공과의 근접도
_W_CARRIER_VELOCITY = 0.4  # 공과의 이동 유사도
_VELOCITY_SCALE = 0.5  # px/ms — 속도차 정규화 기준 (이 이상 차이나면 유사도 0)


def _find_by_track_id(players: list[Detection], track_id: int | None) -> Detection | None:
    """현재 프레임 선수 중 track_id가 일치하는 Detection을 찾는다."""
    if track_id is None:
        return None
    for player in players:
        if player.track_id == track_id:
            return player
    return None


def _select_carrier(
    players: list[Detection],
    ball: Detection,
    ball_velocity: float,
    player_velocity: dict[int, float],
) -> int | None:
    """DET-05: 공과 가장 가깝고(근접도) 함께 이동하는(속도 유사도) 선수의 track_id."""
    candidates = [p for p in players if p.track_id is not None]
    if not candidates:
        return None

    def score(player: Detection) -> float:
        proximity = 1.0 - min(abs(player.center_x - ball.center_x) / SOURCE_WIDTH, 1.0)
        velocity_sim = 0.0
        pv = player_velocity.get(player.track_id)
        if pv is not None and abs(ball_velocity) > 1e-9:
            velocity_sim = 1.0 - min(abs(pv - ball_velocity) / _VELOCITY_SCALE, 1.0)
        return _W_CARRIER_PROXIMITY * proximity + _W_CARRIER_VELOCITY * velocity_sim

    return max(candidates, key=score).track_id


def resolve_targets(samples: list[tuple[int, list[Detection]]]) -> list[TargetSample]:
    """Sample 시퀀스에서 시점별 Target X를 결정한다."""
    tracker = BallTracker()
    results: list[TargetSample] = []

    last_valid_x = float(CENTER_FALLBACK_X)
    last_confidence = 0.0
    last_observed_ms: int | None = None  # 마지막으로 공 또는 선수를 관측한 시각
    carrier_id: int | None = None  # DET-05: 공 든 선수의 track_id
    prev_ball_id: int | None = None  # DET-04: 직전 공의 track_id (연속성)
    player_prev: dict[int, tuple[int, float]] = {}  # track_id → (last_ms, last_center_x)

    for offset_ms, detections in samples:
        balls = [d for d in detections if d.object_type == "ball"]
        players = [d for d in detections if d.object_type == "player"]
        player_centers = [p.center_x for p in players]
        group_x = player_group_center(detections)

        # 선수별 속도 갱신 (track_id 기준) — DET-05 이동 유사도에 사용
        player_velocity: dict[int, float] = {}
        for p in players:
            if p.track_id is None:
                continue
            prev = player_prev.get(p.track_id)
            if prev is not None and offset_ms - prev[0] > 0:
                player_velocity[p.track_id] = (p.center_x - prev[1]) / (offset_ms - prev[0])
            player_prev[p.track_id] = (offset_ms, p.center_x)

        ball = select_ball(balls, tracker.predicted_x, player_centers, prev_ball_id)
        ball_x = tracker.update(ball, offset_ms)
        if ball is not None:
            prev_ball_id = ball.track_id  # DET-04: 연속성용 직전 track_id 유지

        # DET-05: 공이 보이면 공과 가깝고 함께 이동하는 선수를 '공 든 선수'로 갱신
        if ball is not None and players:
            cid = _select_carrier(players, ball, tracker.velocity, player_velocity)
            if cid is not None:
                carrier_id = cid

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
        else:
            # 공 미검출 — DET-05: 공 든 선수를 우선 추적
            carrier = _find_by_track_id(players, carrier_id)
            if carrier is not None:
                target_x = carrier.center_x
                target_type = TargetType.PLAYER_GROUP
                confidence = carrier.confidence
            elif group_x is not None:
                # 공·공 든 선수 모두 미검출 — 선수 군집 Fallback
                target_x = group_x
                target_type = TargetType.PLAYER_GROUP
                confidence = max(p.confidence for p in players)
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
