"""런타임 튜닝 설정.

`constants.py`는 파이프라인의 기본 정책값이다. 실무에서는 config로 주입하고,
연구실(Lab) 튜닝 도구에서는 요청마다 일부 값을 오버라이드해 크롭 거동을 실험한다.

`TrackcropConfig`는 그 값들을 한 객체로 모은 것이다 — 기본값은 constants(및 각
모듈의 구현 세부 상수)에서 가져오고, `resolve_config(overrides)`로 일부만 덮어쓴다.
파생값(데드존 반폭 등)은 import 시점이 아니라 **호출 시점**에 계산되도록 property로 둔다.

지오메트리(SOURCE/CROP 규격)와 imgsz는 스펙 고정이라 여기서 다루지 않는다.
"""

from dataclasses import dataclass, fields

from . import constants


@dataclass(frozen=True)
class TrackcropConfig:
    """추적·안정화·선정에 쓰이는 튜닝 값 묶음. 기본값 = constants + 구현 세부 상수."""

    # --- 핵심 ---
    dead_zone_width: float = constants.DEAD_ZONE_WIDTH  # 208
    sampling_interval_ms: int = constants.SAMPLING_INTERVAL_MS  # 50
    ball_lost_hold_ms: int = constants.BALL_LOST_HOLD_MS  # 1000
    player_lost_hold_ms: int = constants.PLAYER_LOST_HOLD_MS  # 1500
    max_move_px_per_second: float = constants.MAX_MOVE_PX_PER_SECOND  # 1200
    ball_weight: float = constants.BALL_WEIGHT  # 0.7 (player_group = 1 - ball)
    # 예측 위치서 이 거리(px)보다 먼 공 후보는 선정서 제외 (오탐 배제). 0 = 비활성
    match_max_jump_px: float = 300.0

    # --- 고급: 안정화 ---
    outlier_deviation_px: float = 300.0  # stabilization._OUTLIER_DEVIATION_PX
    outlier_window: int = 5  # stabilization._OUTLIER_WINDOW
    low_confidence: float = 0.3  # stabilization._LOW_CONFIDENCE
    smooth_window: int = 5  # stabilization._SMOOTH_WINDOW

    # --- 고급: 선정/추적 ---
    confidence_decay: float = 0.9  # target._CONFIDENCE_DECAY
    reacquire_blend: float = 0.5  # tracking._REACQUIRE_BLEND
    w_continuity: float = 0.5  # tracking._W_CONTINUITY (select_ball)
    w_confidence: float = 0.3  # tracking._W_CONFIDENCE
    w_player_proximity: float = 0.2  # tracking._W_PLAYER_PROXIMITY
    w_carrier_proximity: float = 0.6  # target._W_CARRIER_PROXIMITY (DET-05)
    w_carrier_velocity: float = 0.4  # target._W_CARRIER_VELOCITY
    velocity_scale: float = 0.5  # target._VELOCITY_SCALE (px/ms)

    @property
    def dead_zone_half(self) -> float:
        """데드존 반폭 — 공이 크롭 중심 ±이 값 안이면 고정."""
        return self.dead_zone_width / 2

    @property
    def player_group_weight(self) -> float:
        """공+선수 가중 중심에서 선수 군집 가중치 (1 - ball_weight)."""
        return 1.0 - self.ball_weight

    def max_step_px(self, delta_ms: float) -> float:
        """delta_ms 구간의 최대 이동량 (속도 제한)."""
        return self.max_move_px_per_second * delta_ms / 1000


_FIELD_NAMES = {f.name for f in fields(TrackcropConfig)}


def resolve_config(overrides: dict | None) -> TrackcropConfig:
    """overrides(dict) 중 알려진 필드·비어있지 않은 값만 반영해 Config를 만든다.

    알 수 없는 키·None 값은 무시한다 (프론트가 빈 노브를 안 보내거나 None으로 보냄).
    """
    if not overrides:
        return TrackcropConfig()
    clean = {k: v for k, v in overrides.items() if k in _FIELD_NAMES and v is not None}
    return TrackcropConfig(**clean)
