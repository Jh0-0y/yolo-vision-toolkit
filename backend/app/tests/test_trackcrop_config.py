"""Unit tests for app.ml.trackcrop.config (런타임 튜닝 오버라이드).

순수 계산 — 모델·영상 없음. resolve_config의 병합·파생값과, cfg가 실제
파이프라인(데드존)에 반영되는지 검증.
"""

from app.ml.trackcrop.config import TrackcropConfig, resolve_config
from app.ml.trackcrop.stabilization import apply_dead_zone
from app.ml.trackcrop.tracking import BallTracker, select_ball
from app.ml.trackcrop.types import Detection, TargetSample


def _ball(cx: float, conf: float) -> Detection:
    return Detection("ball", None, cx - 10, 490, 20, 20, conf, 0)


def test_gating_excludes_far_high_conf_ball():
    # 실측 960, 게이트 300. 고신뢰 오탐(300, conf0.95)은 배제, 진짜 공(1000, conf0.5) 선택
    real, fp = _ball(1000, 0.5), _ball(300, 0.95)
    picked = select_ball([fp, real], 960.0, [], None, TrackcropConfig(), gate_radius=300)
    assert picked.center_x == real.center_x


def test_gating_single_far_ball_returns_none():
    # 단일 후보라도 게이트 밖이면 제외 → None (단일-후보 지름길 구멍 방지)
    assert select_ball([_ball(300, 0.95)], 960.0, [], None, TrackcropConfig(), gate_radius=300) is None


def test_gating_disabled_without_radius():
    # gate_radius=None → 게이트 없음, 단일 먼 오탐도 그대로 반환
    fp = _ball(300, 0.95)
    assert select_ball([fp], 960.0, [], None, TrackcropConfig(), gate_radius=None) is fp


def test_gating_skipped_without_reference():
    # expected_x=None(첫 획득) → 게이트 있어도 통과
    fp = _ball(300, 0.95)
    assert select_ball([fp], None, [], None, TrackcropConfig(), gate_radius=300) is fp


def test_gate_radius_grows_with_missed_time():
    # 게이트 반경 = base(300) + max_move(1200)×놓친초. 놓칠수록 넓어짐
    t = BallTracker(TrackcropConfig())
    t.update(_ball(500, 0.9), 0)  # 마지막 실측 500 @ 0ms
    assert t.gate_radius(0) == 300  # 놓침 0 → base
    assert t.gate_radius(500) == 300 + 1200 * 0.5  # 900
    # match_max_jump_px=0 → 비활성
    t2 = BallTracker(resolve_config({"match_max_jump_px": 0}))
    t2.update(_ball(500, 0.9), 0)
    assert t2.gate_radius(100) is None


def test_proximity_only_on_first_acquisition():
    # 추적 중(expected_x 있음): 선수서 먼 공(슛)이 선수 옆 오탐에 안 짐
    shot = _ball(1000, 0.6)  # 선수(200)서 먼 진짜 공, 실측 기준 근처
    near_fp = _ball(210, 0.6)  # 선수 옆 오탐
    picked = select_ball([shot, near_fp], 980.0, [200.0], None, TrackcropConfig(), gate_radius=300)
    assert picked.center_x == shot.center_x  # 슛 공이 선택됨 (proximity 벌점 없음)
    # 첫 획득(expected_x=None): proximity가 살아나 선수 옆 후보 선호
    first = select_ball([shot, near_fp], None, [200.0], None, TrackcropConfig())
    assert first.center_x == near_fp.center_x


def _s(ms: int, x: float) -> TargetSample:
    return TargetSample(video_offset_ms=ms, target_center_x=x, target_type="ball", confidence=0.9)


def test_defaults_match_constants():
    from app.ml.trackcrop import constants

    cfg = TrackcropConfig()
    assert cfg.dead_zone_width == constants.DEAD_ZONE_WIDTH
    assert cfg.sampling_interval_ms == constants.SAMPLING_INTERVAL_MS
    assert cfg.ball_lost_hold_ms == constants.BALL_LOST_HOLD_MS
    assert cfg.max_move_px_per_second == constants.MAX_MOVE_PX_PER_SECOND
    assert cfg.ball_weight == constants.BALL_WEIGHT


def test_derived_properties():
    cfg = TrackcropConfig(dead_zone_width=400, ball_weight=0.6)
    assert cfg.dead_zone_half == 200.0
    assert abs(cfg.player_group_weight - 0.4) < 1e-9
    assert cfg.max_step_px(100) == cfg.max_move_px_per_second * 100 / 1000


def test_resolve_none_is_default():
    assert resolve_config(None) == TrackcropConfig()
    assert resolve_config({}) == TrackcropConfig()


def test_resolve_applies_and_ignores_unknown():
    cfg = resolve_config({"dead_zone_width": 400, "unknown_key": 1, "ball_weight": None})
    assert cfg.dead_zone_width == 400  # 반영
    assert cfg.ball_weight == TrackcropConfig().ball_weight  # None은 무시 → 기본
    assert not hasattr(cfg, "unknown_key")  # 알수없는 키 무시


def test_override_changes_dead_zone_behaviour():
    # 공이 200px 이동: 기본(dead_half 104)은 따라가고, 400(dead_half 200)은 고정
    seq = [_s(0, 800), _s(100, 1000)]
    base = apply_dead_zone(seq, TrackcropConfig())
    wide = apply_dead_zone(seq, resolve_config({"dead_zone_width": 400}))
    assert base[1].target_center_x == 1000 - 104  # 896 — 경계로 추종
    assert wide[1].target_center_x == 800  # 데드존 안 → 고정
