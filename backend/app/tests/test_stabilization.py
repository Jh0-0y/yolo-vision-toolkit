"""Unit tests for app.ml.trackcrop.stabilization.apply_dead_zone (크롭 데드존).

순수 계산 — 모델·영상 없음. 공이 데드존 안이면 크롭 고정, 벗어나면 경계로만 추종하는지 검증.
"""

from app.ml.trackcrop.constants import DEAD_ZONE_WIDTH
from app.ml.trackcrop.stabilization import apply_dead_zone
from app.ml.trackcrop.types import TargetSample

_HALF = DEAD_ZONE_WIDTH / 2  # 104


def _s(ms: int, x: float) -> TargetSample:
    return TargetSample(video_offset_ms=ms, target_center_x=x, target_type="ball", confidence=0.9)


def test_empty_input():
    assert apply_dead_zone([]) == []


def test_hold_when_ball_stays_in_dead_zone():
    # 첫 크롭 중심 = 800. 이후 흔들려도 모두 ±104 안 → 크롭 고정
    xs = [800, 810, 780, 820, 750, 890]  # 890도 800+104=904 이내
    out = apply_dead_zone([_s(i * 100, x) for i, x in enumerate(xs)])
    assert all(o.target_center_x == 800 for o in out)


def test_follow_when_ball_exits_dead_zone():
    # 공이 오른쪽으로 크게 이동 → 크롭이 공을 데드존 경계(104)에 걸치게 이동
    out = apply_dead_zone([_s(0, 800), _s(100, 950)])
    assert out[0].target_center_x == 800  # 첫 프레임 유지
    assert out[1].target_center_x == 950 - _HALF  # 846 — 공이 오른쪽 경계에 걸침
    # 공(950)과 크롭 중심(846) 차이가 정확히 데드존 반폭
    assert out[1].target_center_x == 950 - _HALF


def test_follow_left_direction():
    # 왼쪽 이동도 대칭으로 동작
    out = apply_dead_zone([_s(0, 800), _s(100, 500)])
    assert out[1].target_center_x == 500 + _HALF  # 604 — 공이 왼쪽 경계에 걸침


def test_preserves_offsets_and_metadata():
    samples = [_s(0, 800), _s(100, 950)]
    out = apply_dead_zone(samples)
    assert [o.video_offset_ms for o in out] == [0, 100]
    assert all(o.target_type == "ball" for o in out)
