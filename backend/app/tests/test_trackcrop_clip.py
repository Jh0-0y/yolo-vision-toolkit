"""Unit tests for 클립 플래너 (balltrack / pathopt / clip_planner).

순수 계산 — 모델·영상 없음. 합성 시나리오로 요구사항을 검증한다:
  - 같은 id 내 불가능 점프 쪼개기, id 바뀐 조각 스티칭
  - 정지 공(관중석) 배제 — 트랙 채점
  - 미검출 gap 보간(짧음) / 소유선수 fallback(장기 가림 + 흡수 병합)
  - 경로 최적화: 데드존 안 정지, 이동 추종, 최대 속도 하드 캡
"""

from app.ml.trackcrop.balltrack import (
    ClipPlanConfig,
    build_tracklets,
    select_game_ball_track,
    stitch_tracks,
)
from app.ml.trackcrop.clip_planner import plan_clip, resolve_targets_clip
from app.ml.trackcrop.pathopt import optimize_path
from app.ml.trackcrop.types import Detection, TargetSample, TargetType

CFG = ClipPlanConfig()


def _ball(cx: float, ms: int, track_id: int | None = 1, conf: float = 0.6) -> Detection:
    return Detection("ball", track_id, cx - 10, 490, 20, 20, conf, ms)


def _player(cx: float, ms: int, track_id: int, conf: float = 0.8) -> Detection:
    # y 300~700 — 공(y 490~510 중심)과 세로로 겹치는 위치
    return Detection("player", track_id, cx - 40, 300, 80, 400, conf, ms)


def _target(x: float, ms: int, conf: float = 0.8) -> TargetSample:
    return TargetSample(ms, x, TargetType.BALL.value, conf)


# ---------------------------------------------------------------------------
# 패스 1: tracklet / 스티칭 / 채점


def test_split_impossible_jump_within_same_id():
    # 같은 id인데 100ms에 1600px 점프 — id가 오탐을 문 것 → 그 지점에서 쪼갬
    samples = [
        (0, [_ball(100, 0)]),
        (100, [_ball(200, 100)]),
        (200, [_ball(1800, 200)]),
    ]
    tracklets = build_tracklets(samples, CFG)
    assert len(tracklets) == 2
    assert [len(t.points) for t in tracklets] == [2, 1]


def test_stitch_reconnects_id_change_across_gap():
    # 같은 궤적인데 400ms 끊긴 뒤 새 id로 재등장 → 한 체인으로 스티칭
    samples = [(ms, [_ball(500 + 0.2 * ms, ms, track_id=1)]) for ms in range(0, 600, 100)]
    samples += [(ms, [_ball(500 + 0.2 * ms, ms, track_id=2)]) for ms in range(900, 1400, 100)]
    chains = stitch_tracks(build_tracklets(samples, CFG), CFG)
    assert len(chains) == 1
    assert len(chains[0].points) == 11


def test_simultaneous_balls_never_stitched():
    # 동시에 보이는 두 공은 절대 같은 체인이 될 수 없다
    samples = [
        (ms, [_ball(500, ms, track_id=1), _ball(1500, ms, track_id=2)])
        for ms in range(0, 500, 100)
    ]
    chains = stitch_tracks(build_tracklets(samples, CFG), CFG)
    assert len(chains) == 2


def test_static_spectator_ball_rejected():
    # 경기 공: 이동 + 선수 상호작용. 관중석 공: 고신뢰지만 정지·상호작용 없음
    samples = []
    for ms in range(0, 20000, 100):
        game_x = 400 + ms * 0.05  # 20초 동안 400→1400
        samples.append(
            (
                ms,
                [
                    _ball(game_x, ms, track_id=1, conf=0.5),
                    _ball(1800, ms, track_id=99, conf=0.9),  # 관중석 공
                    _player(game_x, ms, track_id=10),
                ],
            )
        )
    track, info = select_game_ball_track(samples, CFG)
    assert track is not None
    assert track.points[0].det.track_id == 1
    assert info["tracks"] == 2


def test_static_only_clip_has_no_ball():
    # 정지 공(관중석)뿐이면 "공 없음" 클립
    samples = [(ms, [_ball(1800, ms, track_id=99, conf=0.9)]) for ms in range(0, 20000, 100)]
    track, _ = select_game_ball_track(samples, CFG)
    assert track is None


# ---------------------------------------------------------------------------
# 패스 2 전반: 타깃 결정 (보간 / 소유선수 fallback)


def test_short_gap_is_interpolated_as_ball():
    # 1000~1600ms 미검출 — 재등장 위치를 아니까 선형 보간, BALL 유지
    samples = []
    for ms in range(0, 3700, 100):
        if 1000 < ms < 1600:
            samples.append((ms, []))
        else:
            samples.append((ms, [_ball(100 + 0.5 * ms, ms)]))  # 0.5px/ms 등속 이동
    targets, _ = resolve_targets_clip(samples, CFG)
    mid = next(t for t in targets if t.video_offset_ms == 1300)
    assert mid.target_type == TargetType.BALL.value
    assert abs(mid.target_center_x - 750) < 1  # (600+900)/2


def test_long_gap_falls_back_to_carrier():
    # 3초 가림(보간 한계 초과) — use_carrier=True면 마지막으로 공을 든 선수를 따라간다.
    # 흡수 병합(스티칭 한계 2.5s 초과라 별도 체인)도 함께 검증된다.
    cfg = ClipPlanConfig(use_carrier=True)
    samples = []
    for ms in range(0, 1100, 100):
        x = 500 + 0.1 * ms
        samples.append((ms, [_ball(x, ms, track_id=1), _player(x, ms, track_id=5), _player(1500, ms, track_id=6)]))
    for ms in range(1100, 4000, 100):
        samples.append((ms, [_player(620, ms, track_id=5), _player(1500, ms, track_id=6)]))
    for ms in range(4000, 5100, 100):
        samples.append((ms, [_ball(700, ms, track_id=2), _player(700, ms, track_id=5), _player(1500, ms, track_id=6)]))

    targets, info = resolve_targets_clip(samples, cfg)
    assert info["absorbed"] == 1
    mid = next(t for t in targets if t.video_offset_ms == 2500)
    # carrier ON: 군집 중앙값(≈1060)이 아니라 소유선수(620)를 따라가야 한다
    assert mid.target_type == TargetType.PLAYER_GROUP.value
    assert abs(mid.target_center_x - 620) < 1

    # 기본(use_carrier=False): 소유선수 대신 선수 군집 중앙값을 따라간다
    targets_off, _ = resolve_targets_clip(samples, ClipPlanConfig())
    mid_off = next(t for t in targets_off if t.video_offset_ms == 2500)
    assert abs(mid_off.target_center_x - 620) > 100  # 620(carrier) 아님


def test_no_ball_clip_uses_player_group():
    samples = [(ms, [_player(900, ms, track_id=5)]) for ms in range(0, 2000, 100)]
    planned, info = plan_clip(samples, CFG)
    assert len(planned) == len(samples)
    assert all(t.target_type == TargetType.PLAYER_GROUP.value for t in planned)


# ---------------------------------------------------------------------------
# 패스 2 후반: 경로 최적화


def test_path_holds_still_inside_dead_zone():
    # 데드존(±104) 안의 노이즈에는 크롭이 거의 반응하지 않아야 한다
    noise = [17, -23, 9, -14, 28, -5, 12, -19, 3, 22]
    targets = [_target(900 + noise[i % 10], i * 100) for i in range(100)]
    path = optimize_path(targets, CFG)
    total_variation = sum(abs(b - a) for a, b in zip(path, path[1:]))
    assert total_variation < 30  # 원시 타깃의 총 이동량은 ~1700px


def test_path_follows_moving_ball_within_dead_zone_margin():
    # 등속 이동(400px/s)하는 공은 데드존 근처 이내로 계속 추종해야 한다
    targets = [_target(400 + 8 * i, i * 100) for i in range(120)]
    path = optimize_path(targets, CFG)
    dead_zone_half = 104
    for i in range(10, 110):
        assert abs(path[i] - targets[i].target_center_x) <= dead_zone_half + 30


def test_path_respects_speed_cap_and_bounds():
    # 순간이동 타깃 — 하드 캡(1200px/s → 120px/100ms)과 크롭 범위를 지켜야 한다
    targets = [_target(350 if i < 50 else 1600, i * 100) for i in range(100)]
    path = optimize_path(targets, CFG)
    assert all(304 <= x <= 1616 for x in path)
    max_step = max(abs(b - a) for a, b in zip(path, path[1:]))
    assert max_step <= 120 + 1e-6


# ---------------------------------------------------------------------------
# 통합


def test_plan_clip_end_to_end():
    samples = []
    for ms in range(0, 20000, 100):
        dets = [_ball(1800, ms, track_id=99, conf=0.9)]  # 관중석 공 상시
        if not 5000 < ms < 6200:  # 1.2초 미검출 구간
            game_x = 400 + ms * 0.05
            dets.append(_ball(game_x, ms, track_id=1, conf=0.5))
            dets.append(_player(game_x, ms, track_id=10))
        samples.append((ms, dets))

    debug: list = []
    planned, info = plan_clip(samples, CFG, debug=debug)
    assert len(planned) == len(samples) == len(debug)
    assert all(304 <= t.target_center_x <= 1616 for t in planned)
    max_step = max(
        abs(b.target_center_x - a.target_center_x)
        for a, b in zip(planned, planned[1:])
    )
    assert max_step <= 120 + 1e-6
    # 관중석 공(1800)이 아니라 경기 공 궤적을 따라야 한다
    last = planned[-1]
    assert abs(last.target_center_x - (400 + 19900 * 0.05)) < 250
    # 미검출 구간도 보간으로 BALL 유지 (점프 없음)
    gap_sample = next(t for t in planned if t.video_offset_ms == 5600)
    assert gap_sample.target_type in (TargetType.BALL.value, TargetType.BALL_PLAYER.value)


# ---------------------------------------------------------------------------
# 워커 프로세스 경계


def test_trackcrop_error_survives_pickle():
    """워커가 던진 예외는 부모로 pickle되어 돌아온다 — 언피클 실패는
    ProcessPool 전체를 BrokenProcessPool로 만들었다 (실측 회귀)."""
    import pickle

    from app.ml.trackcrop.errors import ErrorCode, TrackCropError

    e = TrackCropError(ErrorCode.MODEL_LOAD_FAILED, "모델에 ball 클래스가 없습니다.", details={"names": {0: "player"}})
    e2 = pickle.loads(pickle.dumps(e))
    assert (e2.code, e2.message, e2.details) == (e.code, e.message, e.details)


def test_gap_interpolation_curves_with_motion():
    """미검출 구간 보간이 직선(chord)이 아니라 이웃 진행 방향을 따라 휘는지 —
    포물선 궤적(슛)의 정점 프레임을 놓쳤을 때 직선보다 정점에 가깝게 복원."""
    from app.ml.trackcrop.balltrack import BallTrack, ClipPlanConfig, TrackPoint
    from app.ml.trackcrop.clip_planner import _ball_at
    from app.ml.trackcrop.types import Detection

    def det(ms):  # x 등속, y 포물선(정점 t=200에서 y=500), 정점 프레임 미검출
        cy = 500 - 0.003 * (ms - 200) ** 2
        return TrackPoint(ms, Detection("ball", None, 0.3 * ms - 10, cy - 10, 20, 20, 0.9, ms))

    track = BallTrack([det(0), det(100), det(300), det(400)])  # 200ms 누락
    got = _ball_at(track, 200, ClipPlanConfig())
    assert got is not None
    _, _, bbox = got
    cy = bbox[1] + bbox[3] / 2
    true_y, chord_y = 500.0, 470.0  # 직선 보간이면 470
    assert abs(cy - true_y) < abs(chord_y - true_y)  # 직선보다 정점에 가까워야 한다
