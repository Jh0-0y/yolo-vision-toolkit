"""lib/crop 계약 테스트 — 좌표 조회 · 그리기 토글 · 잘라내기.

순수 numpy/cv2 — 모델도 영상 디코드도 없다. 크롭 **알고리즘** 은 adaptive-crop
저장소가 테스트한다. 여기서 지키는 것은 짝수 폭 clamp, 100ms 격자 보간('center'
폴백 remap 포함), 그리고 잘라낸 조각이 정확히 크롭 폭의 연속 메모리라는 것.

`geometry` 만 시각(ms)을 알고 그리기·잘라내기는 좌표를 인자로 받는다 — 테스트도
그 경계를 그대로 따라간다.
"""

from dataclasses import dataclass

import numpy as np

from lib import crop
from lib.crop import cut, geometry, highlight, hud, window


@dataclass
class _Sample:
    video_offset_ms: int
    target_center_x: float
    target_type: str


def _blank():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ---------- geometry ----------


def test_crop_width_is_even_and_clamped():
    assert geometry.crop_width_for(1080, 1920) == 608  # round(607.5) -> 608, 짝수
    # 프레임보다 넓지 않고, 짝수를 유지하려 내림한다
    assert geometry.crop_width_for(1080, 500) == 500
    assert geometry.crop_width_for(1080, 501) == 500
    assert geometry.crop_width_for(1080, 1920) % 2 == 0


def test_center_at_interpolates_and_remaps_center_fallback():
    samples = [
        _Sample(0, 100.0, "ball"),
        _Sample(100, 200.0, "ball"),
        _Sample(200, 960.0, "center"),  # 폴백 — 프레임 중앙으로 remap
    ]
    traj = geometry.build_trajectory(samples, frame_width=1000)

    assert geometry.center_at(0, traj) == 100.0
    assert geometry.center_at(50, traj) == 150.0  # 100 과 200 의 중간
    # 'center' 샘플은 원본 960 이 아니라 프레임 중앙(1000/2)을 저장한다
    assert geometry.center_at(200, traj) == 500.0
    # 양 끝 바깥은 clamp
    assert geometry.center_at(-10, traj) == 100.0
    assert geometry.center_at(9999, traj) == 500.0


def test_center_at_empty_trajectory_is_none():
    assert geometry.center_at(0, ([], [])) is None


def test_left_edge_clamps_inside_the_frame():
    assert geometry.left_edge(960, 608, 1920) == 656
    assert geometry.left_edge(0, 608, 1920) == 0  # 왼쪽 밖 → 0
    assert geometry.left_edge(5000, 608, 1920) == 1920 - 608  # 오른쪽 밖 → 끝에 붙임


def test_build_types_and_type_at_step_lookup():
    samples = [_Sample(0, 100.0, "ball"), _Sample(100, 200.0, "player_group")]
    types = geometry.build_types(samples)

    assert types == ([0, 100], ["ball", "player_group"])
    assert geometry.type_at(0, types) == "ball"
    assert geometry.type_at(50, types) == "ball"  # 계단 — 직전 값 유지
    assert geometry.type_at(100, types) == "player_group"
    assert geometry.type_at(9999, types) == "player_group"
    assert geometry.type_at(0, ([], [])) is None


def test_debug_lookup_is_a_step_lookup():
    debug = [
        {"video_offset_ms": 0, "ball_bbox": [900, 480, 20, 20]},
        {"video_offset_ms": 100, "ball_bbox": None},
    ]
    lookup = geometry.build_debug_lookup(debug)

    assert lookup[0] == [0, 100]
    assert geometry.debug_at(50, lookup)["ball_bbox"] == [900, 480, 20, 20]
    assert geometry.debug_at(100, lookup)["ball_bbox"] is None
    assert geometry.debug_at(0, ([], [])) is None


# ---------- cut ----------


def test_cut_returns_contiguous_slice_of_crop_width():
    crop_w = geometry.crop_width_for(1080, 1920)  # 608
    traj = geometry.build_trajectory([_Sample(0, 960.0, "ball")], 1920)

    out = cut.window(_blank(), geometry.center_at(0, traj), crop_w, 1920)

    assert out.shape == (1080, crop_w, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_cut_clamps_at_edges():
    traj = geometry.build_trajectory([_Sample(0, 5000.0, "ball")], 1920)

    out = cut.window(_blank(), geometry.center_at(0, traj), 608, 1920)

    assert out.shape == (1080, 608, 3)


def test_cut_without_a_centre_falls_back_to_the_frame_middle():
    """궤적을 모를 때도 잘린 클립은 나와야 한다 — 중앙으로 자른다."""
    out = cut.window(_blank(), None, 608, 1920)

    assert out.shape == (1080, 608, 3)


# ---------- 그리기 ----------


def test_window_draw_marks_the_crop_edges():
    frame = _blank()

    window.draw(frame, cx=960, crop_w=608, frame_width=1920, frame_height=1080)

    left = geometry.left_edge(960, 608, 1920)
    assert frame[:, left].any()  # 좌변
    assert frame[:, left + 608 - 1].any()  # 우변


def test_hud_draw_smoke_with_and_without_dead_zone():
    frame = _blank()
    hud.draw(frame, 900, 1920, 1080, target_type="ball", dead_zone_half=104)
    assert frame.any()  # 중심선·밴드·라벨 중 무언가 그려짐

    # dead_zone_half=None(데드존 없는 튜닝)이어도 에러 없이 동작
    frame2 = _blank()
    hud.draw(frame2, 900, 1920, 1080, target_type="ball", dead_zone_half=None)
    assert frame2.any()


def _hud_frame(**flags):
    frame = _blank()
    hud.draw(frame, 900, 1920, 1080, target_type="ball", dead_zone_half=104, **flags)
    return frame


def test_hud_draw_toggles():
    # 데드존 밴드는 중심(900) ±104 = 796/1004 세로선
    only_center = _hud_frame(show_dead_zone=False, show_center_line=True)
    assert not only_center[:, 796].any()  # 밴드 열 비어 있음
    assert only_center[:, 900].any()  # 중심선은 있음

    only_band = _hud_frame(show_dead_zone=True, show_center_line=False)
    assert not only_band[:, 900].any()  # 중심선 없음
    assert only_band[:, 796].any()  # 밴드는 있음

    assert not _hud_frame(show_dead_zone=False, show_center_line=False).any()


def test_highlight_draw_marks_selected_boxes():
    entry = {
        "video_offset_ms": 0,
        "ball_bbox": [900, 480, 20, 20],
        "carrier_bbox": [700, 400, 60, 180],
    }
    frame = _blank()

    highlight.draw(frame, entry, 1920, 1080)

    assert frame.any()  # 마커가 그려짐


def test_highlight_draw_is_safe_without_a_selection():
    """공을 못 고른 구간은 정상이다 — 에러 없이 아무것도 그리지 않는다."""
    frame = _blank()

    highlight.draw(frame, {"ball_bbox": None, "carrier_bbox": None}, 1920, 1080)
    assert not frame.any()

    highlight.draw(frame, None, 1920, 1080)
    assert not frame.any()


# ---------- 패키지 경계 ----------


def test_package_reexports_the_submodules():
    """호출자는 `crop.geometry.center_at(...)` 처럼 한 번의 import 로 쓴다."""
    assert crop.geometry is geometry
    assert crop.window is window
    assert crop.hud is hud
    assert crop.highlight is highlight
    assert crop.cut is cut
    assert crop.Trajectory is geometry.Trajectory
