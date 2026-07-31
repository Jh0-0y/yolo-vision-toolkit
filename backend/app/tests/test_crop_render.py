"""Geometry unit tests for app.domain.crop_render (the crop-cut/overlay logic).

Pure numpy/cv2 — no model, no video decode. Covers the even-width clamp, the
100ms-grid interpolation (incl. the 'center' fallback remap), and that cut_window
returns a contiguous slice of exactly the crop width.
"""

from dataclasses import dataclass

import numpy as np

from app.domain import crop_render


@dataclass
class _Sample:
    video_offset_ms: int
    target_center_x: float
    target_type: str


def test_crop_width_is_even_and_clamped():
    assert crop_render.crop_width_for(1080, 1920) == 608  # round(607.5) -> 608, even
    # never wider than the frame, and rounded down to stay even/inside
    assert crop_render.crop_width_for(1080, 500) == 500
    assert crop_render.crop_width_for(1080, 501) == 500  # 501 clamp -> even 500
    assert crop_render.crop_width_for(1080, 1920) % 2 == 0


def test_center_at_interpolates_and_remaps_center_fallback():
    samples = [
        _Sample(0, 100.0, "ball"),
        _Sample(100, 200.0, "ball"),
        _Sample(200, 960.0, "center"),  # fallback — remapped to frame centre
    ]
    traj = crop_render.build_trajectory(samples, frame_width=1000)
    assert crop_render.center_at(0, traj) == 100.0
    assert crop_render.center_at(50, traj) == 150.0  # halfway between 100 and 200
    # 'center' sample stored the frame centre (1000/2), not the source 960
    assert crop_render.center_at(200, traj) == 500.0
    # clamps past the ends
    assert crop_render.center_at(-10, traj) == 100.0
    assert crop_render.center_at(9999, traj) == 500.0


def test_center_at_empty_trajectory_is_none():
    assert crop_render.center_at(0, ([], [])) is None


def test_cut_window_returns_contiguous_slice_of_crop_width():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop_w = crop_render.crop_width_for(1080, 1920)  # 608
    traj = crop_render.build_trajectory([_Sample(0, 960.0, "ball")], 1920)
    out = crop_render.cut_window(frame, 0, traj, crop_w, 1920)
    assert out.shape == (1080, crop_w, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_cut_window_clamps_at_edges():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop_w = 608
    # target far to the right — window must stay inside the frame
    traj = crop_render.build_trajectory([_Sample(0, 5000.0, "ball")], 1920)
    out = crop_render.cut_window(frame, 0, traj, crop_w, 1920)
    assert out.shape == (1080, crop_w, 3)


def test_build_types_and_type_at_step_lookup():
    samples = [_Sample(0, 100.0, "ball"), _Sample(100, 200.0, "player_group")]
    types = crop_render.build_types(samples)
    assert types == ([0, 100], ["ball", "player_group"])
    assert crop_render.type_at(0, types) == "ball"
    assert crop_render.type_at(50, types) == "ball"  # step — 직전 값 유지
    assert crop_render.type_at(100, types) == "player_group"
    assert crop_render.type_at(9999, types) == "player_group"
    assert crop_render.type_at(0, ([], [])) is None


def test_draw_target_overlay_smoke():
    # blank 프레임에 그려도 에러 없이 픽셀이 변경되는지 (HUD가 실제로 그려짐)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    samples = [_Sample(0, 900.0, "ball"), _Sample(100, 900.0, "ball")]
    traj = crop_render.build_trajectory(samples, 1920)
    types = crop_render.build_types(samples)
    crop_render.draw_target_overlay(frame, 50, traj, types, dead_zone_half=104, frame_width=1920, frame_height=1080)
    assert frame.any()  # 무언가 그려짐 (중심선·밴드·라벨)
    # dead_zone_half=None(데드존 없는 버전)이어도 에러 없이 동작
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop_render.draw_target_overlay(frame2, 50, traj, types, dead_zone_half=None, frame_width=1920, frame_height=1080)
    assert frame2.any()


def _overlay_frame(**flags):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    samples = [_Sample(0, 900.0, "ball"), _Sample(100, 900.0, "ball")]
    traj = crop_render.build_trajectory(samples, 1920)
    types = crop_render.build_types(samples)
    crop_render.draw_target_overlay(
        frame, 50, traj, types, dead_zone_half=104, frame_width=1920, frame_height=1080, **flags
    )
    return frame


def test_draw_target_overlay_toggles():
    # 데드존 밴드는 크롭 중심(900) ±104 = 796/1004 세로선. show_dead_zone=False면 그 열이 비어야
    only_center = _overlay_frame(show_dead_zone=False, show_center_line=True)
    assert not only_center[:, 796].any()  # 밴드 열 비어있음
    assert only_center[:, 900].any()  # 중심선은 있음
    # show_center_line=False면 중심선(900 열)이 없어야
    only_band = _overlay_frame(show_dead_zone=True, show_center_line=False)
    assert not only_band[:, 900].any()
    assert only_band[:, 796].any()  # 밴드는 있음
    # 둘 다 False면 아무것도 안 그려짐
    assert not _overlay_frame(show_dead_zone=False, show_center_line=False).any()


def test_draw_selection_overlay_smoke():
    # debug lookup으로 선택 공/소유선수 마커가 그려지는지 (빈 debug·None bbox 안전)
    debug = [
        {"video_offset_ms": 0, "ball_bbox": [900, 480, 20, 20], "carrier_bbox": [700, 400, 60, 180]},
        {"video_offset_ms": 100, "ball_bbox": None, "carrier_bbox": None},
    ]
    lookup = crop_render.build_debug_lookup(debug)
    assert lookup[0] == [0, 100]
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop_render.draw_selection_overlay(frame, 0, lookup, 1920, 1080)
    assert frame.any()  # 마커가 그려짐
    # bbox 둘 다 None인 시각 → 에러 없이 아무것도 안 그림
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop_render.draw_selection_overlay(frame2, 100, lookup, 1920, 1080)
    assert not frame2.any()
    # 빈 lookup도 무해
    crop_render.draw_selection_overlay(frame2, 0, ([], []), 1920, 1080)
