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
