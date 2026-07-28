"""Crop-window rendering from a precomputed trackcrop trajectory.

Pure cv2/CPU. Given the ``TargetSample`` list from ``app.ml.trackcrop`` (the 100ms
target-centre trajectory), this module either

  - **draws** the moving vertical 9:16 crop window onto a frame as an overlay
    (``draw_window``), or
  - **cuts** the frame down to that window — the actual crop (``cut_window``).

No torch, no model, no DB — the trajectory is computed elsewhere; here we only do
geometry + pixel slicing. Shared by ``app.workers.annotate_worker``.
"""

from __future__ import annotations

import bisect

# Trajectory = (ascending video_offset_ms list, matching target-centre-x list).
Trajectory = tuple[list[int], list[float]]

_CROP_COLOR = (0, 255, 255)  # BGR yellow — distinct, always the crop window


def crop_width_for(height: int, frame_width: int) -> int:
    """Even 9:16 crop width for a frame (e.g. 1080 -> 608), clamped to the frame.

    libx264/yuv420p needs even dimensions, so the result is always even. Never
    wider than the source (already-vertical clips just aren't cropped much).
    """
    w = max(2, round(height * 9 / 16))
    w = min(w, frame_width)
    if w % 2:
        w -= 1  # round down to stay within frame_width
    return max(2, w)


def build_trajectory(samples: list, frame_width: int) -> Trajectory:
    """trackcrop samples -> (ms_list, center_x_list) for per-frame interpolation.

    The 'center' fallback stores 1920/2=960 (trackcrop's fixed source width); remap
    it to the actual frame centre so the window is correct at any resolution.
    """
    ms_list: list[int] = []
    cx_list: list[float] = []
    half = frame_width / 2
    for s in samples:
        ms_list.append(s.video_offset_ms)
        cx_list.append(half if s.target_type == "center" else s.target_center_x)
    return ms_list, cx_list


def center_at(ms: float, traj: Trajectory) -> float | None:
    """Linear-interpolate the target centre X at time `ms` from the 100ms grid."""
    ms_list, cx_list = traj
    if not ms_list:
        return None
    if ms <= ms_list[0]:
        return cx_list[0]
    if ms >= ms_list[-1]:
        return cx_list[-1]
    i = bisect.bisect_right(ms_list, ms)  # ms_list[i-1] <= ms < ms_list[i]
    t0, t1 = ms_list[i - 1], ms_list[i]
    x0, x1 = cx_list[i - 1], cx_list[i]
    if t1 == t0:
        return x0
    return x0 + (x1 - x0) * ((ms - t0) / (t1 - t0))


def _left_edge(cx: float, crop_w: int, frame_width: int) -> int:
    """Left X of a crop_w-wide window centred on cx, clamped inside the frame."""
    left = int(round(cx - crop_w / 2))
    return max(0, min(left, frame_width - crop_w))


def draw_window(
    frame, ms: float, traj: Trajectory, crop_w: int, frame_width: int, frame_height: int
) -> None:
    """Overlay mode: draw the full-height crop rectangle + label onto `frame` in place."""
    import cv2

    cx = center_at(ms, traj)
    if cx is None:
        return
    left = _left_edge(cx, crop_w, frame_width)
    right = left + crop_w
    cv2.rectangle(frame, (left, 0), (right - 1, frame_height - 1), _CROP_COLOR, 3)
    cv2.putText(
        frame, "CROP", (left + 6, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CROP_COLOR, 2, cv2.LINE_AA,
    )


def cut_window(frame, ms: float, traj: Trajectory, crop_w: int, frame_width: int):
    """Cut mode: return the crop_w-wide vertical slice centred on the target.

    Falls back to the frame centre when the trajectory is empty. The result is a
    contiguous copy so cv2.VideoWriter can consume it directly.
    """
    import numpy as np

    cx = center_at(ms, traj)
    if cx is None:
        cx = frame_width / 2
    left = _left_edge(cx, crop_w, frame_width)
    return np.ascontiguousarray(frame[:, left : left + crop_w])
