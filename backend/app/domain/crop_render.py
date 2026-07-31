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

# 디버그 HUD 색 (BGR)
_TARGET_COLOR = (0, 165, 255)  # 주황 — 타깃 중심선
_DEADZONE_COLOR = (200, 200, 200)  # 회색 — 데드존 경계
_TYPE_COLORS = {
    "ball": (60, 60, 255),  # 빨강
    "ball_player": (60, 200, 60),  # 초록
    "player_group": (255, 190, 60),  # 파랑
    "center": (160, 160, 160),  # 회색
}

# 대상 하이라이트 마커 색 (BGR)
_BALL_HL_COLOR = (60, 60, 255)  # 빨강 — 선택된 공
_CARRIER_HL_COLOR = (60, 220, 60)  # 초록 — 소유선수


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


def build_types(samples: list) -> tuple[list[int], list[str]]:
    """trackcrop samples -> (ms_list, target_type_list) — 타입 라벨용 step 조회 소스."""
    return [s.video_offset_ms for s in samples], [s.target_type for s in samples]


def type_at(ms: float, types: tuple[list[int], list[str]]) -> str | None:
    """시각 `ms`의 target_type (step 함수 — 직전 sample 값)."""
    ms_list, type_list = types
    if not ms_list:
        return None
    i = bisect.bisect_right(ms_list, ms) - 1
    return type_list[max(0, i)]


def draw_target_overlay(
    frame,
    ms: float,
    traj: Trajectory,
    types: tuple[list[int], list[str]],
    dead_zone_half: float | None,
    frame_width: int,
    frame_height: int,
    *,
    show_dead_zone: bool = True,
    show_center_line: bool = True,
) -> None:
    """디버그 HUD — 타깃 중심선 + 데드존 밴드 + 타깃 타입 라벨을 frame에 그린다.

    show_dead_zone=False면 밴드를, show_center_line=False면 중심선·타입 라벨을 생략한다.
    dead_zone_half가 None이면(데드존 없는 버전) 밴드는 어차피 그리지 않는다.
    """
    import cv2

    cx = center_at(ms, traj)
    if cx is None:
        return
    x = int(round(cx))

    # 데드존 밴드 — 크롭 중심 ±half 세로선 (공이 이 안이면 크롭 고정)
    if show_dead_zone and dead_zone_half:
        for bx in (int(round(cx - dead_zone_half)), int(round(cx + dead_zone_half))):
            if 0 <= bx < frame_width:
                cv2.line(frame, (bx, 0), (bx, frame_height - 1), _DEADZONE_COLOR, 1, cv2.LINE_AA)

    if not show_center_line:
        return

    # 타깃 중심선 — 크롭이 조준하는 X
    if 0 <= x < frame_width:
        cv2.line(frame, (x, 0), (x, frame_height - 1), _TARGET_COLOR, 2, cv2.LINE_AA)

    # 타깃 타입 라벨 (좌하단 고정) — 가독성 위해 검은 외곽선 + 색 채움
    t = type_at(ms, types)
    if t:
        text = f"TARGET: {t}"
        org = (10, frame_height - 16)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, org, font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, 0.7, _TYPE_COLORS.get(t, (220, 220, 220)), 2, cv2.LINE_AA)


def build_debug_lookup(debug: list) -> tuple[list[int], list[dict]]:
    """crop.json debug -> (ms_list, entry_list) — 프레임 시각으로 step 조회."""
    return [d["video_offset_ms"] for d in debug], list(debug)


def _debug_at(ms: float, lookup: tuple[list[int], list[dict]]) -> dict | None:
    ms_list, entries = lookup
    if not ms_list:
        return None
    i = bisect.bisect_right(ms_list, ms) - 1
    return entries[max(0, i)]


def _draw_marker(frame, bbox, color, label, frame_width, frame_height) -> None:
    """대상 정수리 위에 아래쪽 삼각형(▼) 마커 + 라벨 (박스 대신)."""
    import cv2
    import numpy as np

    if not bbox:
        return
    x, y, w, h = bbox
    cx = max(0, min(int(round(x + w / 2)), frame_width - 1))
    top = max(0, min(int(round(y)), frame_height - 1))

    size, gap = 14, 6
    apex_y = max(0, top - gap)  # ▼ 아래꼭짓점 — 대상 바로 위를 가리킴
    base_y = max(0, apex_y - size)  # 삼각형 윗변
    pts = np.array(
        [[cx, apex_y], [cx - size, base_y], [cx + size, base_y]], dtype=np.int32
    )
    cv2.fillPoly(frame, [pts], color)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)  # 외곽선 (가독성)

    ly = max(12, base_y - 4)
    org = (max(0, cx - 24), ly)
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_selection_overlay(
    frame,
    ms: float,
    lookup: tuple[list[int], list[dict]],
    frame_width: int,
    frame_height: int,
) -> None:
    """디버그 — trackcrop이 실제로 고른 공(빨강)·소유선수(초록) bbox를 강조한다."""
    entry = _debug_at(ms, lookup)
    if entry is None:
        return
    _draw_marker(frame, entry.get("ball_bbox"), _BALL_HL_COLOR, "BALL", frame_width, frame_height)
    _draw_marker(
        frame, entry.get("carrier_bbox"), _CARRIER_HL_COLOR, "CARRIER", frame_width, frame_height
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
