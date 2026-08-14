"""프레임을 크롭 창으로 잘라내기 — 오버레이가 아니라 실제 산출물."""

from __future__ import annotations

from lib.crop.geometry import left_edge


def window(
    frame,
    cx: float | None,
    crop_w: int,
    frame_width: int,
    crop_h: int | None = None,
    crop_y: int = 0,
):
    """타깃을 중심으로 한 크롭 조각을 돌려준다.

    가로는 `cx` 를 따라 움직이고, **세로는 고정**이다(`crop_y` 부터 `crop_h` 만큼).
    크롭 높이가 원본과 같으면 `crop_y=0` 이라 가로만 잘리고, 낮게 잡으면 그때 위아래도
    실제로 잘린다.

    `cx` 가 None 이면(궤적을 모르면) 화면 중앙으로 자른다 — 잘린 클립은 나와야 한다.
    cv2.VideoWriter 가 바로 먹을 수 있도록 연속 메모리 복사본으로 만든다.
    """
    import numpy as np

    if cx is None:
        cx = frame_width / 2
    left = left_edge(cx, crop_w, frame_width)
    top = max(0, crop_y)
    bottom = top + crop_h if crop_h else frame.shape[0]
    return np.ascontiguousarray(frame[top:bottom, left : left + crop_w])
