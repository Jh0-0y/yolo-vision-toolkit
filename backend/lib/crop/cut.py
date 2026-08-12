"""프레임을 세로 크롭 창으로 잘라내기 — 오버레이가 아니라 실제 산출물."""

from __future__ import annotations

from lib.crop.geometry import left_edge


def window(frame, cx: float | None, crop_w: int, frame_width: int):
    """타깃을 중심으로 한 crop_w 폭 세로 조각을 돌려준다.

    `cx` 가 None 이면(궤적을 모르면) 화면 중앙으로 자른다 — 잘린 클립은 나와야 한다.
    cv2.VideoWriter 가 바로 먹을 수 있도록 연속 메모리 복사본으로 만든다.
    """
    import numpy as np

    if cx is None:
        cx = frame_width / 2
    left = left_edge(cx, crop_w, frame_width)
    return np.ascontiguousarray(frame[:, left : left + crop_w])
