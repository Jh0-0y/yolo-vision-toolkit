"""크롭 창 사각형 그리기 — 잘리는 자리를 원본 위에 표시한다.

실제로 자르지는 않는다(그건 `cut`). 중심 X 와 창 크기는 인자로 받는다 — 궤적을 모른다.
"""

from __future__ import annotations

from lib.crop.geometry import left_edge

_COLOR = (0, 255, 255)  # BGR 노랑 — 다른 오버레이와 안 겹치게 크롭 창 전용


def draw(
    frame,
    cx: float,
    crop_w: int,
    frame_width: int,
    frame_height: int,
    crop_h: int | None = None,
    crop_y: int = 0,
) -> None:
    """`frame` 에 크롭 사각형 + 라벨을 그린다 (제자리 수정).

    높이를 주면 그만큼만 두른다 — 세로도 잘리는 설정이면 그 자리가 눈에 보여야 한다.
    """
    import cv2

    left = left_edge(cx, crop_w, frame_width)
    right = left + crop_w
    top = max(0, crop_y)
    bottom = min(frame_height, top + crop_h) if crop_h else frame_height
    cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), _COLOR, 3)
    cv2.putText(
        frame, "CROP", (left + 6, top + 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR, 2, cv2.LINE_AA,
    )
