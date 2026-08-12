"""크롭 창 사각형 그리기 — 잘리는 자리를 원본 위에 표시한다.

실제로 자르지는 않는다(그건 `cut`). 중심 X 는 인자로 받는다 — 궤적을 모른다.
"""

from __future__ import annotations

from lib.crop.geometry import left_edge

_COLOR = (0, 255, 255)  # BGR 노랑 — 다른 오버레이와 안 겹치게 크롭 창 전용


def draw(frame, cx: float, crop_w: int, frame_width: int, frame_height: int) -> None:
    """`frame` 에 전체 높이 크롭 사각형 + 라벨을 그린다 (제자리 수정)."""
    import cv2

    left = left_edge(cx, crop_w, frame_width)
    right = left + crop_w
    cv2.rectangle(frame, (left, 0), (right - 1, frame_height - 1), _COLOR, 3)
    cv2.putText(
        frame, "CROP", (left + 6, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR, 2, cv2.LINE_AA,
    )
