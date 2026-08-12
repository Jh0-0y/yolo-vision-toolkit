"""선택 하이라이트 — 플래너가 실제로 고른 공과 소유선수를 짚어 준다.

검출된 모든 박스가 아니라 **좌표 계산에 쓰인 둘**만 표시한다. 박스 대신 정수리
위 ▼ 마커를 쓴다 — 박스는 검출 오버레이와 겹쳐 구분이 안 되기 때문이다.
"""

from __future__ import annotations

_BALL_COLOR = (60, 60, 255)  # 빨강 — 선택된 공
_CARRIER_COLOR = (60, 220, 60)  # 초록 — 소유선수


def _marker(frame, bbox, color, label, frame_width: int, frame_height: int) -> None:
    """대상 정수리 위에 아래쪽 삼각형(▼) 마커 + 라벨."""
    import cv2
    import numpy as np

    if not bbox:
        return
    x, y, w, h = bbox
    cx = max(0, min(int(round(x + w / 2)), frame_width - 1))
    top = max(0, min(int(round(y)), frame_height - 1))

    size, gap = 14, 6
    apex_y = max(0, top - gap)  # ▼ 아래꼭짓점 — 대상 바로 위를 가리킨다
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


def draw(frame, entry: dict | None, frame_width: int, frame_height: int) -> None:
    """debug 항목 하나(공·소유선수 bbox)를 `frame` 에 표시한다.

    `entry` 가 None 이거나 bbox 가 비어 있으면 아무것도 그리지 않는다 — 공을 못
    고른 구간이 정상적으로 존재한다.
    """
    if entry is None:
        return
    _marker(frame, entry.get("ball_bbox"), _BALL_COLOR, "BALL", frame_width, frame_height)
    _marker(
        frame, entry.get("carrier_bbox"), _CARRIER_COLOR, "CARRIER", frame_width, frame_height
    )
