"""튜닝용 HUD — 크롭이 조준하는 중심선 · 데드존 밴드 · 타깃 타입 라벨.

"왜 창이 저기로 갔나"를 눈으로 보려고 그린다. 중심선은 **타깃 타입 색**으로 그리고
신뢰도를 농도로 얹는다 — 색이 바뀌는 구간이 곧 공/선수 비중이 뒤집힌 구간이다.
색은 `lib.crop.palette` 한 곳에서만 온다(프론트 캔버스와 같은 그림이어야 한다).

중심 X · 타입 · 신뢰도는 인자로 받는다 — 궤적을 모른다.
"""

from __future__ import annotations

from lib.crop import palette

_DEADZONE_COLOR = palette.hex_to_bgr(palette.DEAD_ZONE_COLOR)


def draw(
    frame,
    cx: float,
    frame_width: int,
    frame_height: int,
    *,
    target_type: str | None = None,
    confidence: float | None = None,
    dead_zone_half: float | None = None,
    show_dead_zone: bool = True,
    show_center_line: bool = True,
) -> None:
    """중심선 + 데드존 밴드 + 타입 라벨을 `frame` 에 그린다 (제자리 수정).

    `dead_zone_half` 가 None 이면(데드존 없는 튜닝) 밴드는 그리지 않는다.
    """
    import cv2

    # 데드존 밴드 — 크롭 중심 ±half 세로선 (공이 이 안이면 창이 안 움직인다)
    if show_dead_zone and dead_zone_half:
        for bx in (int(round(cx - dead_zone_half)), int(round(cx + dead_zone_half))):
            if 0 <= bx < frame_width:
                cv2.line(frame, (bx, 0), (bx, frame_height - 1), _DEADZONE_COLOR, 1, cv2.LINE_AA)

    if not show_center_line:
        return

    # 타깃 중심선 — 크롭이 조준하는 X. 색 = 타입, 농도 = 신뢰도.
    color = palette.bgr_for(target_type)
    alpha = palette.alpha_for(confidence)
    faded = tuple(int(c * alpha) for c in color)
    x = int(round(cx))
    if 0 <= x < frame_width:
        cv2.line(frame, (x, 0), (x, frame_height - 1), faded, 2, cv2.LINE_AA)

    # 타입 라벨 (좌하단 고정) — 가독성 위해 검은 외곽선 + 색 채움
    if target_type:
        text = f"TARGET: {target_type}"
        if confidence is not None:
            text += f" {confidence:.0%}"
        org = (10, frame_height - 16)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, org, font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, 0.7, color, 2, cv2.LINE_AA)
