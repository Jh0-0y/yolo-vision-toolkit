"""튜닝용 HUD — 크롭이 조준하는 중심선 · 데드존 밴드 · 타깃 타입 라벨.

"왜 창이 저기로 갔나"를 눈으로 보려고 그린다. 산출물에는 필요 없고 Draw 탭에서만
켠다. 중심 X 와 타입은 인자로 받는다 — 궤적을 모른다.
"""

from __future__ import annotations

_TARGET_COLOR = (0, 165, 255)  # 주황 — 타깃 중심선
_DEADZONE_COLOR = (200, 200, 200)  # 회색 — 데드존 경계
_TYPE_COLORS = {
    "ball": (60, 60, 255),  # 빨강
    "ball_player": (60, 200, 60),  # 초록
    "player_group": (255, 190, 60),  # 파랑
    "center": (160, 160, 160),  # 회색
}


def draw(
    frame,
    cx: float,
    frame_width: int,
    frame_height: int,
    *,
    target_type: str | None = None,
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

    # 타깃 중심선 — 크롭이 조준하는 X
    x = int(round(cx))
    if 0 <= x < frame_width:
        cv2.line(frame, (x, 0), (x, frame_height - 1), _TARGET_COLOR, 2, cv2.LINE_AA)

    # 타입 라벨 (좌하단 고정) — 가독성 위해 검은 외곽선 + 색 채움
    if target_type:
        text = f"TARGET: {target_type}"
        org = (10, frame_height - 16)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, org, font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, org, font, 0.7,
            _TYPE_COLORS.get(target_type, (220, 220, 220)), 2, cv2.LINE_AA,
        )
