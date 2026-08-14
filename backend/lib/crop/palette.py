"""타깃 타입의 색 — **이 파일이 유일한 정의다.**

크롭 타깃점은 공을 그대로 따라가지 않는다. 공과 선수 군집의 비율로 정해지고, 그
비율이 뒤집히는 순간 `target_type` 이 바뀐다. 그래서 타입을 색으로 구분해 그리면
"왜 창이 저기로 갔나" 가 눈으로 읽힌다.

서버 렌더(`lib.crop.hud`)와 프론트 캔버스 오버레이가 **같은 그림**을 그려야 하므로
색을 양쪽에 따로 적지 않는다 — 프론트는 `GET /system/crop-palette` 로 이 값을
받아 쓴다.

`TargetSample` 은 비율 값을 들고 있지 않다(`video_offset_ms`·`target_center_x`·
`target_type`·`confidence` 넷뿐). 대신 `ball_player` 구간의 혼합비는 그 런의
`ClipPlanConfig.ball_weight` 그대로이므로, 타입 색 + 런 설정으로 읽어 낸다.
"""

from __future__ import annotations

# adaptive_crop.TargetType 값 → 색. 헥사가 정본이고, cv2 쪽에서 BGR 로 바꿔 쓴다.
TARGET_COLORS: dict[str, str] = {
    "ball": "#ff4d4f",  # 빨강 — 공을 직접 따라간다
    "ball_player": "#52c41a",  # 초록 — 공과 소유선수의 혼합
    "player_group": "#4d94ff",  # 파랑 — 공을 놓쳐 선수 군집을 따라간다
    "center": "#a0a0a0",  # 회색 — 아무것도 못 찾아 화면 중앙으로 폴백
}

# 타입과 무관한 고정 색 — 조준선과 데드존 경계.
TARGET_LINE_COLOR = "#ffa500"
DEAD_ZONE_COLOR = "#c8c8c8"

# 신뢰도를 농도로 그릴 때의 하한. 0 까지 떨어뜨리면 낮은 신뢰 구간이 아예 안 보여
# "그 구간이 없다" 로 읽힌다 — 흐리게라도 남긴다.
MIN_CONFIDENCE_ALPHA = 0.35


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """`#rrggbb` → cv2 가 쓰는 (B, G, R)."""
    h = value.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def bgr_for(target_type: str | None) -> tuple[int, int, int]:
    """타입의 BGR. 모르는 타입은 회색 — 그리지 않는 것보다 낫다."""
    return hex_to_bgr(TARGET_COLORS.get(target_type or "", TARGET_COLORS["center"]))


def alpha_for(confidence: float | None) -> float:
    """신뢰도(0..1) → 농도(MIN_CONFIDENCE_ALPHA..1)."""
    if confidence is None:
        return 1.0
    c = max(0.0, min(1.0, float(confidence)))
    return MIN_CONFIDENCE_ALPHA + (1.0 - MIN_CONFIDENCE_ALPHA) * c


def as_dict() -> dict:
    """프론트에 넘길 형태 — 색 정의를 복제하지 않게 이 한 벌만 내보낸다."""
    return {
        "target_colors": dict(TARGET_COLORS),
        "target_line_color": TARGET_LINE_COLOR,
        "dead_zone_color": DEAD_ZONE_COLOR,
        "min_confidence_alpha": MIN_CONFIDENCE_ALPHA,
    }
