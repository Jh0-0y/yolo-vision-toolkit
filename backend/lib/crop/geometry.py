"""크롭 창 좌표 — "이 시각에 어디를 잘라야 하나"를 답한다. 그리지 않는다.

`adaptive_crop` 이 낸 100ms 격자 샘플을 프레임 시각마다 조회할 수 있는 형태로
바꾸고, 창의 폭과 왼쪽 끝을 계산한다. `lib/crop` 의 나머지 모듈(window · hud ·
highlight · cut)은 전부 여기서 나온 값을 **인자로 받는다** — 궤적을 직접 뒤지지
않는다. 덕분에 한 프레임에 보간을 한 번만 한다.

조회는 두 종류다. 중심 X 는 **선형 보간**(창이 부드럽게 움직여야 하니까),
타입·디버그는 **계단**(직전 샘플 값을 유지)이다.
"""

from __future__ import annotations

import bisect

# (오름차순 video_offset_ms 목록, 대응하는 타깃 중심 X 목록)
Trajectory = tuple[list[int], list[float]]


# 크롭 창 기본 크기(px) — 1080p 소스의 9:16 과 같은 값. 연구실 프리셋이 런마다 바꾼다.
DEFAULT_CROP_W = 608
DEFAULT_CROP_H = 1080


def _even(v: int, limit: int) -> int:
    """소스 안에 들어가는 짝수 — libx264/yuv420p 가 짝수 해상도를 요구한다."""
    v = min(max(2, int(v)), limit)
    if v % 2:
        v -= 1  # limit 를 넘지 않도록 내림
    return max(2, v)


def crop_window_for(
    source_width: int,
    source_height: int,
    crop_w: int = DEFAULT_CROP_W,
    crop_h: int = DEFAULT_CROP_H,
) -> tuple[int, int, int]:
    """요청한 크롭 창(px)을 소스 안에 맞춘 `(width, height, y)`.

    **크기는 비율이 아니라 절대 px 다.** 소스가 더 작으면 소스 크기로 clamp 한다 —
    잘라낼 것이 없는 창을 만드느니 들어가는 만큼 잘라 내는 편이 낫다.

    `y` 는 세로 가운데다. 높이가 소스보다 낮을 때만 의미가 있고(그때 위아래가
    실제로 잘린다), 같으면 0 이라 지금까지와 똑같이 가로만 잘린다.
    """
    w = _even(crop_w or DEFAULT_CROP_W, source_width)
    h = _even(crop_h or DEFAULT_CROP_H, source_height)
    y = max(0, (source_height - h) // 2)
    return w, h, y


def build_trajectory(samples: list, frame_width: int) -> Trajectory:
    """crop 샘플 -> (ms 목록, 중심 X 목록).

    'center' 폴백 샘플은 원본 중앙을 뜻하므로 넘겨받은 폭의 절반으로 바꿔 둔다
    (호출자가 나중에 프리뷰 해상도로 스케일한다).
    """
    ms_list: list[int] = []
    cx_list: list[float] = []
    half = frame_width / 2
    for s in samples:
        ms_list.append(s.video_offset_ms)
        cx_list.append(half if s.target_type == "center" else s.target_center_x)
    return ms_list, cx_list


def center_at(ms: float, traj: Trajectory) -> float | None:
    """100ms 격자에서 시각 `ms` 의 타깃 중심 X 를 선형 보간한다. 궤적이 비면 None."""
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


def left_edge(cx: float, crop_w: int, frame_width: int) -> int:
    """cx 를 중심으로 한 crop_w 폭 창의 왼쪽 X. 프레임 안으로 clamp 한다."""
    left = int(round(cx - crop_w / 2))
    return max(0, min(left, frame_width - crop_w))


def build_types(samples: list) -> tuple[list[int], list[str]]:
    """crop 샘플 -> (ms 목록, target_type 목록) — 계단 조회용."""
    return [s.video_offset_ms for s in samples], [s.target_type for s in samples]


def type_at(ms: float, types: tuple[list[int], list[str]]) -> str | None:
    """시각 `ms` 의 target_type (계단 — 직전 샘플 값)."""
    ms_list, type_list = types
    if not ms_list:
        return None
    i = bisect.bisect_right(ms_list, ms) - 1
    return type_list[max(0, i)]


def build_targets(samples: list) -> tuple[list[int], list[tuple[str, float]]]:
    """crop 샘플 -> (ms 목록, (target_type, confidence) 목록) — 계단 조회용.

    HUD 가 색(타입)과 농도(신뢰도)를 함께 쓰므로 둘을 한 번에 조회한다 —
    프레임마다 두 번 bisect 하지 않게.
    """
    return (
        [s.video_offset_ms for s in samples],
        [(s.target_type, float(s.confidence)) for s in samples],
    )


def target_at(
    ms: float, targets: tuple[list[int], list[tuple[str, float]]]
) -> tuple[str, float] | None:
    """시각 `ms` 의 (target_type, confidence) (계단 — 직전 샘플 값)."""
    ms_list, entries = targets
    if not ms_list:
        return None
    i = bisect.bisect_right(ms_list, ms) - 1
    return entries[max(0, i)]


def build_debug_lookup(debug: list) -> tuple[list[int], list[dict]]:
    """crop.json 의 debug -> (ms 목록, 항목 목록) — 계단 조회용."""
    return [d["video_offset_ms"] for d in debug], list(debug)


def debug_at(ms: float, lookup: tuple[list[int], list[dict]]) -> dict | None:
    """시각 `ms` 의 debug 항목 (계단 — 직전 샘플 값)."""
    ms_list, entries = lookup
    if not ms_list:
        return None
    i = bisect.bisect_right(ms_list, ms) - 1
    return entries[max(0, i)]
