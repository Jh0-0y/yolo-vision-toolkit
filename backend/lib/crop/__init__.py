"""세로 9:16 크롭 — 좌표 조회와 그리기·잘라내기.

**좌표를 계산하는 알고리즘은 여기 없다.** 공·선수를 추적해 타깃 중심을 내는 일은
`adaptive_crop` 패키지가 한다. 이 패키지는 그 결과(100ms 격자 샘플)를 받아
프레임마다 조회하고, 화면에 표시하거나 실제로 잘라낸다.

    plan       adaptive_crop 어댑터 — 튜닝 해석 · CropSpec · crop.json
    geometry   시각 -> 중심 X · 창 폭 · 왼쪽 끝        (그리지 않음)
    window     크롭 창 사각형                          (표시용)
    hud        중심선 · 데드존 · 타입 라벨              (튜닝용)
    highlight  플래너가 고른 공 · 소유선수 마커          (튜닝용)
    cut        프레임을 창으로 잘라내기                  (산출물)

`geometry` 만 시각(ms)을 안다. 나머지는 **이미 계산된 좌표를 인자로 받는다** —
그래서 한 프레임에 보간이 한 번만 돈다. 조립은 호출자가 한다:

    cx = crop.geometry.center_at(ms, traj)
    if cx is not None:
        crop.window.draw(frame, cx, crop_w, w, h)
"""

from lib.crop import cut, geometry, highlight, hud, plan, window
from lib.crop.geometry import Trajectory

__all__ = ["Trajectory", "cut", "geometry", "highlight", "hud", "plan", "window"]
