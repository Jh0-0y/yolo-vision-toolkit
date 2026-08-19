"""타일 추론 — 큰 이미지를 격자로 잘라 각각 추론하고 원본 좌표로 되돌린다.

1920px 원본에서 수십 px 인 공은 풀 프레임 추론이 640 으로 줄이는 순간 사라진다.
타일로 자르면 그 공이 타일 안에서는 큼직해서 잡힌다. 대가는 추론 횟수다 —
타일 수만큼 늘어난다.

**연구실의 `adaptive-crop` 을 쓰지 않는다.** 그쪽에도 타일 검출기가 있지만 검출을
`ball`·`player` 라는 **역할**로 좁혀서 돌려주기 때문에, 학습실의 임의 클래스
(`hoop` · `net` · 무엇이든)를 다룰 수 없다. 학습실은 어떤 연구실 패키지도 몰라야 한다
(`docs/agents/architecture.md`).

좌표는 **원본 정규화 좌표**로 나간다 — 풀 프레임 추론과 같은 형태라, 부르는 쪽은
어느 방식으로 얻었는지 알 필요가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.detect.ensemble import Detection, iou
from lib.media.tiling import TilingParams, tile_grid


@dataclass
class TiledParams:
    """타일 추론 노브.

    `border_margin_px` 는 **타일 경계에 닿은 검출을 버리는 폭**이다. 타일 안에서는
    잘린 객체도 온전해 보이므로("작은 공"인지 "잘린 공"인지 구분할 수 없다), 경계에
    붙은 것은 버리고 **겹침 구간에서 옆 타일이 온전히 본 것**을 쓴다. 그래서 겹침이
    가장 큰 객체보다 커야 한다. 이미지 바깥 가장자리는 예외다 — 거기 붙은 객체는
    어느 타일에서도 온전할 수 없다.
    """

    tile_size: int = 640
    stride: int = 480
    merge_iou: float = 0.5
    border_margin_px: int = 4

    def validate(self) -> list[str]:
        errors = TilingParams(tile_size=self.tile_size, stride=self.stride).validate()
        if not 0.0 <= self.merge_iou <= 1.0:
            errors.append("merge_iou must be in [0, 1]")
        if self.border_margin_px < 0:
            errors.append("border_margin_px must be >= 0")
        return errors


def _touches_inner_border(
    box: tuple[float, float, float, float],
    tx: int,
    ty: int,
    tile: int,
    img_w: int,
    img_h: int,
    margin: int,
) -> bool:
    """타일 경계에 붙었나 — **이미지 바깥 테두리는 빼고.**

    `box` 는 타일 안 픽셀 좌표다. 이미지 가장자리와 맞닿은 변은 "잘린 것"이 아니라
    원래 거기가 끝이므로 버리면 안 된다.
    """
    x1, y1, x2, y2 = box
    if x1 <= margin and tx > 0:
        return True
    if y1 <= margin and ty > 0:
        return True
    if x2 >= tile - margin and tx + tile < img_w:
        return True
    if y2 >= tile - margin and ty + tile < img_h:
        return True
    return False


def merge_tile_detections(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """겹침 구간에서 두 번 잡힌 같은 객체를 하나로 — **NMS 다, WBF 가 아니다.**

    좌표를 평균하지 않고 점수 높은 쪽을 남긴다. 평균은 없던 박스를 만드는데, 타일
    경계에서는 한쪽이 잘린 박스일 수 있어 평균이 둘 다보다 나빠진다.
    """
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if any(o.cls == d.cls and iou(o.xyxy, d.xyxy) >= iou_thr for o in out):
            continue
        out.append(d)
    return out


def tiles_for(img_w: int, img_h: int, params: TiledParams) -> list[tuple[int, int]]:
    """타일 좌상단 좌표 목록. 격자 계산은 `lib/media/tiling.py` 것을 그대로 쓴다."""
    grid = TilingParams(tile_size=params.tile_size, stride=params.stride)
    return [(x, y) for _c, _r, x, y in tile_grid(img_w, img_h, grid)]


def restore(
    box: tuple[float, float, float, float],
    tx: int,
    ty: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """타일 안 픽셀 좌표 → **원본 정규화 좌표.**"""
    x1, y1, x2, y2 = box
    return (
        (tx + x1) / img_w,
        (ty + y1) / img_h,
        (tx + x2) / img_w,
        (ty + y2) / img_h,
    )


def collect(
    tile_boxes: list[tuple[int, int, tuple[float, float, float, float], int, float]],
    img_w: int,
    img_h: int,
    params: TiledParams,
    model_id: str,
    cls_map: dict[int, int],
) -> list[Detection]:
    """타일별 검출을 원본 좌표의 `Detection` 목록으로 모은다.

    `tile_boxes` 는 `(tx, ty, 타일 안 픽셀 xyxy, 모델 클래스 id, 점수)` 다 — 추론을
    누가 돌렸는지는 이 함수가 모른다. **ultralytics 를 import 하지 않으므로** 테스트가
    모델 없이 좌표 규칙만 검증할 수 있다.
    """
    kept: list[Detection] = []
    for tx, ty, box, cls, score in tile_boxes:
        if _touches_inner_border(
            box, tx, ty, params.tile_size, img_w, img_h, params.border_margin_px
        ):
            continue
        mapped = cls_map.get(cls)
        if mapped is None:
            continue
        kept.append(
            Detection(
                cls=mapped,
                xyxy=restore(box, tx, ty, img_w, img_h),
                score=score,
                model_id=model_id,
            )
        )
    return merge_tile_detections(kept, params.merge_iou)
