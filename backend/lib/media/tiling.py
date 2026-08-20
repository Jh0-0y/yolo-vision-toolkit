"""이미지 타일링 — 큰 프레임(예: 1920×1080)을 학습용 타일로 쪼갠다.

목적: 1920px 원본을 타일(기본 640, stride 480 = 겹침 160px/25%)로 나눠
작은 객체(공)를 상대적으로 크게 학습할 데이터셋을 만든다.

  - 겹침 필수: 겹침 0이면 타일 경계에 걸린 객체가 양쪽에 반쪽씩 남아
    NMS가 합치지 못하고 두 개로 탐지된다. 겹침을 최대 객체 크기 이상으로
    주면 그 객체는 어떤 타일엔 온전히 담긴다.
  - 마지막 타일은 프레임 끝에 클램프한다 (그리드가 딱 안 떨어져도 커버).
  - 라벨은 스크립트가 자동 처리: 타일에 들어온 박스의 가시 비율을 계산해
    min_visibility 이상이면 타일 경계로 클립해 유지, 미만이면 삭제.

1920×1080 기준: 640/480 → 가로 4 × 세로 2 = 8타일/장.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TILE_SIZE = 640
DEFAULT_STRIDE = 480  # 겹침 = tile - stride = 160px (25%)
# 데이터 특성상 60% 이상 보여야 탐지 대상으로 취급한다 (프로젝트 결정).
DEFAULT_MIN_VISIBILITY = 0.6


@dataclass(frozen=True)
class TilingParams:
    tile_size: int = DEFAULT_TILE_SIZE
    stride: int = DEFAULT_STRIDE
    min_visibility: float = DEFAULT_MIN_VISIBILITY
    drop_empty: bool = True  # 라벨 있는 원본에서 박스가 하나도 없는 타일은 제외

    @property
    def overlap(self) -> int:
        return self.tile_size - self.stride

    def validate(self) -> list[str]:
        """파라미터 위반 목록 (빈 리스트 = 통과)."""
        errors: list[str] = []
        if self.tile_size <= 0:
            errors.append("tile_size must be > 0")
        if self.stride <= 0:
            errors.append("stride must be > 0")
        elif self.stride > self.tile_size:
            errors.append("stride must be <= tile_size (overlap would be negative)")
        if not 0.0 <= self.min_visibility <= 1.0:
            errors.append("min_visibility must be in [0, 1]")
        return errors


def tile_offsets(length: int, tile: int, stride: int) -> list[int]:
    """한 축의 타일 시작 좌표들 — stride 간격, 마지막은 끝에 클램프."""
    if tile >= length:
        return [0]
    offsets = list(range(0, length - tile + 1, stride))
    last = length - tile
    if offsets[-1] != last:
        offsets.append(last)  # 프레임 끝 클램프
    return offsets


def tile_grid(img_w: int, img_h: int, params: TilingParams) -> list[tuple[int, int, int, int]]:
    """타일 목록 [(col, row, x, y)] — x/y는 픽셀 좌상단."""
    xs = tile_offsets(img_w, params.tile_size, params.stride)
    ys = tile_offsets(img_h, params.tile_size, params.stride)
    return [(c, r, x, y) for r, y in enumerate(ys) for c, x in enumerate(xs)]


def tile_stem(stem: str, col: int, row: int) -> str:
    """타일 파일명 stem — 원본 stem + 타일 그리드 위치."""
    return f"{stem}_r{row}c{col}"


def clip_boxes_to_tile(
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    img_w: int,
    img_h: int,
    tx: int,
    ty: int,
    params: TilingParams,
    tile_w: int | None = None,
    tile_h: int | None = None,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """타일 (tx, ty)에 대한 라벨 자동 변환.

    boxes: 원본 이미지 정규화 xyxy [(cls, (x1,y1,x2,y2)), ...].
    각 박스의 타일 내 가시 비율(교차면적/원면적)을 계산해
    min_visibility 이상이면 타일 경계로 클립 후 타일 좌표로 재정규화, 미만이면 버린다.

    `tile_w`·`tile_h` 는 **실제** 타일 크기다. 이미지가 타일보다 작으면
    `tile_offsets` 가 `[0]` 하나를 돌려주고 그 타일은 이미지 크기 그대로라,
    `params.tile_size` 로 나누면 좌표가 찌그러진다. 생략하면 `params.tile_size`.
    """
    tw = tile_w if tile_w is not None else params.tile_size
    th = tile_h if tile_h is not None else params.tile_size
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for cls, (x1n, y1n, x2n, y2n) in boxes:
        # 픽셀 좌표로
        x1, y1, x2, y2 = x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        # 타일과 교차
        ix1, iy1 = max(x1, tx), max(y1, ty)
        ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter / area < params.min_visibility:
            continue  # 너무 잘림 — 반쪽 박스로 학습 오염 방지
        # 타일 좌표로 재정규화 (클립된 박스)
        out.append(
            (
                cls,
                (
                    (ix1 - tx) / tw,
                    (iy1 - ty) / th,
                    (ix2 - tx) / tw,
                    (iy2 - ty) / th,
                ),
            )
        )
    return out
