"""데이터셋 → 타일 데이터셋. 순수 파일 IO — DB 도 HTTP 도 torch 도 모른다.

`dataset_export.py` 의 형제다. 저쪽이 데이터셋을 YOLO 트리로 펼친다면 이쪽은
데이터셋을 **더 작은 데이터셋**으로 바꾼다.

1920px 원본에서 수십 px 인 공은 풀 프레임 학습이 640 으로 줄이는 순간 사라진다.
타일로 잘라 학습하면 그 공이 타일 안에서 큼직해진다. 문제는 타일 대부분이
배경이라는 것 — 객체 하나뿐인 프레임을 8조각 내면 포지티브는 많아야 절반이고
보통 한둘이다. 그대로 넣으면 모델이 "아무것도 없다"로 수렴한다. 그래서 **네거티브를
얼마나 남길지**를 사람이 정한다.

함수 셋이 **한 계획을 공유한다.**

    plan()         이미지 크기 + 라벨만 읽는다 (픽셀 디코딩 없음).
                   어떤 타일이 나오고 어느 게 포지티브고 어느 네거티브를
                   남길지까지 여기서 전부 결정한다.
    estimate()     plan 을 센다        → 미리보기
    materialize()  plan 을 실행한다     → 실제 타일 저장

미리보기와 결과가 갈라지는 건 두 곳에서 따로 세기 때문이다. 하나로 묶으면
**"1,122장"이라고 했으면 반드시 1,122장이 나온다.**
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from lib.formats import IMAGE_EXTS
from lib.labels.io import read_label_file, write_label_file
from lib.media.tiling import (
    TilingParams,
    clip_boxes_to_tile,
    tile_grid,
    tile_offsets,
    tile_stem,
)

Box = tuple[int, tuple[float, float, float, float]]
Emit = Callable[[dict], None]


class TileError(Exception):
    """타일링할 것이 없거나 설정이 틀렸다."""


class TileCancelled(Exception):
    """CANCEL 센티널이 보였다."""


@dataclass(frozen=True)
class TileDatasetParams:
    """타일링 노브.

    `negative_ratio` 는 **포지티브 장수 대비 비율**이다 — `0.1` 이면 포지티브
    100장에 네거티브 10장. "전부 담기"는 비율로 표현하지 않고
    `keep_all_negatives` 로 둔다. 700장을 넣으려고 `7.0` 을 계산해서 치게 하지 않는다.
    """

    tile_size: int = 640
    stride: int = 480
    min_visibility: float = 0.6
    negative_ratio: float = 0.1
    keep_all_negatives: bool = False
    seed: int = 0

    def tiling(self) -> TilingParams:
        return TilingParams(
            tile_size=self.tile_size,
            stride=self.stride,
            min_visibility=self.min_visibility,
        )

    def validate(self) -> list[str]:
        errors = self.tiling().validate()
        if self.negative_ratio < 0:
            errors.append("negative_ratio must be >= 0")
        return errors


@dataclass(frozen=True)
class TilePlanItem:
    src_stem: str
    col: int
    row: int
    x: int
    y: int
    w: int
    h: int
    boxes: list[Box]

    @property
    def stem(self) -> str:
        return tile_stem(self.src_stem, self.col, self.row)

    @property
    def positive(self) -> bool:
        return bool(self.boxes)


@dataclass(frozen=True)
class SizeGroup:
    w: int
    h: int
    images: int
    cols: int
    rows: int


@dataclass(frozen=True)
class TilePlan:
    items: list[TilePlanItem]  # **남길 것만** — 네거티브 샘플링이 이미 적용됐다
    positive: int
    negative_candidates: int
    negative_kept: int
    sizes: list[SizeGroup]
    undersized: int
    images: int

    @property
    def total(self) -> int:
        return len(self.items)


def _images_by_stem(raw_dir: Path) -> dict[str, Path]:
    # `dataset_export.py` 와 같은 규칙이다 — 데이터셋의 raw/ 를 보는 방식은 하나다
    if not raw_dir.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(raw_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    }


def _sample_negatives(
    by_stem: dict[str, list[TilePlanItem]], target: int, seed: int
) -> list[TilePlanItem]:
    """원본 라운드로빈으로 `target` 장까지 고른다. **결정론적이다.**

    한 프레임에서 뭉텅이로 집으면 배경이 한 장면에 쏠린다. 원본을 돌아가며 한 장씩
    집으면 원본 장면 수만큼 배경이 다양해진다. 순서는 `seed` 로만 정해지므로
    같은 입력이면 언제나 같은 목록이 나온다.
    """
    if target <= 0:
        return []
    rng = random.Random(seed)
    stems = sorted(by_stem)
    rng.shuffle(stems)
    pools = {}
    for stem in stems:
        pool = list(by_stem[stem])
        rng.shuffle(pool)
        pools[stem] = pool

    picked: list[TilePlanItem] = []
    depth = 0
    while len(picked) < target:
        added = False
        for stem in stems:
            pool = pools[stem]
            if depth >= len(pool):
                continue
            picked.append(pool[depth])
            added = True
            if len(picked) >= target:
                break
        if not added:
            break  # 후보 소진 — 있는 만큼만 담는다
        depth += 1
    return picked


def plan(
    *, dataset_dir: Path, reviewed: set[str], params: TileDatasetParams
) -> TilePlan:
    """무엇을 만들지 정한다. **픽셀을 디코딩하지 않는다.**

    이미지 크기는 Pillow 가 헤더만 읽고 라벨은 텍스트다 — 수천 장도 초 단위다.
    """
    errors = params.validate()
    if errors:
        raise TileError("; ".join(errors))

    images = _images_by_stem(dataset_dir / "raw")
    labels_dir = dataset_dir / "labels"
    stems = sorted(s for s in reviewed if s in images)
    if not stems:
        raise TileError("Nothing to tile — review some images first")

    grid = params.tiling()
    positives: list[TilePlanItem] = []
    negatives: dict[str, list[TilePlanItem]] = {}
    sizes: dict[tuple[int, int], int] = {}
    undersized = 0
    counted = 0

    for stem in stems:
        try:
            with Image.open(images[stem]) as im:
                img_w, img_h = im.size
        except OSError:
            continue  # 열 수 없는 파일 — 배정표에만 남은 줄
        counted += 1
        sizes[(img_w, img_h)] = sizes.get((img_w, img_h), 0) + 1
        if img_w < params.tile_size or img_h < params.tile_size:
            undersized += 1

        label_path = labels_dir / f"{stem}.txt"
        boxes: list[Box] = read_label_file(label_path) if label_path.exists() else []

        tw, th = min(params.tile_size, img_w), min(params.tile_size, img_h)
        for col, row, x, y in tile_grid(img_w, img_h, grid):
            kept = clip_boxes_to_tile(
                boxes, img_w, img_h, x, y, grid, tile_w=tw, tile_h=th
            )
            item = TilePlanItem(
                src_stem=stem, col=col, row=row, x=x, y=y, w=tw, h=th, boxes=kept
            )
            if kept:
                positives.append(item)
            else:
                negatives.setdefault(stem, []).append(item)

    # **포지티브가 하나도 없어도 막지 않는다.** 오탐이 많이 나는 배경 프레임만
    # 모은 데이터셋(하드 네거티브)은 정당한 입력이고, 그걸 전부 담아 학습에 섞는
    # 것이 바로 하려는 일이다. 다만 "포지티브 대비 %" 는 그때 의미를 잃으므로
    # (0의 10% 는 0) 화면이 슬라이더를 잠그고 '전부 담기'를 권한다.
    candidates = sum(len(v) for v in negatives.values())

    target = (
        candidates
        if params.keep_all_negatives
        else min(candidates, round(len(positives) * params.negative_ratio))
    )
    kept_negatives = _sample_negatives(negatives, target, params.seed)

    items = sorted([*positives, *kept_negatives], key=lambda it: it.stem)
    size_groups = [
        SizeGroup(
            w=w,
            h=h,
            images=n,
            cols=len(tile_offsets(w, params.tile_size, params.stride)),
            rows=len(tile_offsets(h, params.tile_size, params.stride)),
        )
        for (w, h), n in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return TilePlan(
        items=items,
        positive=len(positives),
        negative_candidates=candidates,
        negative_kept=len(kept_negatives),
        sizes=size_groups,
        undersized=undersized,
        images=counted,
    )


def estimate(
    *, dataset_dir: Path, reviewed: set[str], params: TileDatasetParams
) -> dict:
    """미리보기용 수치. `plan` 을 세기만 한다."""
    p = plan(dataset_dir=dataset_dir, reviewed=reviewed, params=params)
    return {
        "images": p.images,
        "tiles": p.positive + p.negative_candidates,
        "positive": p.positive,
        "negative": p.negative_candidates,
        "negative_kept": p.negative_kept,
        "total": p.total,
        "sizes": [
            {"w": s.w, "h": s.h, "images": s.images, "cols": s.cols, "rows": s.rows}
            for s in p.sizes
        ],
        "undersized": p.undersized,
    }


def _save_tile(im: Image.Image, dst: Path) -> None:
    """원본 확장자를 따라 저장한다.

    타일은 잘라낸 새 픽셀이라 하드링크가 불가능하다 — 어차피 다시 인코딩하므로
    JPEG 는 품질을 높게 잡는다. 학습 데이터에 압축 잡음을 얹을 이유가 없다.
    """
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        im.convert("RGB").save(dst, "JPEG", quality=95, subsampling=0)
    else:
        im.save(dst)


def materialize(
    *,
    dataset_dir: Path,
    out_dir: Path,
    reviewed: set[str],
    params: TileDatasetParams,
    emit: Emit | None = None,
    cancel_path: Path | None = None,
) -> dict:
    """`plan` 이 정한 타일을 `out_dir/raw` 와 `out_dir/labels` 에 쓴다.

    같은 원본을 여러 타일이 쓰므로 원본은 **한 번만 연다.**
    """

    def _tick(ev: dict) -> None:
        if emit is not None:
            emit(ev)

    def _check_cancel() -> None:
        if cancel_path is not None and cancel_path.exists():
            raise TileCancelled()

    p = plan(dataset_dir=dataset_dir, reviewed=reviewed, params=params)
    images = _images_by_stem(dataset_dir / "raw")
    raw_out = out_dir / "raw"
    labels_out = out_dir / "labels"
    raw_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    by_src: dict[str, list[TilePlanItem]] = {}
    for item in p.items:
        by_src.setdefault(item.src_stem, []).append(item)

    total = p.total
    _tick({"phase": "start", "total": total})

    saved = 0
    for src_stem in sorted(by_src):
        _check_cancel()
        src = images[src_stem]
        with Image.open(src) as im:
            im.load()
            for item in by_src[src_stem]:
                tile = im.crop((item.x, item.y, item.x + item.w, item.y + item.h))
                _save_tile(tile, raw_out / f"{item.stem}{src.suffix}")
                # 박스가 없어도 빈 파일을 쓴다 — "여기엔 없다"는 학습 신호다
                write_label_file(labels_out / f"{item.stem}.txt", item.boxes)
                saved += 1
                if saved % 20 == 0 or saved == total:
                    _check_cancel()
                    _tick({"phase": "tile", "done": saved, "total": total})

    return {
        "images": p.images,
        "tiles": total,
        "positive": p.positive,
        "negative": p.negative_kept,
        "saved": saved,
    }
