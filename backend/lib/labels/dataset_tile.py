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
    estimate()                  plan 을 센다        → 미리보기
    materialize_for_training()  plan 을 분할별로 실행한다 → train/val 트리 저장

미리보기와 결과가 갈라지는 건 두 곳에서 따로 세기 때문이다. 하나로 묶으면
**"1,122장"이라고 했으면 반드시 1,122장이 나온다.**
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from lib.formats import IMAGE_EXTS
from lib.labels.io import read_label_file, write_data_yaml, write_label_file
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
    hard: int          # 라벨 없는 프레임에서 나온 네거티브 후보 (전부 담긴다)
    incidental: int    # 라벨 있는 프레임의 빈 타일 (목표까지만 담긴다)
    negative_kept: int
    sizes: list[SizeGroup]
    undersized: int
    excluded: int  # 박스와 겹쳤지만 min_visibility 미만이라 통째로 뺀 타일 수
    images: int

    @property
    def negative_candidates(self) -> int:
        return self.hard + self.incidental

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


def _tile_overlaps_any_box(
    boxes: list[Box], img_w: int, img_h: int, tx: int, ty: int, tw: int, th: int
) -> bool:
    """타일 사각형이 박스 중 하나와 조금이라도 겹치는지 — `clip_boxes_to_tile` 이
    쓰는 것과 같은 교차 계산(픽셀 좌표, 면적 기준)이라야 두 판정이 어긋나지 않는다.

    여기서 "겹친다"는 `min_visibility` 를 따지지 않는다 — 미만이라 잘려 나간 박스도
    겹친 것으로 친다. 그래야 그 타일을 "객체가 아예 없는 배경"과 구분할 수 있다.
    """
    for _cls, (x1n, y1n, x2n, y2n) in boxes:
        x1, y1, x2, y2 = x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        ix1, iy1 = max(x1, tx), max(y1, ty)
        ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter > 0:
            return True
    return False


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
    hard: dict[str, list[TilePlanItem]] = {}
    incidental: dict[str, list[TilePlanItem]] = {}
    sizes: dict[tuple[int, int], int] = {}
    undersized = 0
    counted = 0
    excluded = 0

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
            if kept:
                item = TilePlanItem(
                    src_stem=stem, col=col, row=row, x=x, y=y, w=tw, h=th, boxes=kept
                )
                positives.append(item)
            elif _tile_overlaps_any_box(boxes, img_w, img_h, x, y, tw, th):
                # 박스와 겹쳤지만 살아남지 못했다 — 배경이 아니라 "판정 불가".
                # 네거티브로 재활용하면 잘린 객체 픽셀을 배경이라고 가르치게 된다.
                excluded += 1
            else:
                item = TilePlanItem(
                    src_stem=stem, col=col, row=row, x=x, y=y, w=tw, h=th, boxes=kept
                )
                # 라벨이 하나도 없는 프레임 = 일부러 넣은 배경(하드 네거티브).
                # 라벨이 있는 프레임의 빈 구석 = 우연히 생긴 배경.
                pool = incidental if boxes else hard
                pool.setdefault(stem, []).append(item)

    # **포지티브가 하나도 없어도 막지 않는다.** 오탐이 많이 나는 배경 프레임만
    # 모은 데이터셋(하드 네거티브)은 정당한 입력이고, 그걸 전부 담아 학습에 섞는
    # 것이 바로 하려는 일이다. 다만 "포지티브 대비 %" 는 그때 의미를 잃으므로
    # (0의 10% 는 0) 화면이 슬라이더를 잠그고 '전부 담기'를 권한다.
    n_hard = sum(len(v) for v in hard.values())
    n_incidental = sum(len(v) for v in incidental.values())

    # 하드 네거티브는 **언제나 전부** 담는다 — 사용자가 넣은 지시이지 후보가 아니다.
    kept_hard = [item for stem in sorted(hard) for item in hard[stem]]

    if params.keep_all_negatives:
        kept_incidental = [
            item for stem in sorted(incidental) for item in incidental[stem]
        ]
    else:
        target = round(len(positives) * params.negative_ratio)
        remaining = max(0, target - len(kept_hard))
        kept_incidental = _sample_negatives(incidental, remaining, params.seed)

    items = sorted(
        [*positives, *kept_hard, *kept_incidental], key=lambda it: it.stem
    )
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
        hard=n_hard,
        incidental=n_incidental,
        negative_kept=len(kept_hard) + len(kept_incidental),
        sizes=size_groups,
        undersized=undersized,
        excluded=excluded,
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
        "hard": p.hard,
        "incidental": p.incidental,
        "negative_kept": p.negative_kept,
        "total": p.total,
        "sizes": [
            {"w": s.w, "h": s.h, "images": s.images, "cols": s.cols, "rows": s.rows}
            for s in p.sizes
        ],
        "undersized": p.undersized,
        "excluded": p.excluded,
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


TRAIN_SPLITS = ("train", "val")


@dataclass(frozen=True)
class SplitPlan:
    """한 분할이 무엇을 낼지. 화면 미리보기와 실체화가 같은 값을 본다."""

    split: str
    positive: int
    hard: int
    incidental: int
    negative_kept: int
    excluded: int
    total: int
    overshoot: bool  # 하드 네거티브만으로 목표를 넘었나


def _stems_of(reviewed: set[str], splits: dict[str, str], split: str) -> set[str]:
    return {s for s in reviewed if splits.get(s) == split}


def _split_plan(split: str, p: TilePlan, params: TileDatasetParams) -> SplitPlan:
    target = round(p.positive * params.negative_ratio)
    return SplitPlan(
        split=split,
        positive=p.positive,
        hard=p.hard,
        incidental=p.incidental,
        negative_kept=p.negative_kept,
        excluded=p.excluded,
        total=p.total,
        overshoot=not params.keep_all_negatives and p.hard > target,
    )


_EMPTY_SPLIT = dict(
    positive=0, hard=0, incidental=0, negative_kept=0, excluded=0, total=0, overshoot=False
)


def plan_for_training(
    *,
    dataset_dir: Path,
    reviewed: set[str],
    splits: dict[str, str],
    params: TileDatasetParams,
) -> list[SplitPlan]:
    """train 과 val 을 **각각 따로** 계획한다. `test` 는 건드리지 않는다.

    분할마다 독립적으로 비율을 맞추는 이유: train 의 비율은 학습 하이퍼파라미터고,
    val 의 비율은 에폭별 곡선을 안정시키는 값이다. 둘을 합산해 맞추면 어느 쪽도
    의도한 균형이 되지 않는다.
    """
    out: list[SplitPlan] = []
    for split in TRAIN_SPLITS:
        stems = _stems_of(reviewed, splits, split)
        if not stems:
            out.append(SplitPlan(split=split, **_EMPTY_SPLIT))
            continue
        p = plan(dataset_dir=dataset_dir, reviewed=stems, params=params)
        out.append(_split_plan(split, p, params))
    return out


def _class_names(dataset_dir: Path) -> dict[int, str]:
    """`data.yaml` 에 넣을 클래스 이름. `dataset_export.py` 와 같은 규칙이다 —
    데이터셋의 classes.json 을 읽는 방식은 하나뿐이라 형태를 맞춰 둔다."""
    path = dataset_dir / "classes.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text()).get("classes", [])
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(c["id"]): c["name"] for c in raw}


def materialize_for_training(
    *,
    dataset_dir: Path,
    out_dir: Path,
    reviewed: set[str],
    splits: dict[str, str],
    params: TileDatasetParams,
    emit: Emit | None = None,
    cancel_path: Path | None = None,
) -> dict:
    """타일을 `out_dir/images/{split}` · `out_dir/labels/{split}` 에 쓰고 `data.yaml`
    을 만든다. `dataset_export.materialize` 가 내는 것과 같은 모양이라 ultralytics 는
    타일인지 원본인지 모른다.

    **`test` 는 다루지 않는다** — 온전한 원본으로 남겨 두는 것이 이 함수의 존재 이유다.
    """

    def _tick(ev: dict) -> None:
        if emit is not None:
            emit(ev)

    def _check_cancel() -> None:
        if cancel_path is not None and cancel_path.exists():
            raise TileCancelled()

    plans: dict[str, TilePlan | None] = {}
    for split in TRAIN_SPLITS:
        stems = _stems_of(reviewed, splits, split)
        plans[split] = (
            plan(dataset_dir=dataset_dir, reviewed=stems, params=params) if stems else None
        )

    train_plan = plans["train"]
    if train_plan is None or train_plan.total == 0:
        raise TileError("Nothing to train on — assign some reviewed images to train")

    images = _images_by_stem(dataset_dir / "raw")
    total = sum(p.total for p in plans.values() if p is not None)
    _tick({"phase": "tiling", "done": 0, "total": total})

    saved = 0
    counts: dict[str, dict] = {}
    for split in TRAIN_SPLITS:
        p = plans[split]
        counts[split] = {
            "positive": p.positive if p else 0,
            "negative": p.negative_kept if p else 0,
            "total": p.total if p else 0,
        }
        if p is None:
            continue
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        by_src: dict[str, list[TilePlanItem]] = {}
        for item in p.items:
            by_src.setdefault(item.src_stem, []).append(item)

        # 같은 원본을 여러 타일이 쓰므로 원본은 한 번만 연다
        for src_stem in sorted(by_src):
            _check_cancel()
            src = images[src_stem]
            with Image.open(src) as im:
                im.load()
                for item in by_src[src_stem]:
                    tile = im.crop((item.x, item.y, item.x + item.w, item.y + item.h))
                    _save_tile(tile, img_out / f"{item.stem}{src.suffix}")
                    # 박스가 없어도 빈 파일을 쓴다 — "여기엔 없다"는 학습 신호다
                    write_label_file(lbl_out / f"{item.stem}.txt", item.boxes)
                    saved += 1
                    if saved % 20 == 0 or saved == total:
                        _check_cancel()
                        _tick({"phase": "tiling", "done": saved, "total": total})

    # val 이 비면 ultralytics 가 학습을 못 돈다 — train 을 가리켜 두면 최소한 돈다
    write_data_yaml(
        out_dir / "data.yaml",
        _class_names(dataset_dir),
        train="images/train",
        val="images/val" if counts["val"]["total"] else "images/train",
    )
    return {**counts, "saved": saved}
