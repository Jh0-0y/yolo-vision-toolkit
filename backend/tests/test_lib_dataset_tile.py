"""타일 데이터셋 계획 — 격자·포지티브 판정·네거티브 샘플링."""

from pathlib import Path

import pytest
from PIL import Image

from lib.labels.dataset_tile import (
    TileCancelled,
    TileDatasetParams,
    TileError,
    estimate,
    materialize,
    plan,
)
from lib.labels.io import read_label_file, write_label_file

P = TileDatasetParams()  # 640 / 480 / 0.6 / 10%


def make_dataset(root: Path, images: dict[str, tuple[int, int]],
                 labels: dict[str, list[tuple[int, tuple[float, float, float, float]]]] | None = None) -> Path:
    """`{stem: (w, h)}` 로 이미지를, `{stem: boxes}` 로 라벨을 만든다."""
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for stem, (w, h) in images.items():
        Image.new("RGB", (w, h), (10, 20, 30)).save(root / "raw" / f"{stem}.jpg")
    for stem, boxes in (labels or {}).items():
        write_label_file(root / "labels" / f"{stem}.txt", boxes)
    return root


def test_grid_is_eight_tiles_for_full_hd(tmp_path):
    """1920×1080 + 640/480 = 가로 4 × 세로 2 = 8타일."""
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(keep_all_negatives=True))
    assert p.positive + p.negative_candidates == 8
    assert p.sizes[0].cols == 4
    assert p.sizes[0].rows == 2
    assert p.undersized == 0


def test_positive_is_decided_after_clipping(tmp_path):
    """박스가 있는 타일만 포지티브다. 가운데 작은 박스 하나면 포지티브는 소수다."""
    # 960~1000px, 500~540px 에 있는 40px 박스 — 타일 (480,440) 안에 온전히 들어간다
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(keep_all_negatives=True))
    assert p.positive >= 1
    assert p.positive < 8
    assert p.positive + p.negative_candidates == 8


def test_negative_ratio_is_relative_to_positives(tmp_path):
    """포지티브 대비 %. 후보가 남아돌아도 목표치까지만 담는다."""
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    images = {f"img{i:03d}": (1920, 1080) for i in range(10)}
    root = make_dataset(tmp_path / "ds", images, {s: boxes for s in images})
    p = plan(dataset_dir=root, reviewed=set(images), params=TileDatasetParams(negative_ratio=1.0))
    assert p.negative_kept == p.positive
    assert p.total == p.positive * 2


def test_keep_all_negatives_takes_everything(tmp_path):
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(keep_all_negatives=True))
    assert p.negative_kept == p.negative_candidates
    assert p.total == 8


def test_zero_ratio_drops_every_negative(tmp_path):
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(negative_ratio=0.0))
    assert p.negative_kept == 0
    assert p.total == p.positive


def test_fewer_candidates_than_target_is_not_an_error(tmp_path):
    """후보가 목표보다 적으면 있는 만큼만. 에러가 아니다.

    640×640 이미지는 타일 한 장이고 그 안에 박스가 있으니 네거티브 후보가 0이다.
    500% 를 달라고 해도 0장이 나오고, 그건 정상이다.
    """
    boxes = [(0, (0.4, 0.4, 0.6, 0.6))]
    root = make_dataset(tmp_path / "ds", {"a": (640, 640)}, {"a": boxes})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(negative_ratio=5.0))
    assert p.positive == 1
    assert p.negative_candidates == 0
    assert p.negative_kept == 0
    assert p.total == 1


def test_sampling_is_deterministic(tmp_path):
    """같은 입력 → 같은 목록. 미리보기와 결과가 갈라지면 기능이 거짓말이 된다."""
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    images = {f"img{i:03d}": (1920, 1080) for i in range(6)}
    root = make_dataset(tmp_path / "ds", images, {s: boxes for s in images})
    a = plan(dataset_dir=root, reviewed=set(images), params=TileDatasetParams(negative_ratio=0.5))
    b = plan(dataset_dir=root, reviewed=set(images), params=TileDatasetParams(negative_ratio=0.5))
    assert [i.stem for i in a.items] == [i.stem for i in b.items]


def test_negatives_spread_across_source_images(tmp_path):
    """라운드로빈 — 한 프레임에서 뭉텅이로 집지 않는다.

    원본 6장 × (포지티브 4 · 네거티브 4) = 포지티브 24 · 후보 24.
    25% 면 목표 6장이고, 라운드로빈이면 **원본마다 정확히 한 장**이다.
    한 원본에서 몰아 뽑으면 6장이 두 원본에서만 나온다.
    """
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    images = {f"img{i:03d}": (1920, 1080) for i in range(6)}
    root = make_dataset(tmp_path / "ds", images, {s: boxes for s in images})
    p = plan(dataset_dir=root, reviewed=set(images), params=TileDatasetParams(negative_ratio=0.25))
    negatives = [i for i in p.items if not i.positive]
    assert len(negatives) == 6
    assert len({i.src_stem for i in negatives}) == 6


def test_hard_negative_dataset_is_not_blocked(tmp_path):
    """라벨이 하나도 없는 데이터셋(하드 네거티브)도 그대로 타일링된다.

    오탐이 많이 나는 배경 프레임만 모아 전부 담는 건 정당한 사용이다 — 막지 않는다.
    """
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(keep_all_negatives=True))
    assert p.positive == 0
    assert p.total == 8


def test_ratio_against_zero_positives_yields_nothing_but_does_not_raise(tmp_path):
    """포지티브가 0이면 "포지티브 대비 %" 는 0장이다. 그래도 예외는 아니다 —
    무엇을 만들지는 사람이 정하고, 화면이 0장임을 보여준다."""
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(negative_ratio=0.1))
    assert p.total == 0


def test_nothing_reviewed_is_rejected(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)})
    with pytest.raises(TileError):
        plan(dataset_dir=root, reviewed=set(), params=TileDatasetParams())


def test_bad_params_are_rejected(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)})
    with pytest.raises(TileError):
        plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(stride=700))


def test_undersized_image_yields_one_tile(tmp_path):
    """타일보다 작으면 원본 크기 타일 한 장. 패딩하지 않는다."""
    boxes = [(0, (0.4, 0.4, 0.6, 0.6))]
    root = make_dataset(tmp_path / "ds", {"a": (400, 300)}, {"a": boxes})
    p = plan(dataset_dir=root, reviewed={"a"}, params=TileDatasetParams(keep_all_negatives=True))
    assert p.total == 1
    assert p.undersized == 1
    _cls, (x1, y1, x2, y2) = p.items[0].boxes[0]
    assert (x1, y1, x2, y2) == pytest.approx((0.4, 0.4, 0.6, 0.6))


def test_estimate_matches_plan(tmp_path):
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    params = TileDatasetParams(negative_ratio=0.5)
    p = plan(dataset_dir=root, reviewed={"a"}, params=params)
    e = estimate(dataset_dir=root, reviewed={"a"}, params=params)
    assert e["total"] == p.total
    assert e["positive"] == p.positive
    assert e["negative"] == p.negative_candidates
    assert e["tiles"] == 8
    assert e["images"] == 1
    assert e["sizes"][0] == {"w": 1920, "h": 1080, "images": 1, "cols": 4, "rows": 2}


def test_materialize_writes_exactly_what_estimate_promised(tmp_path):
    """예상 장수와 실제 파일 수가 같다 — 이게 어긋나면 미리보기가 거짓말이다."""
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    images = {f"img{i:03d}": (1920, 1080) for i in range(4)}
    root = make_dataset(tmp_path / "ds", images, {s: boxes for s in images})
    out = tmp_path / "tiled"
    params = TileDatasetParams(negative_ratio=0.5)

    e = estimate(dataset_dir=root, reviewed=set(images), params=params)
    r = materialize(dataset_dir=root, out_dir=out, reviewed=set(images), params=params)

    written = sorted(p.name for p in (out / "raw").iterdir())
    assert len(written) == e["total"]
    assert r["saved"] == e["total"]
    assert len(list((out / "labels").iterdir())) == e["total"]


def test_tile_images_are_tile_sized(tmp_path):
    boxes = [(0, (0.0, 0.0, 1.0, 1.0))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    out = tmp_path / "tiled"
    materialize(dataset_dir=root, out_dir=out, reviewed={"a"},
                params=TileDatasetParams(keep_all_negatives=True))
    for p in (out / "raw").iterdir():
        with Image.open(p) as im:
            assert im.size == (640, 640)


def test_tile_names_carry_grid_position(tmp_path):
    boxes = [(0, (0.0, 0.0, 1.0, 1.0))]
    root = make_dataset(tmp_path / "ds", {"game_00001": (1920, 1080)}, {"game_00001": boxes})
    out = tmp_path / "tiled"
    materialize(dataset_dir=root, out_dir=out, reviewed={"game_00001"},
                params=TileDatasetParams(keep_all_negatives=True))
    names = {p.stem for p in (out / "raw").iterdir()}
    assert "game_00001_r0c0" in names
    assert "game_00001_r1c3" in names


def test_empty_tiles_get_an_empty_label_file(tmp_path):
    """빈 라벨은 "안 그렸다"가 아니라 "여기엔 없다"는 학습 신호다."""
    boxes = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    out = tmp_path / "tiled"
    materialize(dataset_dir=root, out_dir=out, reviewed={"a"},
                params=TileDatasetParams(keep_all_negatives=True))
    empties = [p for p in (out / "labels").iterdir() if p.read_text().strip() == ""]
    assert empties  # 라벨 파일이 존재하되 내용이 없다


def test_clipped_boxes_are_renormalized_to_the_tile(tmp_path):
    """타일 (0,0) 안에 온전히 든 박스는 타일 기준 좌표로 다시 매겨진다."""
    # 원본 320~640px, 320~640px → 타일 0 (0~640) 안에서 0.5~1.0
    boxes = [(0, (320 / 1920, 320 / 1080, 640 / 1920, 640 / 1080))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    out = tmp_path / "tiled"
    materialize(dataset_dir=root, out_dir=out, reviewed={"a"},
                params=TileDatasetParams(keep_all_negatives=True))
    got = read_label_file(out / "labels" / "a_r0c0.txt")
    assert len(got) == 1
    _cls, (x1, y1, x2, y2) = got[0]
    assert (x1, y1, x2, y2) == pytest.approx((0.5, 0.5, 1.0, 1.0), abs=1e-3)


def test_progress_events_are_emitted(tmp_path):
    boxes = [(0, (0.0, 0.0, 1.0, 1.0))]
    root = make_dataset(tmp_path / "ds", {"a": (1920, 1080)}, {"a": boxes})
    events: list[dict] = []
    materialize(dataset_dir=root, out_dir=tmp_path / "tiled", reviewed={"a"},
                params=TileDatasetParams(keep_all_negatives=True), emit=events.append)
    assert events[0]["phase"] == "start"
    assert events[0]["total"] == 8
    assert events[-1]["phase"] == "tile"
    assert events[-1]["done"] == 8


def test_cancel_sentinel_stops_the_run(tmp_path):
    boxes = [(0, (0.0, 0.0, 1.0, 1.0))]
    images = {f"img{i:03d}": (1920, 1080) for i in range(20)}
    root = make_dataset(tmp_path / "ds", images, {s: boxes for s in images})
    cancel = tmp_path / "CANCEL"
    cancel.touch()  # 시작 전부터 켜 둔다 — 첫 확인 지점에서 멈춰야 한다
    with pytest.raises(TileCancelled):
        materialize(dataset_dir=root, out_dir=tmp_path / "tiled", reviewed=set(images),
                    params=TileDatasetParams(keep_all_negatives=True), cancel_path=cancel)
