"""학습용 타일 트리 — 분할별 계획과 실체화."""

from pathlib import Path

import pytest
from PIL import Image

from lib.labels.dataset_tile import (
    TileCancelled,
    TileDatasetParams,
    TileError,
    materialize_for_training,
    plan_for_training,
)
from lib.labels.io import read_label_file, write_label_file

BALL = [(0, (960 / 1920, 500 / 1080, 1000 / 1920, 540 / 1080))]


def make_dataset(root: Path, images: dict[str, tuple[int, int]], labels: dict | None = None) -> Path:
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for stem, (w, h) in images.items():
        Image.new("RGB", (w, h), (10, 20, 30)).save(root / "raw" / f"{stem}.jpg")
    for stem, boxes in (labels or {}).items():
        write_label_file(root / "labels" / f"{stem}.txt", boxes)
    (root / "classes.json").write_text('{"classes": [{"id": 0, "name": "ball"}]}')
    return root


def _seed(tmp_path: Path, n_train: int, n_val: int, n_hard_train: int = 0, n_hard_val: int = 0):
    """train/val 에 포지티브와 하드 네거티브를 원하는 만큼 깔아 준다."""
    images, labels, splits = {}, {}, {}
    for i in range(n_train):
        s = f"t{i:02d}"; images[s] = (1920, 1080); labels[s] = BALL; splits[s] = "train"
    for i in range(n_val):
        s = f"v{i:02d}"; images[s] = (1920, 1080); labels[s] = BALL; splits[s] = "val"
    for i in range(n_hard_train):
        s = f"ht{i:02d}"; images[s] = (1920, 1080); splits[s] = "train"
    for i in range(n_hard_val):
        s = f"hv{i:02d}"; images[s] = (1920, 1080); splits[s] = "val"
    root = make_dataset(tmp_path / "ds", images, labels)
    return root, set(images), splits


def test_each_split_hits_its_own_ratio(tmp_path):
    """train 과 val 은 **각각** 목표를 맞춘다 — 합산이 아니다."""
    root, reviewed, splits = _seed(tmp_path, n_train=10, n_val=10)

    plans = plan_for_training(
        dataset_dir=root, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(negative_ratio=0.5),
    )
    by = {p.split: p for p in plans}

    # 각 분할 포지티브 40장 → 목표 20장씩
    assert by["train"].positive == 40 and by["train"].negative_kept == 20
    assert by["val"].positive == 40 and by["val"].negative_kept == 20


def test_hard_negatives_are_counted_per_split(tmp_path):
    """하드 네거티브는 그 프레임이 배정된 분할에만 들어간다."""
    root, reviewed, splits = _seed(tmp_path, n_train=10, n_val=10, n_hard_train=2)

    by = {p.split: p for p in plan_for_training(
        dataset_dir=root, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(negative_ratio=0.1),
    )}

    assert by["train"].hard == 16   # 2프레임 × 8타일
    assert by["val"].hard == 0


def test_overshoot_is_flagged(tmp_path):
    """하드만으로 목표를 넘으면 그 사실이 드러난다."""
    root, reviewed, splits = _seed(tmp_path, n_train=10, n_val=1, n_hard_val=1)

    by = {p.split: p for p in plan_for_training(
        dataset_dir=root, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(negative_ratio=0.1),
    )}

    # val: 포지티브 4장 · 목표 0장 · 하드 8장 → 초과
    assert by["val"].overshoot is True
    assert by["train"].overshoot is False


def test_test_split_is_never_tiled(tmp_path):
    """test 는 손대지 않는다 — 그게 이 설계의 전부다."""
    root, reviewed, splits = _seed(tmp_path, n_train=5, n_val=5)
    splits["t00"] = "test"

    plans = plan_for_training(
        dataset_dir=root, reviewed=reviewed, splits=splits, params=TileDatasetParams(),
    )

    assert {p.split for p in plans} == {"train", "val"}


def test_materialize_writes_the_yolo_split_tree(tmp_path):
    root, reviewed, splits = _seed(tmp_path, n_train=4, n_val=2)
    out = tmp_path / "run"

    result = materialize_for_training(
        dataset_dir=root, out_dir=out, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(negative_ratio=0.25),
    )

    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "val").is_dir()
    assert (out / "labels" / "train").is_dir()
    assert (out / "data.yaml").exists()
    assert len(list((out / "images" / "train").iterdir())) == result["train"]["total"]
    assert len(list((out / "images" / "val").iterdir())) == result["val"]["total"]


def test_materialize_matches_the_plan_exactly(tmp_path):
    """미리보기가 약속한 장수가 실제로 나온다."""
    root, reviewed, splits = _seed(tmp_path, n_train=6, n_val=3, n_hard_train=1)
    params = TileDatasetParams(negative_ratio=0.3)

    planned = {p.split: p.total for p in plan_for_training(
        dataset_dir=root, reviewed=reviewed, splits=splits, params=params)}
    result = materialize_for_training(
        dataset_dir=root, out_dir=tmp_path / "run", reviewed=reviewed,
        splits=splits, params=params)

    assert result["train"]["total"] == planned["train"]
    assert result["val"]["total"] == planned["val"]


def test_tiles_are_tile_sized_and_labels_renormalized(tmp_path):
    root, reviewed, splits = _seed(tmp_path, n_train=1, n_val=0)
    out = tmp_path / "run"

    materialize_for_training(
        dataset_dir=root, out_dir=out, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(keep_all_negatives=True),
    )

    for p in (out / "images" / "train").iterdir():
        with Image.open(p) as im:
            assert im.size == (640, 640)
    # 라벨이 있는 타일은 좌표가 [0,1] 안에 있다
    for p in (out / "labels" / "train").iterdir():
        for _cls, (x1, y1, x2, y2) in read_label_file(p):
            assert 0.0 <= x1 <= x2 <= 1.0
            assert 0.0 <= y1 <= y2 <= 1.0


def test_empty_train_split_is_rejected(tmp_path):
    root, reviewed, splits = _seed(tmp_path, n_train=0, n_val=2)

    with pytest.raises(TileError):
        materialize_for_training(
            dataset_dir=root, out_dir=tmp_path / "run", reviewed=reviewed,
            splits=splits, params=TileDatasetParams(),
        )


def test_empty_val_split_points_data_yaml_at_train(tmp_path):
    """val 이 비어도 학습은 돈다 — data.yaml 의 val 이 train 을 가리킨다."""
    root, reviewed, splits = _seed(tmp_path, n_train=4, n_val=0)
    out = tmp_path / "run"

    materialize_for_training(
        dataset_dir=root, out_dir=out, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(),
    )

    text = (out / "data.yaml").read_text()
    assert "images/train" in text


def test_progress_events_are_emitted(tmp_path):
    root, reviewed, splits = _seed(tmp_path, n_train=4, n_val=2)
    events: list[dict] = []

    materialize_for_training(
        dataset_dir=root, out_dir=tmp_path / "run", reviewed=reviewed, splits=splits,
        params=TileDatasetParams(negative_ratio=0.25), emit=events.append,
    )

    total = events[0]["total"]
    assert events[0]["phase"] == "tiling"
    assert total > 0
    assert events[-1]["done"] == total


def test_cancel_sentinel_stops_the_run(tmp_path):
    """CANCEL 센티널이 보이면 남은 타일을 쓰지 않고 즉시 멈춘다."""
    root, reviewed, splits = _seed(tmp_path, n_train=20, n_val=0)
    cancel = tmp_path / "CANCEL"
    cancel.touch()  # 시작 전부터 켜 둔다 — 그래야 진짜로 일찍 멈추는지 드러난다

    with pytest.raises(TileCancelled):
        materialize_for_training(
            dataset_dir=root, out_dir=tmp_path / "run", reviewed=reviewed, splits=splits,
            params=TileDatasetParams(keep_all_negatives=True), cancel_path=cancel,
        )


def test_empty_tiles_get_an_empty_label_file(tmp_path):
    """빈 라벨은 "안 그렸다"가 아니라 "여기엔 없다"는 학습 신호다."""
    root, reviewed, splits = _seed(tmp_path, n_train=1, n_val=0)
    out = tmp_path / "run"

    materialize_for_training(
        dataset_dir=root, out_dir=out, reviewed=reviewed, splits=splits,
        params=TileDatasetParams(keep_all_negatives=True),
    )

    empties = [p for p in (out / "labels" / "train").iterdir() if p.read_text().strip() == ""]
    assert empties  # 네거티브 타일도 라벨 파일이 존재하되 내용이 없다
