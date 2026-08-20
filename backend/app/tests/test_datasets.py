"""데이터셋의 파일 규약 — 자리와 **수치 파생**.

데이터셋은 자기 것을 전부 갖고 다른 데이터셋과 아무것도 공유하지 않는다. 수치는
저장하지 않고 파일에서 그때 센다 — 저장해 두면 반드시 실제와 어긋나는 순간이 온다.
DB 가 끼지 않는 부분이라 파일만 깔아 놓고 검증할 수 있다.
"""

import pytest

from app.core.config import settings
from app.services import datasets


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _images(project_id: str, dataset_id: str, *stems: str) -> None:
    for stem in stems:
        (datasets.raw_dir(project_id, dataset_id) / f"{stem}.jpg").write_bytes(b"x")


# ---------- 자리 ----------


def test_create_lays_out_the_dataset(data_dir):
    ds = datasets.create("p1", "공만")

    assert datasets.raw_dir("p1", ds).is_dir()
    assert datasets.labels_dir("p1", ds).is_dir()
    assert datasets.read_meta("p1", ds)["name"] == "공만"


def test_datasets_of_other_projects_are_not_mixed_in(data_dir):
    mine = datasets.create("p1", "a")
    datasets.create("p2", "b")

    assert [d["id"] for d in datasets.list_datasets("p1")] == [mine]


def test_list_of_an_unknown_project_is_empty(data_dir):
    assert datasets.list_datasets("nope") == []


def test_delete_takes_everything_with_it(data_dir):
    """데이터셋 밖으로 공유하는 것이 없으니 통째로 지운다."""
    ds = datasets.create("p1", "a")
    _images("p1", ds, "img1")

    datasets.delete("p1", ds)

    assert not datasets.dataset_dir("p1", ds).exists()
    assert datasets.list_datasets("p1") == []


def test_rename_keeps_the_rest_of_the_meta(data_dir):
    ds = datasets.create("p1", "옛 이름")
    created = datasets.read_meta("p1", ds)["created_at"]

    datasets.rename("p1", ds, "새 이름")

    meta = datasets.read_meta("p1", ds)
    assert (meta["name"], meta["created_at"]) == ("새 이름", created)


# ---------- 수치 ----------


def test_everything_starts_unreviewed(data_dir):
    """가져온 이미지는 전부 미검수로 들어온다."""
    ds = datasets.create("p1", "a")
    _images("p1", ds, "img1", "img2", "img3")

    c = datasets.counts("p1", ds)
    assert (c["images"], c["unreviewed"], c["reviewed"]) == (3, 3, 0)


def test_reviewing_moves_an_image_across(data_dir):
    ds = datasets.create("p1", "a")
    _images("p1", ds, "img1", "img2")
    datasets.write_reviewed("p1", ds, {"img1"})

    c = datasets.counts("p1", ds)
    assert (c["unreviewed"], c["reviewed"], c["unassigned"]) == (1, 1, 1)


def test_split_counts_come_from_splits_json(data_dir):
    ds = datasets.create("p1", "a")
    _images("p1", ds, "a", "b", "c", "d")
    datasets.write_reviewed("p1", ds, {"a", "b", "c", "d"})
    datasets.write_splits("p1", ds, {"a": "train", "b": "train", "c": "val", "d": "test"})

    c = datasets.counts("p1", ds)
    assert (c["train"], c["val"], c["test"], c["unassigned"]) == (2, 1, 1, 0)


def test_an_unreviewed_image_never_counts_as_split(data_dir):
    """검수를 취소하면 분할에서도 빠진 것으로 보인다 — splits.json 을 고치러 다니지 않는다."""
    ds = datasets.create("p1", "a")
    _images("p1", ds, "a")
    datasets.write_splits("p1", ds, {"a": "train"})

    c = datasets.counts("p1", ds)
    assert (c["reviewed"], c["train"], c["unreviewed"]) == (0, 0, 1)


def test_a_split_entry_without_an_image_is_ignored(data_dir):
    """이미지를 지우면 splits.json 에 남은 줄이 수치를 부풀리면 안 된다."""
    ds = datasets.create("p1", "a")
    datasets.write_reviewed("p1", ds, {"gone"})
    datasets.write_splits("p1", ds, {"gone": "train"})

    c = datasets.counts("p1", ds)
    assert (c["images"], c["reviewed"], c["train"]) == (0, 0, 0)


def test_unknown_split_values_are_dropped(data_dir):
    ds = datasets.create("p1", "a")
    datasets.write_splits("p1", ds, {"a": "train"})
    (datasets.dataset_dir("p1", ds) / datasets.SPLITS_NAME).write_text(
        '{"a": "train", "b": "holdout"}'
    )

    assert datasets.read_splits("p1", ds) == {"a": "train"}


def test_non_images_in_raw_are_not_counted(data_dir):
    ds = datasets.create("p1", "a")
    _images("p1", ds, "img1")
    (datasets.raw_dir("p1", ds) / "notes.txt").write_text("x")

    assert datasets.counts("p1", ds)["images"] == 1


def test_a_broken_json_does_not_break_the_list(data_dir):
    ds = datasets.create("p1", "a")
    (datasets.dataset_dir("p1", ds) / datasets.REVIEWED_NAME).write_text("{not json")

    assert datasets.counts("p1", ds)["reviewed"] == 0


def test_list_is_newest_first(data_dir):
    first = datasets.create("p1", "a")
    second = datasets.create("p1", "b")
    datasets.write_meta("p1", first, {**datasets.read_meta("p1", first), "created_at": "2020"})
    datasets.write_meta("p1", second, {**datasets.read_meta("p1", second), "created_at": "2030"})

    assert [d["id"] for d in datasets.list_datasets("p1")] == [second, first]


@pytest.mark.parametrize("bad", ["", ".", "../secrets", "a/b", "a\\b"])
def test_invalid_ids_are_rejected(bad):
    assert datasets.valid_id(bad) is False


def test_split_stratifies_by_whether_a_frame_has_labels(data_dir):
    """라벨 없는 프레임도 train/val/test 에 비례해 흩어진다 — 한쪽에 몰리지 않는다.

    안 그러면 val 에 배경이 하나도 없어, 배경 오검출을 학습 내내 못 잰다.
    """
    from app.api.v1.endpoints.dataset_export import stratified_assign
    from lib.labels.io import write_label_file

    ds = datasets.create("p1", "mixed")
    datasets.ensure_dirs("p1", ds)
    labeled, unlabeled = [], []
    for i in range(90):
        stem = f"pos{i:03d}"
        _images("p1", ds, stem)
        write_label_file(datasets.labels_dir("p1", ds) / f"{stem}.txt", [(0, (0.4, 0.4, 0.6, 0.6))])
        labeled.append(stem)
    for i in range(10):
        stem = f"neg{i:03d}"
        _images("p1", ds, stem)
        unlabeled.append(stem)
    datasets.write_reviewed("p1", ds, set(labeled) | set(unlabeled))

    assigned = stratified_assign(
        "p1", ds, sorted(labeled + unlabeled), {}, train=8, val=1, test=1, seed=0,
        reassign_all=False,
    )

    # 라벨 없는 10장이 세 분할에 흩어진다 (한 곳에 10장이 몰리지 않는다)
    per_split = {"train": 0, "val": 0, "test": 0}
    for stem in unlabeled:
        per_split[assigned[stem]] += 1
    assert per_split["train"] == 8
    assert per_split["val"] == 1
    assert per_split["test"] == 1
