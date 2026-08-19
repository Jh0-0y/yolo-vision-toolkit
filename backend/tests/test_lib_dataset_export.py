"""데이터셋 → YOLO zip.

지키는 것 셋:
  **검수완료만 나간다** — 사람이 확인하지 않은 라벨을 학습에 흘리지 않는다
  **이미지는 하드링크** — 몇 번을 내보내도 디스크는 한 벌이다
  **클래스를 고르면 id 를 다시 매긴다** — 라벨이 엉뚱한 클래스를 가리키면 조용히 망한다
"""

import json
import zipfile

import pytest

from lib.labels.dataset_export import ExportError, build, select_stems


@pytest.fixture
def dataset(tmp_path):
    """검수완료 4장(train2·val1·test1) + 미검수 1장을 가진 데이터셋."""
    d = tmp_path / "ds"
    (d / "raw").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    for stem in ("a", "b", "c", "t", "unrev"):
        (d / "raw" / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        (d / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    (d / "classes.json").write_text(
        json.dumps({"classes": [
            {"id": 0, "name": "ball", "sources": []},
            {"id": 1, "name": "player", "sources": []},
        ]})
    )
    return d


REVIEWED = {"a", "b", "c", "t"}
SPLITS = {"a": "train", "b": "train", "c": "val", "t": "test"}


def _names_in(zip_path, prefix):
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/"))


# ---------- 무엇이 나가나 ----------


def test_train_export_bundles_train_and_val(dataset, tmp_path):
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="train",
              reviewed=REVIEWED, splits=SPLITS)

    assert (r["train"], r["val"], r["test"]) == (2, 1, 0)
    assert _names_in(r["zip"], "images/train") == ["images/train/a.jpg", "images/train/b.jpg"]


def test_test_export_takes_only_test(dataset, tmp_path):
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="test",
              reviewed=REVIEWED, splits=SPLITS)

    assert r["count"] == 1
    assert _names_in(r["zip"], "images/test") == ["images/test/t.jpg"]


def test_all_export_ignores_the_split(dataset, tmp_path):
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS)

    assert r["count"] == 4  # 검수완료 전부, 분할 무관
    assert len(_names_in(r["zip"], "images/train")) == 4


def test_unreviewed_never_leaves(dataset, tmp_path):
    """사람이 확인하지 않은 라벨은 학습에도 평가에도 안 나간다."""
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS)

    with zipfile.ZipFile(r["zip"]) as zf:
        assert not [n for n in zf.namelist() if "unrev" in n]


def test_selection_skips_unassigned_for_train(dataset):
    assert select_stems("train", {"a"}, {}) == {}


def test_nothing_to_export_is_an_error(dataset, tmp_path):
    with pytest.raises(ExportError, match="Nothing to export"):
        build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="test",
              reviewed=set(), splits={})


def test_an_unknown_kind_is_rejected(dataset, tmp_path):
    with pytest.raises(ExportError, match="Unknown export kind"):
        build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="everything",
              reviewed=REVIEWED, splits=SPLITS)


# ---------- 하드링크 ----------


def test_images_are_hardlinked_not_copied(dataset, tmp_path):
    """몇 번을 내보내도 이미지 바이트는 한 벌이다."""
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS)

    assert r["linked"] == 4 and r["copied"] == 0


# ---------- 클래스 ----------


def test_all_classes_go_out_by_default(dataset, tmp_path):
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS)

    with zipfile.ZipFile(r["zip"]) as zf:
        yaml_text = zf.read("data.yaml").decode()
    assert "ball" in yaml_text and "player" in yaml_text
    assert r["classes"] == 2


def test_choosing_classes_filters_boxes_and_renumbers(dataset, tmp_path):
    """`player`(id 1)만 담으면 그 박스만 남고 id 는 0 이 된다."""
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS, class_ids=[1])

    with zipfile.ZipFile(r["zip"]) as zf:
        label = zf.read("labels/train/a.txt").decode().strip().splitlines()
        yaml_text = zf.read("data.yaml").decode()
    assert len(label) == 1 and label[0].startswith("0 ")
    assert "player" in yaml_text and "ball" not in yaml_text


def test_an_image_with_no_kept_boxes_still_gets_an_empty_label(dataset, tmp_path):
    """빈 라벨은 '안 그렸다'가 아니라 '여기엔 없다'는 학습 신호다."""
    (dataset / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # ball 만

    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="all",
              reviewed=REVIEWED, splits=SPLITS, class_ids=[1])

    with zipfile.ZipFile(r["zip"]) as zf:
        assert zf.read("labels/train/a.txt").decode().strip() == ""
        assert "images/train/a.jpg" in zf.namelist()


# ---------- 그 밖 ----------


def test_a_missing_image_is_skipped(dataset, tmp_path):
    (dataset / "raw" / "b.jpg").unlink()

    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="train",
              reviewed=REVIEWED, splits=SPLITS)

    assert r["train"] == 1


def test_val_falls_back_to_train_when_empty(dataset, tmp_path):
    """val 이 비면 ultralytics 가 학습을 못 돈다 — train 을 가리켜 둔다."""
    r = build(dataset_dir=dataset, out_dir=tmp_path / "out", kind="test",
              reviewed=REVIEWED, splits=SPLITS)

    with zipfile.ZipFile(r["zip"]) as zf:
        assert "val: images/train" in zf.read("data.yaml").decode()


def test_the_unzipped_tree_is_removed(dataset, tmp_path):
    """zip 을 만들고 나면 펼친 트리는 쓸모가 없다."""
    out = tmp_path / "out"
    build(dataset_dir=dataset, out_dir=out, kind="all", reviewed=REVIEWED, splits=SPLITS)

    assert not out.exists()
