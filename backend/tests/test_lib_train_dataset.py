"""lib/train/dataset — 업로드 zip 임포트 계약.

업로드되는 zip 은 출처가 제각각이라(절대경로 · `path:` 루트 · data.yml · 한 단계
아래 구조) 정규화가 실제로 도는지가 중요하다. 라우터 안에 있던 동안은 HTTP 없이
못 돌려 테스트가 없었다.
"""

import json
import zipfile

import pytest
import yaml

from lib.train import dataset


def _make_dataset(root, *, yaml_name="data.yaml", nest=None, data=None, images=("a", "b")):
    """train/val 이미지가 있는 최소 YOLO 데이터셋을 만들고 data.yaml 경로를 돌려준다."""
    base = root / nest if nest else root
    for split in ("train", "val"):
        d = base / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        for name in images:
            (d / f"{name}.jpg").write_bytes(b"x")
    payload = data if data is not None else {
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "ball", 1: "player"},
    }
    path = base / yaml_name
    path.write_text(yaml.safe_dump(payload))
    return path


def _zip_of(src, dst_zip):
    with zipfile.ZipFile(dst_zip, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    return dst_zip


# ---------- count_images ----------


def test_count_images_counts_a_directory_recursively(tmp_path):
    d = tmp_path / "imgs"
    (d / "sub").mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"x")
    (d / "sub" / "b.PNG").write_bytes(b"x")  # 대문자 확장자도 센다
    (d / "notes.txt").write_text("x")  # 이미지가 아니면 안 센다

    assert dataset.count_images(d) == 2


def test_count_images_reads_a_txt_list_file(tmp_path):
    """train 이 디렉터리가 아니라 목록 파일을 가리키는 데이터셋이 있다."""
    listing = tmp_path / "train.txt"
    listing.write_text("a.jpg\n\nb.jpg\n")

    assert dataset.count_images(listing) == 2


def test_count_images_missing_path_is_zero(tmp_path):
    assert dataset.count_images(tmp_path / "nope") == 0


# ---------- normalize_data_yaml ----------


def test_normalize_rewrites_relative_paths_and_counts(tmp_path):
    yaml_path = _make_dataset(tmp_path)

    counts = dataset.normalize_data_yaml(yaml_path)

    assert counts == {"train": 2, "val": 2, "classes": 2}
    written = yaml.safe_load(yaml_path.read_text())
    assert written["train"] == "images/train"
    assert written["val"] == "images/val"


def test_normalize_drops_the_path_root_key(tmp_path):
    """`path:` 는 다른 기계 기준이라 남겨두면 학습이 엉뚱한 곳을 본다."""
    yaml_path = _make_dataset(
        tmp_path,
        nest="ds",
        data={"path": "/somewhere/else", "train": "images/train", "val": "images/val", "nc": 1,
              "names": {0: "ball"}},
    )

    dataset.normalize_data_yaml(yaml_path)

    written = yaml.safe_load(yaml_path.read_text())
    assert "path" not in written
    assert written["train"] == "images/train"


def test_normalize_resolves_absolute_paths_from_another_machine(tmp_path):
    """다른 기계의 절대경로는 꼬리 두 조각을 압축 해제 루트에 맞춰 되찾는다."""
    yaml_path = _make_dataset(
        tmp_path,
        data={"train": "/mnt/nas/proj/images/train", "val": "/mnt/nas/proj/images/val",
              "nc": 1, "names": {0: "ball"}},
    )

    counts = dataset.normalize_data_yaml(yaml_path)

    assert yaml.safe_load(yaml_path.read_text())["train"] == "images/train"
    assert counts["train"] == 2


def test_normalize_falls_back_to_train_when_val_is_missing(tmp_path):
    yaml_path = _make_dataset(
        tmp_path, data={"train": "images/train", "nc": 1, "names": {0: "ball"}}
    )

    counts = dataset.normalize_data_yaml(yaml_path)

    assert yaml.safe_load(yaml_path.read_text())["val"] == "images/train"
    assert counts["val"] == counts["train"]


def test_normalize_infers_class_count_from_names(tmp_path):
    yaml_path = _make_dataset(
        tmp_path,
        data={"train": "images/train", "val": "images/val", "names": {0: "a", 1: "b", 2: "c"}},
    )

    assert dataset.normalize_data_yaml(yaml_path)["classes"] == 3


def test_normalize_raises_when_train_cannot_be_found(tmp_path):
    yaml_path = _make_dataset(
        tmp_path, data={"train": "does/not/exist", "nc": 1, "names": {0: "ball"}}
    )

    with pytest.raises(dataset.DatasetError, match="train path"):
        dataset.normalize_data_yaml(yaml_path)


# ---------- extract_zip ----------


def _extract(tmp_path, zip_path, **kw):
    from datetime import datetime, timezone

    opts = {
        "dataset_id": "u_test",
        "name": "my-set",
        "auto_delete": False,
        "now": datetime(2026, 8, 12, tzinfo=timezone.utc),
    }
    opts.update(kw)
    return dataset.extract_zip(zip_path, tmp_path / "dest", **opts)


def test_extract_zip_writes_metadata(tmp_path):
    src = tmp_path / "src"
    _make_dataset(src)
    z = _zip_of(src, tmp_path / "ds.zip")

    meta = _extract(tmp_path, z)

    assert meta["id"] == "u_test"
    assert meta["name"] == "my-set"
    assert meta["yaml_dir"] == "."
    assert (meta["train"], meta["val"], meta["classes"]) == (2, 2, 2)
    assert meta["auto_delete"] is False
    assert meta["created_at"].startswith("2026-08-12")
    on_disk = json.loads((tmp_path / "dest" / "dataset.json").read_text())
    assert on_disk == meta


def test_extract_zip_finds_data_yaml_one_level_down(tmp_path):
    """zip 이 폴더 하나를 감싸고 있는 흔한 형태."""
    src = tmp_path / "src"
    _make_dataset(src, nest="my-dataset")
    z = _zip_of(src, tmp_path / "ds.zip")

    meta = _extract(tmp_path, z)

    assert meta["yaml_dir"] == "my-dataset"
    assert meta["train"] == 2


def test_extract_zip_renames_data_yml_to_data_yaml(tmp_path):
    """학습 러너는 'data.yaml' 이라는 이름만 찾는다."""
    src = tmp_path / "src"
    _make_dataset(src, yaml_name="data.yml")
    z = _zip_of(src, tmp_path / "ds.zip")

    _extract(tmp_path, z)

    assert (tmp_path / "dest" / "data.yaml").exists()
    assert not (tmp_path / "dest" / "data.yml").exists()


def test_extract_zip_rejects_a_dataset_without_data_yaml(tmp_path):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.jpg").write_bytes(b"x")
    z = _zip_of(src, tmp_path / "ds.zip")

    with pytest.raises(dataset.DatasetError, match="data.yaml not found"):
        _extract(tmp_path, z)


def test_extract_zip_rejects_a_dataset_with_no_train_images(tmp_path):
    """경로는 멀쩡한데 안에 이미지가 하나도 없는 zip."""
    src = tmp_path / "src"
    _make_dataset(src, images=())
    # zip 은 빈 디렉터리를 담지 않는다 — 경로가 살아남도록 이미지 아닌 파일을 둔다
    (src / "images" / "train" / "README.md").write_text("empty on purpose")
    (src / "images" / "val" / "README.md").write_text("empty on purpose")
    z = _zip_of(src, tmp_path / "ds.zip")

    with pytest.raises(dataset.DatasetError, match="No train images"):
        _extract(tmp_path, z)


def test_extract_zip_rejects_a_corrupted_zip(tmp_path):
    z = tmp_path / "broken.zip"
    z.write_bytes(b"not a zip at all")

    with pytest.raises(dataset.DatasetError, match="Corrupted zip"):
        _extract(tmp_path, z)


def test_extract_zip_removes_the_destination_on_failure(tmp_path):
    """반쯤 풀린 데이터셋이 목록에 뜨면 안 된다."""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.jpg").write_bytes(b"x")
    z = _zip_of(src, tmp_path / "ds.zip")

    with pytest.raises(dataset.DatasetError):
        _extract(tmp_path, z)

    assert not (tmp_path / "dest").exists()
