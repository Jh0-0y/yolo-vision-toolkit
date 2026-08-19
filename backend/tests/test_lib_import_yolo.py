"""외부 YOLO zip 들여오기 — 클래스 병합과 id 재매김.

받는 zip 은 저쪽 번호 체계를 쓴다. 우리 쪽 클래스가 이미 있으면 이름으로 맞추고,
없으면 뒤에 붙인 뒤 라벨의 id 를 그 매핑으로 다시 쓴다. 이게 어긋나면 라벨이
조용히 엉뚱한 클래스를 가리키게 되므로 여기서 못 박는다.
"""

import json
import zipfile
from pathlib import Path

import pytest

from lib.labels.import_yolo import (
    YoloImportError,
    _extract_all,
    import_zip,
    read_yaml_names,
    remap_label_text,
    split_of,
    zip_entry_name,
)


def _make_zip(tmp_path, names, images, *, names_as_list=True, labels=None, split="train"):
    """이미지 `images`(stem 목록)와 data.yaml 을 담은 zip 을 만든다.

    `split=None` 이면 폴더를 나누지 않은 평탄한 zip 을 만든다.
    """
    src = tmp_path / "src"
    img_dir = (src / "images" / split) if split else (src / "images")
    lbl_dir = (src / "labels" / split) if split else (src / "labels")
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for stem in images:
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        text = (labels or {}).get(stem)
        if text is not None:
            (lbl_dir / f"{stem}.txt").write_text(text)
    spec = names if names_as_list else {i: n for i, n in enumerate(names)}
    (src / "data.yaml").write_text(json.dumps({"train": "images/train", "names": spec}))

    zip_path = tmp_path / "ds.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    return zip_path


def test_imports_images_and_labels(tmp_path):
    zip_path = _make_zip(
        tmp_path, ["ball"], ["a", "b"], labels={"a": "0 0.5 0.5 0.1 0.1\n"}
    )
    dest = tmp_path / "ds"

    result = import_zip(zip_path, dest)

    assert result["images"] == 2
    assert result["labeled"] == 1
    assert result["classes"] == 1
    assert sorted(result["stems"]) == ["a", "b"]
    assert (dest / "raw" / "a.jpg").exists()
    assert (dest / "labels" / "a.txt").exists()
    assert not (dest / "labels" / "b.txt").exists()  # 라벨 없는 이미지는 라벨도 없다


def test_classes_land_in_the_dataset_registry(tmp_path):
    zip_path = _make_zip(tmp_path, ["hoop", "net"], ["a"])
    dest = tmp_path / "ds"

    import_zip(zip_path, dest)

    classes = json.loads((dest / "classes.json").read_text())["classes"]
    assert [c["name"] for c in classes] == ["hoop", "net"]


def test_label_ids_are_remapped_onto_existing_classes(tmp_path):
    """이미 `ball` 이 0 인 데이터셋에 `net,ball` zip 이 오면 ball 은 0 으로 가야 한다."""
    dest = tmp_path / "ds"
    dest.mkdir()
    (dest / "classes.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "ball", "sources": []}]})
    )
    # zip 쪽에서는 net=0, ball=1
    zip_path = _make_zip(
        tmp_path, ["net", "ball"], ["a"], labels={"a": "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n"}
    )

    import_zip(zip_path, dest)

    classes = json.loads((dest / "classes.json").read_text())["classes"]
    assert [c["name"] for c in classes] == ["ball", "net"]  # 기존 id 는 그대로
    # net(저쪽 0) → 우리 1, ball(저쪽 1) → 우리 0
    assert (dest / "labels" / "a.txt").read_text().splitlines() == [
        "1 0.1 0.1 0.1 0.1",
        "0 0.2 0.2 0.1 0.1",
    ]


def test_names_can_be_a_dict(tmp_path):
    zip_path = _make_zip(tmp_path, ["ball"], ["a"], names_as_list=False)
    dest = tmp_path / "ds"

    import_zip(zip_path, dest)

    assert json.loads((dest / "classes.json").read_text())["classes"][0]["name"] == "ball"


def test_a_zip_without_data_yaml_is_rejected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"\xff\xd8\xff")
    zip_path = tmp_path / "ds.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src / "a.jpg", "a.jpg")

    with pytest.raises(YoloImportError, match="data.yaml"):
        import_zip(zip_path, tmp_path / "ds")


def test_a_corrupted_zip_is_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_text("not a zip")

    with pytest.raises(YoloImportError, match="Corrupted"):
        import_zip(bad, tmp_path / "ds")


def test_importing_twice_keeps_both_as_numbered_copies(tmp_path):
    """이름이 같아도 **덮어쓰지 않는다.**

    같은 이름이 다른 데이터일 수 있어서, 지울지는 사람이 보고 정한다. 실수로 같은
    zip 을 두 번 넣으면 두 벌이 되지만 `(2)` 로 보이므로 찾아서 지울 수 있다 —
    조용히 사라지는 것보다 낫다.
    """
    zip_path = _make_zip(tmp_path, ["ball"], ["a", "b"], labels={"a": "0 0.5 0.5 0.1 0.1\n"})
    dest = tmp_path / "ds"

    import_zip(zip_path, dest)
    result = import_zip(zip_path, dest)

    assert {p.name for p in (dest / "raw").iterdir()} == {
        "a.jpg", "b.jpg", "a (2).jpg", "b (2).jpg",
    }
    # 두 번째 것도 자기 라벨을 갖는다 — 첫 번째 라벨을 가리키지 않는다
    assert (dest / "labels" / "a (2).txt").exists()
    assert sorted(result["stems"]) == ["a (2)", "b (2)"]


def test_remap_drops_malformed_lines(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.1 0.1 0.1 0.1\ngarbage\n2 0.2\nx 0.1 0.1 0.1 0.1\n")

    assert remap_label_text(path, {0: 5}) == "5 0.1 0.1 0.1 0.1\n"


# ---------- 폴더에서 읽는 분할 ----------
#
# zip 의 train/val/test 는 **읽기만** 한다. 실제 배정으로 쓸지는 호출자가 정한다
# (사용자가 "이미 검증된 데이터"라고 했을 때만).


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("images/train/a.jpg", "train"),
        ("images/val/a.jpg", "val"),
        ("images/valid/a.jpg", "val"),  # Roboflow 는 valid 를 쓴다
        ("images/test/a.jpg", "test"),
        ("train/images/a.jpg", "train"),  # 반대 배치도 있다
        ("a.jpg", None),
        ("images/a.jpg", None),
    ],
)
def test_split_is_read_from_any_folder_level(rel, expected):
    assert split_of(Path(rel)) == expected


def test_import_reports_the_split_each_image_came_from(tmp_path):
    dest = tmp_path / "ds"
    for split in ("train", "val", "test"):
        zip_path = _make_zip(tmp_path / split, ["ball"], [f"{split}_a"], split=split)
        import_zip(zip_path, dest)

    result = import_zip(_make_zip(tmp_path / "again", ["ball"], ["extra"], split="val"), dest)
    assert result["splits"] == {"extra": "val"}


def test_a_flat_zip_reports_no_splits(tmp_path):
    zip_path = _make_zip(tmp_path, ["ball"], ["a", "b"], split=None)

    result = import_zip(zip_path, tmp_path / "ds")

    assert result["splits"] == {}
    assert sorted(result["stems"]) == ["a", "b"]


def test_read_yaml_names_finds_a_nested_yaml(tmp_path):
    """zip 이 폴더 하나를 감싸고 있는 형태가 흔하다."""
    nested = tmp_path / "wrapper"
    nested.mkdir()
    (nested / "data.yaml").write_text(json.dumps({"names": ["a", "b"]}))

    assert read_yaml_names(tmp_path) == {0: "a", 1: "b"}


# ---------- macOS zip 의 파일명 ----------
#
# zip 규격은 파일명을 CP437 로 쓰되 UTF-8 이면 플래그를 세우라고 한다. macOS 는
# UTF-8 바이트를 넣고도 플래그를 안 세워서, 규격대로 읽으면 한글이 뒤집힌다.
# 실제로 `스크린샷 …png` 이 `ßäëßà│…` 로 들어와 데이터셋에 쌓인 적이 있다.


def _info(filename: str, *, utf8_flag: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename="x")
    info.filename = filename  # 생성자를 우회해 플래그를 우리가 정한다
    info.flag_bits = 0x800 if utf8_flag else 0
    return info


def test_unflagged_utf8_name_is_recovered():
    """macOS zip — 플래그가 없어 CP437 로 뒤집힌 이름을 되돌린다."""
    mojibake = "스크린샷 2026-04-16.png".encode().decode("cp437")

    assert zip_entry_name(_info(mojibake, utf8_flag=False)) == "스크린샷 2026-04-16.png"


def test_flagged_name_is_left_alone():
    """플래그가 있으면 `zipfile` 이 이미 UTF-8 로 읽었다 — 건드리면 오히려 깨진다."""
    assert zip_entry_name(_info("스크린샷.png", utf8_flag=True)) == "스크린샷.png"


def test_a_real_cp437_name_survives():
    """옛날 zip 이 진짜 CP437 로 쓴 이름은 UTF-8 해독이 실패한다 — 그대로 둔다."""
    assert zip_entry_name(_info("café.png", utf8_flag=False)) == "café.png"


def test_zip_slip_is_blocked(tmp_path):
    """직접 푸는 만큼 경로 탈출도 직접 막는다 — `extractall` 이 해 주던 일이다."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
        zf.writestr("ok.txt", "yes")

    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        _extract_all(zf, dest)

    assert (dest / "ok.txt").exists()
    assert not (tmp_path / "escaped.txt").exists()
