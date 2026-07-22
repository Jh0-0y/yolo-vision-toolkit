import pytest
import yaml

from app.core.yolo_io import (
    read_label_file,
    write_data_yaml,
    write_label_file,
    xyxyn_to_yolo,
    yolo_to_xyxyn,
)


def test_coordinate_roundtrip():
    xyxy = (0.1, 0.2, 0.5, 0.8)
    back = yolo_to_xyxyn(xyxyn_to_yolo(xyxy))
    assert back == pytest.approx(xyxy)


def test_label_file_roundtrip(tmp_path):
    path = tmp_path / "labels" / "img.txt"
    boxes = [(0, (0.1, 0.2, 0.5, 0.8)), (3, (0.0, 0.0, 1.0, 1.0))]
    write_label_file(path, boxes)

    loaded = read_label_file(path)
    assert len(loaded) == 2
    assert loaded[0][0] == 0
    assert loaded[0][1] == pytest.approx(boxes[0][1], abs=1e-5)
    assert loaded[1][0] == 3


def test_write_label_clamps_out_of_range(tmp_path):
    path = tmp_path / "img.txt"
    write_label_file(path, [(0, (-0.1, 0.2, 1.3, 0.8))])
    (cls, xyxy), = read_label_file(path)
    assert xyxy[0] == pytest.approx(0.0, abs=1e-5)
    assert xyxy[2] == pytest.approx(1.0, abs=1e-5)


def test_empty_label_file(tmp_path):
    path = tmp_path / "img.txt"
    write_label_file(path, [])
    assert path.read_text() == ""
    assert read_label_file(path) == []


def test_data_yaml_covers_full_id_space(tmp_path):
    path = tmp_path / "data.yaml"
    # sparse names (id 1 missing) must still produce contiguous nc=3
    write_data_yaml(path, {0: "person", 2: "dog"}, train="images/train", val="images/val")
    data = yaml.safe_load(path.read_text())
    assert data["nc"] == 3
    assert data["names"] == {0: "person", 1: "class_1", 2: "dog"}
    assert data["train"] == "images/train"
