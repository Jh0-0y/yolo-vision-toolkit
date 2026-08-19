"""End-to-end pipeline test with a stubbed ultralytics.YOLO (no weights, no GPU)."""

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from lib.detect.labeling import DetectorSpec, LabelJobConfig, run_labeling


def _spec(path, **kw):
    """검출기 엔트리 하나. 기본은 풀 프레임 — 대부분의 테스트가 방식을 안 따진다."""
    return DetectorSpec(model_path=path, model_id=path.stem, **kw)



class _Arr:
    """Mimics a torch tensor just enough: .cpu().numpy()."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class _Boxes:
    def __init__(self, dets):
        self.xyxyn = _Arr([d[1] for d in dets])
        self.cls = _Arr([d[0] for d in dets])
        self.conf = _Arr([d[2] for d in dets])


class _Result:
    def __init__(self, dets, shape=(100, 200)):  # (h, w)
        self.orig_shape = shape
        self.boxes = _Boxes(dets) if dets else None


def _stem(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


# per model: image stem -> list of (local_cls, xyxyn, conf)
FAKE_PREDICTIONS = {
    "model_a": {  # classes: 0=person, 1=car
        "img_agree": [(1, (0.10, 0.10, 0.30, 0.30), 0.90)],
        "img_person": [(0, (0.40, 0.40, 0.60, 0.60), 0.55)],
        "img_empty": [],
    },
    "model_b": {  # classes: 0=car, 1=dog
        "img_agree": [(0, (0.11, 0.11, 0.31, 0.31), 0.85)],
        "img_person": [],
        "img_empty": [],
    },
}

MODEL_NAMES = {
    "model_a": {0: "person", 1: "car"},
    "model_b": {0: "car", 1: "dog"},
}


class FakeYOLO:
    def __init__(self, path):
        self._id = _stem(path)
        self.names = MODEL_NAMES[self._id]

    def predict(self, paths, **kwargs):
        preds = FAKE_PREDICTIONS[self._id]
        return [_Result(preds.get(_stem(p), [])) for p in paths]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_pipeline_end_to_end(tmp_path, fake_ultralytics):
    images = tmp_path / "raw"
    images.mkdir()
    for stem in ("img_agree", "img_person", "img_empty"):
        Image.new("RGB", (200, 100)).save(images / f"{stem}.jpg")

    out = tmp_path / "out"
    events = []
    result = run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "model_a.pt"), _spec(tmp_path / "model_b.pt")],
            images_dir=images,
            out_dir=out,
            device="cpu",
        ),
        progress=events.append,
    )

    # every image gets a label file — no routing
    assert result.total == 3
    assert result.labeled == 3
    assert sorted(result.stems) == ["img_agree", "img_empty", "img_person"]

    # class union: person, car, dog
    classes = json.loads((out / "classes.json").read_text())
    assert [c["name"] for c in classes["classes"]] == ["person", "car", "dog"]

    # fused label uses the global class id for "car" (=1)
    label = (out / "labels" / "img_agree.txt").read_text().strip()
    assert label.startswith("1 ")

    # single-model person detection lands as global id 0
    person = (out / "labels" / "img_person.txt").read_text().strip()
    assert person.startswith("0 ")

    # empty image → deliberate negative (empty file)
    assert (out / "labels" / "img_empty.txt").read_text() == ""

    # progress stream terminates with done
    assert events[-1]["phase"] == "done"


def test_registry_seeded_from_existing_classes(tmp_path, fake_ultralytics):
    images = tmp_path / "raw"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "img_agree.jpg")

    out = tmp_path / "out"
    out.mkdir()
    # a previous job registered "dog" as id 0 — ids must never renumber
    (out / "classes.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "dog", "sources": ["old"]}]})
    )

    run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "model_a.pt"), _spec(tmp_path / "model_b.pt")],
            images_dir=images,
            out_dir=out,
            device="cpu",
        )
    )

    classes = json.loads((out / "classes.json").read_text())
    assert [c["name"] for c in classes["classes"]] == ["dog", "person", "car"]
    # "car" is now global id 2
    label = (out / "labels" / "img_agree.txt").read_text().strip()
    assert label.startswith("2 ")


class _BallYOLO:
    """One model owning a single class 'ball' with five descending-conf boxes."""

    def __init__(self, path):
        self.names = {0: "ball"}

    def predict(self, paths, **kwargs):
        dets = [
            (0, (0.10, 0.10, 0.20, 0.20), 0.90),
            (0, (0.30, 0.30, 0.40, 0.40), 0.70),
            (0, (0.50, 0.50, 0.60, 0.60), 0.50),
            (0, (0.70, 0.70, 0.80, 0.80), 0.30),
            (0, (0.05, 0.05, 0.09, 0.09), 0.10),
        ]
        return [_Result(dets) for _ in paths]


@pytest.fixture
def fake_ball(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _BallYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def _read_label(out, stem):
    from lib.labels.io import read_box_meta

    label = out / "labels" / f"{stem}.txt"
    lines = [ln for ln in label.read_text().splitlines() if ln.strip()]
    return lines, read_box_meta(label)


def test_conf_is_a_hard_filter(tmp_path, fake_ball):
    images = tmp_path / "raw"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "img.jpg")
    out = tmp_path / "out"

    run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "m.pt")],
            images_dir=images,
            out_dir=out,
            conf=0.4,
            device="cpu",
        )
    )

    lines, meta = _read_label(out, "img")
    # only the three boxes with conf >= 0.4 survive
    assert len(lines) == 3
    assert all(m["score"] >= 0.4 for m in meta)
    # review status is never set by auto-labeling
    assert all(m["status"] is None for m in meta)


def test_max_boxes_per_class_keeps_top_n(tmp_path, fake_ball):
    images = tmp_path / "raw"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "img.jpg")
    out = tmp_path / "out"

    run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "m.pt")],
            images_dir=images,
            out_dir=out,
            conf=0.05,  # keep everything, then cap
            max_boxes_per_class={"ball": 2},
            device="cpu",
        )
    )

    lines, meta = _read_label(out, "img")
    # capped to the 2 highest-confidence ball boxes
    assert len(lines) == 2
    assert sorted((m["score"] for m in meta), reverse=True) == [0.9, 0.7]


def test_max_boxes_matches_class_name_case_insensitively(tmp_path, fake_ball):
    images = tmp_path / "raw"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "img.jpg")
    out = tmp_path / "out"

    run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "m.pt")],
            images_dir=images,
            out_dir=out,
            conf=0.05,
            max_boxes_per_class={"BALL": 1},  # normalized to match "ball"
            device="cpu",
        )
    )

    lines, _ = _read_label(out, "img")
    assert len(lines) == 1


def test_image_names_filter(tmp_path, fake_ultralytics):
    images = tmp_path / "raw"
    images.mkdir()
    for stem in ("img_agree", "img_empty"):
        Image.new("RGB", (200, 100)).save(images / f"{stem}.jpg")

    out = tmp_path / "out"
    result = run_labeling(
        LabelJobConfig(
            detectors=[_spec(tmp_path / "model_a.pt")],
            images_dir=images,
            out_dir=out,
            image_names=["img_agree.jpg"],
            device="cpu",
        )
    )
    assert result.total == 1
    assert (out / "labels" / "img_agree.txt").exists()
    assert not (out / "labels" / "img_empty.txt").exists()
