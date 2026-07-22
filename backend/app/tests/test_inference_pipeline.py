"""End-to-end pipeline test with a stubbed ultralytics.YOLO (no weights, no GPU)."""

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from app.core.decision import DecisionConfig
from app.core.inference import LabelJobConfig, run_labeling


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


# per model: image stem -> list of (local_cls, xyxyn, conf)
FAKE_PREDICTIONS = {
    "model_a": {  # classes: 0=person, 1=car
        "img_agree": [(1, (0.10, 0.10, 0.30, 0.30), 0.90)],
        "img_lowconf": [(0, (0.40, 0.40, 0.60, 0.60), 0.20)],
        "img_empty": [],
    },
    "model_b": {  # classes: 0=car, 1=dog
        "img_agree": [(0, (0.11, 0.11, 0.31, 0.31), 0.85)],
        "img_lowconf": [],
        "img_empty": [],
    },
}

MODEL_NAMES = {
    "model_a": {0: "person", 1: "car"},
    "model_b": {0: "car", 1: "dog"},
}


class FakeYOLO:
    def __init__(self, path):
        self._id = path.rsplit("/", 1)[-1].removesuffix(".pt")
        self.names = MODEL_NAMES[self._id]

    def predict(self, paths, **kwargs):
        preds = FAKE_PREDICTIONS[self._id]
        out = []
        for p in paths:
            stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out.append(_Result(preds.get(stem, [])))
        return out


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_pipeline_end_to_end(tmp_path, fake_ultralytics):
    images = tmp_path / "raw"
    images.mkdir()
    for stem in ("img_agree", "img_lowconf", "img_empty"):
        Image.new("RGB", (200, 100)).save(images / f"{stem}.jpg")

    out = tmp_path / "out"
    events = []
    result = run_labeling(
        LabelJobConfig(
            model_paths=[tmp_path / "model_a.pt", tmp_path / "model_b.pt"],
            images_dir=images,
            out_dir=out,
            decision=DecisionConfig(conf_min=0.10, conf_confirm=0.60),
            device="cpu",
        ),
        progress=events.append,
    )

    assert result.total == 3
    assert result.confirmed == 1  # both models agree on "car" with high conf
    assert result.review == 2  # low-conf person + empty image

    # class union: person, car, dog
    classes = json.loads((out / "classes.json").read_text())
    assert [c["name"] for c in classes["classes"]] == ["person", "car", "dog"]

    # confirmed label uses the global class id for "car" (=1)
    label = (out / "confirmed" / "labels" / "img_agree.txt").read_text().strip()
    assert label.startswith("1 ")
    assert (out / "confirmed" / "images" / "img_agree.jpg").exists()

    # review items carry flag reasons and source scores
    lowconf = json.loads((out / "review" / "img_lowconf.json").read_text())
    flagged = [b for b in lowconf["boxes"] if b["status"] == "needs_review"]
    assert flagged and flagged[0]["reason"] == "low_conf"
    assert flagged[0]["sources"][0]["model"] == "model_a"

    empty = json.loads((out / "review" / "img_empty.json").read_text())
    assert empty["boxes"] == []

    # progress stream terminates with done
    assert events[-1]["phase"] == "done"


def test_pipeline_negative_policy(tmp_path, fake_ultralytics):
    images = tmp_path / "raw"
    images.mkdir()
    Image.new("RGB", (200, 100)).save(images / "img_empty.jpg")

    out = tmp_path / "out"
    result = run_labeling(
        LabelJobConfig(
            model_paths=[tmp_path / "model_a.pt", tmp_path / "model_b.pt"],
            images_dir=images,
            out_dir=out,
            decision=DecisionConfig(empty_policy="negative"),
            device="cpu",
        )
    )
    assert result.negative == 1
    assert (out / "confirmed" / "labels" / "img_empty.txt").read_text() == ""
