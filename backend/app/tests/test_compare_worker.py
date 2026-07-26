"""compare_worker end-to-end with a stubbed ultralytics.YOLO (no weights/GPU).

Verifies per-model scoring and the class-mismatch fix: a prediction whose class
isn't in the project is counted as a false positive, not silently dropped."""

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.domain.labels import write_boxes
from app.workers.compare_worker import run_compare


class _Arr:
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
    def __init__(self, dets, shape=(100, 200)):
        self.orig_shape = shape
        self.boxes = _Boxes(dets) if dets else None


# model stem -> class names; and its detections on any image
MODEL_NAMES = {"m1": {0: "ball"}, "m2": {0: "cat"}}
MODEL_PREDS = {
    "m1": [(0, (0.10, 0.10, 0.30, 0.30), 0.90)],  # "ball" over the GT box → TP
    "m2": [(0, (0.10, 0.10, 0.30, 0.30), 0.90)],  # "cat" (not in project) → FP
}


class FakeYOLO:
    def __init__(self, path):
        self._id = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self.names = MODEL_NAMES[self._id]

    def to(self, device):
        return self

    def predict(self, image, **kwargs):
        return [_Result(MODEL_PREDS[self._id])]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def _seed_project(tmp_path, with_classes=True):
    monkeypatch_dir = tmp_path
    pdir = monkeypatch_dir / "projects" / "p1"
    (pdir / "raw").mkdir(parents=True)
    (pdir / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(pdir / "raw" / "img1.jpg")
    # ground truth: one "ball" (class 0) box
    write_boxes(pdir, "img1", [{"cls": 0, "xyxy_n": [0.10, 0.10, 0.30, 0.30]}])
    if with_classes:
        (pdir / "classes.json").write_text(
            json.dumps({"classes": [{"id": 0, "name": "ball", "sources": []}]})
        )
    return pdir


def test_compare_scores_each_model_and_counts_unknown_class_as_fp(
    tmp_path, fake_ultralytics, monkeypatch
):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _seed_project(tmp_path)
    (settings.jobs_dir / "job1").mkdir(parents=True, exist_ok=True)

    run_compare(
        "job1",
        {
            "project_id": "p1",
            "specs": [("m1", "m1.pt"), ("m2", "m2.pt")],
            "model_names": {"m1": "Model A", "m2": "Model B"},
            "image_names": ["img1.jpg"],
            "conf": 0.4,
            "iou": 0.5,
            "imgsz": 640,
            "device": "cpu",
        },
        str(settings.jobs_dir),
    )

    result = json.loads((settings.jobs_dir / "job1" / "result.json").read_text())
    by_id = {m["model_id"]: m for m in result["per_model"]}

    # m1 predicted "ball" over the GT → true positive, perfect scores
    assert by_id["m1"]["overall"]["tp"] == 1
    assert by_id["m1"]["overall"]["fp"] == 0
    assert by_id["m1"]["overall"]["fn"] == 0
    assert by_id["m1"]["name"] == "Model A"

    # m2 predicted "cat" (not a project class) → counted as FP, GT missed → FN
    assert by_id["m2"]["overall"]["tp"] == 0
    assert by_id["m2"]["overall"]["fp"] == 1
    assert by_id["m2"]["overall"]["fn"] == 1

    # per-image structure carries GT + each model's predicted boxes
    img = result["images"][0]
    assert img["stem"] == "img1"
    assert len(img["gt_boxes"]) == 1
    assert len(img["per_model"]) == 2
    assert result["warning"] is None


def test_compare_warns_when_no_classes_defined(tmp_path, fake_ultralytics, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _seed_project(tmp_path, with_classes=False)
    (settings.jobs_dir / "job2").mkdir(parents=True, exist_ok=True)

    run_compare(
        "job2",
        {
            "project_id": "p1",
            "specs": [("m1", "m1.pt")],
            "model_names": {"m1": "Model A"},
            "image_names": ["img1.jpg"],
            "conf": 0.4,
            "iou": 0.5,
            "imgsz": 640,
            "device": "cpu",
        },
        str(settings.jobs_dir),
    )

    result = json.loads((settings.jobs_dir / "job2" / "result.json").read_text())
    assert result["warning"]  # non-empty warning string
    # with no project classes, the "ball" prediction can't map → FP, GT → FN
    overall = result["per_model"][0]["overall"]
    assert overall["fp"] == 1 and overall["fn"] == 1
