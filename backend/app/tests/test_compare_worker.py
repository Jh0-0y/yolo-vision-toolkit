"""compare_worker end-to-end with a stubbed ultralytics.YOLO (no weights/GPU).

Verifies per-model scoring against an uploaded YOLO dataset (images + labels +
data.yaml), the class-mismatch fix (a prediction whose class isn't in the dataset
is a false positive, not dropped), and the mAP metrics."""

import json
import sys
import types

import numpy as np
import pytest
import yaml
from PIL import Image

from app.core.config import settings
from app.workers.compare_worker import run_compare
from lib.labels.io import write_label_file


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
    "m2": [(0, (0.10, 0.10, 0.30, 0.30), 0.90)],  # "cat" (not in dataset) → FP
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


def _seed_dataset(tmp_path, names=("ball",)):
    """A minimal YOLO test set: one image + one 'ball' GT box + data.yaml."""
    ds = tmp_path / "dataset"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(ds / "images" / "img1.jpg")
    # ground truth: one class-0 box at xyxy_n [0.10,0.10,0.30,0.30]
    write_label_file(ds / "labels" / "img1.txt", [(0, (0.10, 0.10, 0.30, 0.30))])
    (ds / "data.yaml").write_text(
        yaml.safe_dump({"names": {i: n for i, n in enumerate(names)}, "nc": len(names)})
    )
    return ds


def _run(job, ds, specs, model_names):
    (settings.jobs_dir / job).mkdir(parents=True, exist_ok=True)
    run_compare(
        job,
        {
            "project_id": "p1",
            "specs": specs,
            "model_names": model_names,
            "dataset_dir": str(ds),
            "conf": 0.4,
            "iou": 0.5,
            "imgsz": 640,
            "device": "cpu",
        },
        str(settings.jobs_dir),
    )
    return json.loads((settings.jobs_dir / job / "result.json").read_text())


def test_compare_scores_each_model_and_counts_unknown_class_as_fp(
    tmp_path, fake_ultralytics, monkeypatch
):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # the image route confines served files to test_dir/compare/{job}; the worker
    # writes absolute image paths into the manifest regardless, so just seed a ds.
    ds = _seed_dataset(tmp_path)

    result = _run("job1", ds, [("m1", "m1.pt"), ("m2", "m2.pt")], {"m1": "Model A", "m2": "Model B"})
    by_id = {m["model_id"]: m for m in result["per_model"]}

    # m1 predicted "ball" over the GT → true positive, perfect scores + mAP 1.0
    assert by_id["m1"]["overall"]["tp"] == 1
    assert by_id["m1"]["overall"]["fp"] == 0
    assert by_id["m1"]["overall"]["fn"] == 0
    assert by_id["m1"]["name"] == "Model A"
    assert by_id["m1"]["map50"] == 1.0
    assert by_id["m1"]["map"] == 1.0

    # m2 predicted "cat" (not a dataset class) → FP, GT missed → FN, mAP 0
    assert by_id["m2"]["overall"]["tp"] == 0
    assert by_id["m2"]["overall"]["fp"] == 1
    assert by_id["m2"]["overall"]["fn"] == 1
    assert by_id["m2"]["map50"] == 0.0

    # per-class rows carry AP; the ball class is present for m1
    ball = next(c for c in by_id["m1"]["per_class"] if c["name"] == "ball")
    assert ball["ap50"] == 1.0 and ball["ap"] == 1.0

    # per-image structure carries GT + each model's predicted boxes
    img = result["images"][0]
    assert img["stem"] == "img1"
    assert len(img["gt_boxes"]) == 1
    assert len(img["per_model"]) == 2
    assert result["warning"] is None

    # manifest maps the image index → an existing file (for the overlay route)
    manifest = json.loads((settings.jobs_dir / "job1" / "images_manifest.json").read_text())
    assert "0" in manifest


def test_compare_warns_when_dataset_has_no_class_names(tmp_path, fake_ultralytics, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    ds = _seed_dataset(tmp_path)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": {}, "nc": 0}))  # no names

    result = _run("job2", ds, [("m1", "m1.pt")], {"m1": "Model A"})
    assert result["warning"]  # non-empty warning string
    # with no dataset classes, the "ball" prediction can't map → FP, GT → FN
    overall = result["per_model"][0]["overall"]
    assert overall["fp"] == 1 and overall["fn"] == 1
