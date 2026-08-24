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


class _Box:
    """One detection, stubbing the singular per-box API the TILED path reads
    directly off `r.boxes` iteration (`b.xyxy[0].tolist()`, `b.cls.item()`,
    `b.conf.item()`) — mirrors one entry of ultralytics' `Boxes.__iter__()`.
    Additive: the full-frame path never iterates `_Boxes`, it reads the
    batched `.xyxyn`/`.cls`/`.conf` arrays instead."""

    def __init__(self, cls, xyxy, conf):
        self.xyxy = np.asarray([xyxy])
        self.cls = np.asarray(cls)
        self.conf = np.asarray(conf)


class _Boxes:
    def __init__(self, dets):
        self._dets = dets  # [(cls, xyxy, conf), ...] — kept for __iter__
        self.xyxyn = _Arr([d[1] for d in dets])
        self.cls = _Arr([d[0] for d in dets])
        self.conf = _Arr([d[2] for d in dets])

    def __iter__(self):
        for cls, xyxy, conf in self._dets:
            yield _Box(cls, xyxy, conf)


class _Result:
    def __init__(self, dets, shape=(100, 200)):
        self.orig_shape = shape
        self.boxes = _Boxes(dets) if dets else None


# model stem -> class names; and its detections on any image
MODEL_NAMES = {
    "m1": {0: "ball"},
    "m2": {0: "cat"},
    "t_oov": {0: "cat"},  # tiled, out-of-vocabulary (not a dataset class)
    "t_iv": {0: "ball"},  # tiled, in-vocabulary
}
MODEL_PREDS = {
    "m1": [(0, (0.10, 0.10, 0.30, 0.30), 0.90)],  # "ball" over the GT box → TP
    "m2": [(0, (0.10, 0.10, 0.30, 0.30), 0.90)],  # "cat" (not in dataset) → FP
}
# tiled path: one det-list per tile (tile-pixel xyxy, matches `tiles_for`'s
# order for a 200x100 image at tile_size=stride=128 → [(0,0), (72,0)]).
TILE_PREDS = {
    "t_oov": [
        [(0, (20.0, 20.0, 50.0, 50.0), 0.9)],  # tile (0,0): a "cat" box, clear of any inner border
        [],  # tile (72,0): nothing — also exercises the `r.boxes is None` guard
    ],
    "t_iv": [
        [(0, (20.0, 10.0, 60.0, 30.0), 0.9)],  # tile (0,0): "ball" placed exactly over the GT box
        [],
    ],
}


class FakeYOLO:
    def __init__(self, path):
        self._id = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        self.names = MODEL_NAMES[self._id]

    def to(self, device):
        return self

    def predict(self, image, **kwargs):
        if isinstance(image, list):
            # tiled path: `image` is the list of tile crops, one `_Result` per crop
            return [_Result(dets) for dets in TILE_PREDS[self._id]]
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


def _entry(entry_id, model_id, name, pt):
    """A `full`-mode entry — the tile knobs are unused on this path but the
    cfg contract requires every entry to carry them."""
    return {
        "entry_id": entry_id,
        "model_id": model_id,
        "name": name,
        "pt": pt,
        "mode": "full",
        "imgsz": 640,
        "tile_size": 640,
        "stride": 480,
        "merge_iou": 0.5,
        "border_margin_px": 4,
    }


def _tiled_entry(entry_id, model_id, name, pt):
    """A `tiled`-mode entry sized for the 200x100 fixture image: 128px
    tile/stride makes `tiles_for` yield exactly two tiles, (0,0) and (72,0)."""
    return {
        "entry_id": entry_id,
        "model_id": model_id,
        "name": name,
        "pt": pt,
        "mode": "tiled",
        "imgsz": 640,
        "tile_size": 128,
        "stride": 128,
        "merge_iou": 0.5,
        "border_margin_px": 4,
    }


def _run(job, ds, out_dir, entries):
    (settings.jobs_dir / job).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_compare(
        job,
        {
            "project_id": "p1",
            "entries": entries,
            "dataset_dir": str(ds),
            "out_dir": str(out_dir),
            "conf": 0.4,
            "iou": 0.5,
            "device": "cpu",
        },
        str(settings.jobs_dir),
    )
    return json.loads((out_dir / "result.json").read_text())


def test_compare_scores_each_model_and_counts_unknown_class_as_fp(
    tmp_path, fake_ultralytics, monkeypatch
):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    # the image route confines served files to the benchmark run directory; the
    # worker writes absolute image paths into the manifest regardless, so just seed a ds.
    ds = _seed_dataset(tmp_path)
    out_dir = tmp_path / "out1"

    entries = [_entry("m1", "m1", "Model A", "m1.pt"), _entry("m2", "m2", "Model B", "m2.pt")]
    result = _run("job1", ds, out_dir, entries)
    by_id = {e["entry_id"]: e for e in result["per_entry"]}

    # m1 predicted "ball" over the GT → true positive, perfect scores + mAP 1.0
    assert by_id["m1"]["overall"]["tp"] == 1
    assert by_id["m1"]["overall"]["fp"] == 0
    assert by_id["m1"]["overall"]["fn"] == 0
    assert by_id["m1"]["model_id"] == "m1"
    assert by_id["m1"]["name"] == "Model A"
    assert by_id["m1"]["mode"] == "full"
    # 화면의 엔트리 제목(`Model A · full 640`)이 여기서 나온다
    assert by_id["m1"]["imgsz"] == 640
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

    # 새 지표 — 곡선 · 동작점 · 혼동행렬 · 크기별 AP · 속도
    e = by_id["m1"]
    assert e["curves"]["pr"], "PR 곡선이 실려야 한다"
    assert e["curves"]["best_f1"]["value"] >= 0
    assert len(e["operating_points"]) == 19
    # 동작점은 conf 오름차순이고, 대표 동작점의 표와 같은 모양이어야 한다
    assert [o["conf"] for o in e["operating_points"]] == sorted(
        o["conf"] for o in e["operating_points"]
    )
    assert set(e["operating_points"][0]["overall"]) == set(e["overall"])
    assert e["operating_points"][0]["confusion"]["labels"][-1] == "background"
    assert e["speed"] is None or e["speed"]["ms_median"] >= 0
    assert "by_size" in e
    # ap50 은 map50 과 같은 값이고, ap75 는 정답이 있으니 수치가 나온다
    assert e["ap50"] == e["map50"]
    assert e["ap75"] == 1.0
    # 200x100 프레임에서 [0.10,0.10,0.30,0.30] 은 40x20=800px² → small 하나뿐
    assert set(e["by_size"]) == {"small"}
    assert e["by_size"]["small"] == {"ap50": 1.0, "ap": 1.0, "gt": 1}
    # conf 0.05 에서는 0.9 짜리 예측이 살아 있어 대각선이 1, 0.95 에서는 놓침으로 강등된다
    assert e["operating_points"][0]["confusion"]["rows"][0][0] == 1
    assert e["operating_points"][-1]["conf"] == 0.95
    assert e["operating_points"][-1]["confusion"]["rows"][0][-1] == 1
    # 이미지가 한 장뿐이라 워밍업 한 장을 버리면 표본이 없다 — 터지지 않고 None 이어야 한다
    assert e["speed"] is None
    # 가짜 YOLO 는 실제 가중치 파일도 .model 도 없어 둘 다 못 읽는다 → None
    assert e["model"] is None

    # 데이터셋에 없는 클래스로 예측한 m2 는 그 열을 따로 가져야 오검출이 숨지 않는다
    m2_confusion = by_id["m2"]["operating_points"][0]["confusion"]
    assert m2_confusion["labels"] == ["ball", "(not in dataset)", "background"]
    assert m2_confusion["rows"][0][1] == 1  # 실제 ball 을 어휘 밖 클래스로 봤다
    assert m2_confusion["rows"][-1][-1] is None  # background×background 는 비어 있다
    # m2 는 정답 클래스에 맞은 예측이 없어 곡선이 비고 ap75 는 0
    assert by_id["m2"]["curves"]["pr"] == []
    assert by_id["m2"]["curves"]["best_f1"] is None
    assert by_id["m2"]["ap75"] == 0.0

    # per-image structure carries GT + each entry's predicted boxes
    img = result["images"][0]
    assert img["stem"] == "img1"
    assert len(img["gt_boxes"]) == 1
    assert len(img["per_entry"]) == 2
    assert result["warning"] is None

    # manifest maps the image index → an existing file (for the overlay route);
    # it lives in the benchmark's out_dir, not the job directory
    manifest = json.loads((out_dir / "images_manifest.json").read_text())
    assert "0" in manifest


def test_compare_warns_when_dataset_has_no_class_names(tmp_path, fake_ultralytics, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    ds = _seed_dataset(tmp_path)
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": {}, "nc": 0}))  # no names
    out_dir = tmp_path / "out2"

    result = _run("job2", ds, out_dir, [_entry("m1", "m1", "Model A", "m1.pt")])
    assert result["warning"]  # non-empty warning string
    # with no dataset classes, the "ball" prediction can't map → FP, GT → FN
    overall = result["per_entry"][0]["overall"]
    assert overall["fp"] == 1 and overall["fn"] == 1


def test_compare_tiled_entry_counts_oov_as_fp_and_scores_in_vocab_as_tp(
    tmp_path, fake_ultralytics, monkeypatch
):
    """The whole tiled feature rests on: `collect()` already maps a tiled
    detection's class to the dataset's id (or -1), and the worker must NOT
    re-map it by name — an out-of-vocabulary tiled box has to be counted as a
    false positive, not silently dropped, and an in-vocabulary one has to
    score normally (proving the two paths land on the same kind of result)."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    ds = _seed_dataset(tmp_path)  # names=("ball",); GT xyxy_n [0.10,0.10,0.30,0.30] on 200x100
    out_dir = tmp_path / "out3"

    entries = [
        _tiled_entry("t_oov", "t_oov", "OOV Tiled", "t_oov.pt"),
        _tiled_entry("t_iv", "t_iv", "In-vocab Tiled", "t_iv.pt"),
    ]
    result = _run("job3", ds, out_dir, entries)
    by_id = {e["entry_id"]: e for e in result["per_entry"]}

    # out-of-vocabulary: cls_map sends "cat" (unknown to the dataset) to -1;
    # collect() already applied that mapping, so this must still show up as
    # a counted false positive — never dropped.
    oov = by_id["t_oov"]
    assert oov["mode"] == "tiled"
    # 타일 엔트리의 제목은 tile_size 로 붙는다 (`OOV Tiled · tiled 128`)
    assert oov["tile_size"] == 128
    assert oov["overall"]["tp"] == 0
    assert oov["overall"]["fp"] == 1
    assert oov["overall"]["fn"] == 1
    assert oov["detections"] == 1
    assert oov["map50"] == 0.0

    # in-vocabulary: cls_map sends "ball" to the dataset's own id 0; the box
    # sits exactly over the GT box after tile->image coordinate restoration,
    # so it must match as a true positive — proving the tiled path is not
    # re-running the name-based lookup on an already-mapped class id.
    iv = by_id["t_iv"]
    assert iv["mode"] == "tiled"
    assert iv["overall"]["tp"] == 1
    assert iv["overall"]["fp"] == 0
    assert iv["overall"]["fn"] == 0
    assert iv["map50"] == 1.0
