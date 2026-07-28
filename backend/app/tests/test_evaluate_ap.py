"""AP / mAP math in ml.evaluate — pure, no torch."""

from app.ml.evaluate import (
    IOU_THRESHOLDS,
    average_precision,
    map_from_accumulated,
    match_for_ap,
)


def test_iou_thresholds_are_coco_sweep():
    assert IOU_THRESHOLDS == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def test_average_precision_edges():
    assert average_precision([(0.9, True)], 1) == 1.0  # single perfect TP
    assert average_precision([(0.9, False)], 1) == 0.0  # single FP
    assert average_precision([], 3) == 0.0  # no predictions
    assert average_precision([(0.9, True)], 0) == 0.0  # no GT → 0 by convention


def test_average_precision_partial_recall():
    # 2 GT, one TP (recall caps at 0.5 with precision 1.0), one FP
    ap = average_precision([(0.9, True), (0.8, False)], 2)
    assert round(ap, 3) == 0.505  # 51/101 recall levels covered at precision 1


def test_match_for_ap_flags_tp_and_fp():
    gt = [{"cls": 0, "xyxy_n": [0.1, 0.1, 0.3, 0.3]}]
    pred = [
        {"cls": 0, "xyxyn": [0.1, 0.1, 0.3, 0.3], "score": 0.9},  # overlaps GT → TP
        {"cls": 0, "xyxyn": [0.6, 0.6, 0.8, 0.8], "score": 0.5},  # elsewhere → FP
    ]
    rows = match_for_ap(gt, pred, 0.5)
    assert (0, 0.9, True) in rows
    assert (0, 0.5, False) in rows


def test_map_from_accumulated_excludes_classes_without_gt():
    # class 0 has GT and a perfect TP at every IoU; sentinel -1 has detections but
    # no GT and must not dilute the mean.
    acc = {t: {0: [(0.9, True)], -1: [(0.8, False)]} for t in IOU_THRESHOLDS}
    out = map_from_accumulated(acc, {0: 1, -1: 0})
    assert out["map50"] == 1.0
    assert out["map"] == 1.0
    assert set(out["per_class"]) == {0}  # -1 excluded
