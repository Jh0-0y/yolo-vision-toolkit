"""배열 핵심 — 리스트 진입점과 **같은 답**을 내는가, 그리고 float32 로 들었을 때도 그런가.

리스트를 받는 기존 함수들은 이 핵심의 어댑터라, 그 함수들을 pin 한 테스트가 곧 등가성
검증이다. 여기서는 배열 경로에만 있는 것(임계 열을 겹친 `correct`, float32 점수)을 본다.
"""

import numpy as np

from lib.detect.evaluate import (
    IOU_THRESHOLDS,
    _average_precision_arrays,
    _confusion_from_arrays,
    _counts_from_arrays,
    _curves_from_arrays,
    _map_from_arrays,
    average_precision,
    confusion_at,
    counts_at,
    curves_from_flags,
    map_from_accumulated,
)


def _flags(rows):
    scores = np.array([s for s, _ in rows], np.float64)
    is_tp = np.array([t for _, t in rows], bool)
    return scores, is_tp


def test_array_core_and_list_adapter_agree_on_ap():
    rows = [(0.9, True), (0.8, False), (0.75, True), (0.6, True), (0.5, False)]
    assert _average_precision_arrays(*_flags(rows), 4) == average_precision(rows, 4)


def test_array_core_and_list_adapter_agree_on_curves():
    rows = [(0.9, True), (0.8, False), (0.7, True), (0.6, True), (0.5, False)]
    assert _curves_from_arrays(*_flags(rows), 4) == curves_from_flags(rows, 4)


def test_ap_is_unchanged_when_scores_are_held_as_float32():
    """워커는 점수를 float32 로 든다. 4 자리로 반올림된 점수는 float32 에서도 순서가
    보존되므로 AP 가 흔들리면 안 된다 — 흔들리면 저장 레이아웃이 수치를 바꾼 것이다."""
    rows = [(round(1.0 - i / 50, 4), i % 3 != 0) for i in range(50)]
    scores64, is_tp = _flags(rows)
    scores32 = scores64.astype(np.float32)

    assert _average_precision_arrays(scores32, is_tp, 33) == _average_precision_arrays(
        scores64, is_tp, 33
    )


def test_counts_compare_conf_at_the_precision_the_scores_are_stored_in():
    """0.7 은 float32 로 내리면 float64 0.7 보다 **작아진다.** 정밀도를 맞추지 않으면
    점수와 conf 가 딱 같은 자리에서 예측이 통째로 사라진다."""
    scores = np.array([0.7, 0.7], np.float32)
    classes = np.array([0, 0], np.int32)
    is_tp = np.array([True, False], bool)

    out = _counts_from_arrays(scores, classes, is_tp, {0: 1}, 0.7)

    assert out[0] == {"tp": 1, "fp": 1, "fn": 0}


def test_counts_array_core_matches_the_list_adapter():
    rows = {0: [(0.9, True), (0.4, False)], -1: [(0.8, False)]}
    gt = {0: 2, -1: 0}
    scores = np.array([0.9, 0.4, 0.8], np.float64)
    classes = np.array([0, 0, -1], np.int32)
    is_tp = np.array([True, False, False], bool)

    for conf in (0.1, 0.5, 0.95):
        assert _counts_from_arrays(scores, classes, is_tp, gt, conf) == counts_at(rows, gt, conf)


def test_confusion_array_core_matches_the_list_adapter():
    matched = [(0, 0, 0.9), (0, 1, 0.4), (1, 1, 0.6)]
    missed = [1]
    spurious = [(1, 0.7), (-1, 0.8)]
    ids = [0, 1, -1]
    names = {0: "a", 1: "b", -1: "(not in dataset)"}

    for conf in (0.1, 0.5, 0.75, 0.95):
        arrays = _confusion_from_arrays(
            conf,
            np.array([m[0] for m in matched], np.int32),
            np.array([m[1] for m in matched], np.int32),
            np.array([m[2] for m in matched], np.float64),
            np.array(missed, np.int32),
            np.array([s[0] for s in spurious], np.int32),
            np.array([s[1] for s in spurious], np.float64),
            ids,
            names,
        )
        assert arrays == confusion_at(conf, matched, missed, spurious, ids, names)


def test_map_from_arrays_matches_the_accumulated_dict():
    """`correct` 의 열 하나가 IoU 임계 하나다 — 임계마다 리스트를 따로 쌓던 것과 같은 답."""
    rows = [(0, 0.9), (0, 0.6), (1, 0.8), (-1, 0.7)]
    hits = [
        [True] * 10,
        [i < 4 for i in range(10)],
        [i < 7 for i in range(10)],
        [False] * 10,
    ]
    gt = {0: 2, 1: 1, -1: 0}

    acc = {t: {} for t in IOU_THRESHOLDS}
    for (cls, score), hit in zip(rows, hits):
        for j, t in enumerate(IOU_THRESHOLDS):
            acc[t].setdefault(cls, []).append((score, hit[j]))

    out = _map_from_arrays(
        np.array([s for _, s in rows], np.float32),
        np.array([c for c, _ in rows], np.int32),
        np.array(hits, bool),
        gt,
    )

    assert out == map_from_accumulated(acc, gt)
    assert set(out["per_class"]) == {0, 1}  # 정답 없는 -1 은 평균을 흐리지 않는다


def test_empty_arrays_score_zero_without_blowing_up():
    empty_s = np.empty(0, np.float32)
    empty_b = np.empty(0, bool)
    empty_c = np.empty(0, np.int32)

    assert _average_precision_arrays(empty_s, empty_b, 5) == 0.0
    assert _curves_from_arrays(empty_s, empty_b, 5) == {"pr": [], "f1_conf": [], "best_f1": None}
    assert _counts_from_arrays(empty_s, empty_c, empty_b, {0: 3}, 0.5)[0] == {
        "tp": 0, "fp": 0, "fn": 3,
    }
    assert _map_from_arrays(empty_s, empty_c, np.empty((0, 10), bool), {0: 3})["map50"] == 0.0
