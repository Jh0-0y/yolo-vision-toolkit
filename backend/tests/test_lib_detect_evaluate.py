"""AP / mAP math in ml.evaluate — pure, no torch."""

from lib.detect.evaluate import (
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


def test_build_cls_map_maps_known_names_to_dataset_ids():
    from lib.detect.evaluate import build_cls_map

    model_names = {0: "ball", 1: "player"}
    ds_by_norm = {"ball": 3, "player": 7}

    assert build_cls_map(model_names, ds_by_norm) == {0: 3, 1: 7}


def test_build_cls_map_sends_unknown_classes_to_the_sentinel():
    """데이터셋에 없는 클래스는 **버리지 않고** 오검출로 센다 — 풀 프레임 경로와
    같은 처리라야 두 방식의 점수를 나란히 놓을 수 있다."""
    from lib.detect.evaluate import build_cls_map

    model_names = {0: "ball", 1: "referee"}
    ds_by_norm = {"ball": 0}

    assert build_cls_map(model_names, ds_by_norm) == {0: 0, 1: -1}


def test_build_cls_map_covers_every_model_class():
    """하나라도 빠지면 collect 가 그 클래스를 조용히 버린다."""
    from lib.detect.evaluate import build_cls_map

    model_names = {0: "a", 1: "b", 2: "c"}

    assert set(build_cls_map(model_names, {}).keys()) == {0, 1, 2}
    assert set(build_cls_map(model_names, {}).values()) == {-1}


def test_build_cls_map_normalizes_names():
    """이름 대조는 정규화해서 한다 — 'Ball' 과 'ball' 은 같은 클래스다."""
    from lib.detect.evaluate import build_cls_map
    from lib.labels.registry import normalize

    ds_by_norm = {normalize("Ball"): 2}

    assert build_cls_map({0: "ball"}, ds_by_norm) == {0: 2}


def test_pr_curve_walks_recall_up_as_confidence_drops():
    """점수 내림차순으로 훑으면 재현율은 단조 증가한다 — 정답을 더 주워 담기 때문이다."""
    from lib.detect.evaluate import curves_from_flags

    # 정답 4개 중 3개를 맞히고 오검출 2개
    flags = [(0.9, True), (0.8, False), (0.7, True), (0.6, True), (0.5, False)]

    out = curves_from_flags(flags, n_gt=4)

    recalls = [r for r, _ in out["pr"]]
    assert recalls == sorted(recalls)
    assert recalls[-1] == 0.75  # 3/4
    assert out["pr"][0] == [0.25, 1.0]  # 첫 예측이 정답 → P 1.0, R 1/4


def test_f1_curve_peaks_at_the_best_operating_point():
    from lib.detect.evaluate import curves_from_flags

    # 0.7 위에서 정답만 3개, 그 아래로 오검출만 → 최고 F1 은 0.7 언저리
    flags = [(0.9, True), (0.8, True), (0.7, True), (0.4, False), (0.3, False)]

    out = curves_from_flags(flags, n_gt=3)

    assert out["best_f1"]["value"] == 1.0
    assert out["best_f1"]["conf"] == 0.7
    # 곡선의 최댓값과 best_f1 이 어긋나면 둘 중 하나가 틀린 것이다
    assert max(f1 for _, f1 in out["f1_conf"]) == out["best_f1"]["value"]


def test_curves_are_downsampled_but_keep_the_ends():
    """점이 많아도 파일이 붓지 않게 줄이되, 양 끝은 남겨 곡선 모양을 지킨다."""
    from lib.detect.evaluate import curves_from_flags

    # (i+1)%3 로 두어 **첫 예측이 정답**이 되게 한다 — 그래야 첫 점의 재현율이 0 이 아니고
    # "양 끝을 남긴다"를 실제로 검사할 수 있다. 정답 수는 1000 중 3의 배수를 뺀 667.
    flags = [(1.0 - i / 1000, (i + 1) % 3 != 0) for i in range(1000)]

    out = curves_from_flags(flags, n_gt=667, max_points=50)

    assert len(out["pr"]) <= 50
    assert len(out["f1_conf"]) <= 50
    assert out["pr"][0][0] > 0  # 첫 점이 남았다
    assert out["pr"][-1][0] == max(r for r, _ in out["pr"])  # 마지막이 최대 재현율


def test_no_ground_truth_yields_empty_curves():
    """정답이 없는 클래스는 곡선이 없다 — 0 짜리 곡선을 그리면 거짓말이 된다."""
    from lib.detect.evaluate import curves_from_flags

    out = curves_from_flags([(0.9, False)], n_gt=0)

    assert out == {"pr": [], "f1_conf": [], "best_f1": None}


def test_best_f1_survives_downsampling():
    """라벨이 가리키는 봉우리가 그림에 실제로 있어야 한다 — 솎아 내면서 잃기 쉽다."""
    from lib.detect.evaluate import curves_from_flags

    # 가장 높은 점수 5개만 정답이고 나머지는 전부 오검출 → 최댓값이 곡선 맨 앞에 몰린다
    flags = [(1.0 - i / 500, i < 5) for i in range(500)]

    out = curves_from_flags(flags, n_gt=5, max_points=50)

    assert out["best_f1"]["value"] == max(f1 for _, f1 in out["f1_conf"])
    assert len(out["f1_conf"]) <= 50
    assert len(out["pr"]) <= 50


def _g(cls, x1, y1, x2, y2):
    return {"cls": cls, "xyxy_n": [x1, y1, x2, y2]}


def _p(cls, score, x1, y1, x2, y2):
    return {"cls": cls, "score": score, "xyxyn": [x1, y1, x2, y2]}


def test_indexed_match_reports_which_ground_truth_was_claimed():
    from lib.detect.evaluate import match_for_ap_indexed

    gt = [_g(0, 0.0, 0.0, 0.1, 0.1), _g(0, 0.5, 0.5, 0.6, 0.6)]
    pred = [_p(0, 0.9, 0.5, 0.5, 0.6, 0.6), _p(0, 0.8, 0.0, 0.0, 0.1, 0.1)]

    rows = match_for_ap_indexed(gt, pred, 0.5)

    assert rows == [(0, 0.9, 1), (0, 0.8, 0)]  # 점수 높은 것이 먼저 claim


def test_indexed_match_marks_unmatched_predictions_with_minus_one():
    from lib.detect.evaluate import match_for_ap_indexed

    gt = [_g(0, 0.0, 0.0, 0.1, 0.1)]
    pred = [_p(0, 0.9, 0.8, 0.8, 0.9, 0.9)]  # 전혀 다른 자리

    assert match_for_ap_indexed(gt, pred, 0.5) == [(0, 0.9, -1)]


def test_indexed_match_agrees_with_the_existing_matcher():
    """기존 match_for_ap 과 답이 갈리면 안 된다 — 같은 규칙의 다른 표현일 뿐이다."""
    from lib.detect.evaluate import match_for_ap, match_for_ap_indexed

    gt = [_g(0, 0.0, 0.0, 0.2, 0.2), _g(1, 0.5, 0.5, 0.7, 0.7)]
    pred = [
        _p(0, 0.9, 0.0, 0.0, 0.2, 0.2),
        _p(1, 0.7, 0.5, 0.5, 0.7, 0.7),
        _p(0, 0.6, 0.9, 0.9, 1.0, 1.0),
    ]

    old = match_for_ap(gt, pred, 0.5)
    new = match_for_ap_indexed(gt, pred, 0.5)

    assert [(c, s) for c, s, _ in new] == [(c, s) for c, s, _ in old]
    assert [i >= 0 for _, _, i in new] == [t for _, _, t in old]


def test_size_buckets_follow_coco_area_thresholds():
    from lib.detect.evaluate import size_of

    # 1000×1000 이미지에서 넓이가 픽셀로 얼마인지 계산해 구간을 고른다
    assert size_of([0.0, 0.0, 0.03, 0.03], 1000, 1000) == "small"    # 30×30 = 900 < 1024
    assert size_of([0.0, 0.0, 0.05, 0.05], 1000, 1000) == "medium"   # 50×50 = 2500
    assert size_of([0.0, 0.0, 0.2, 0.2], 1000, 1000) == "large"      # 200×200 = 40000


def test_size_bucket_uses_real_pixels_not_normalized_area():
    """정규화 넓이가 같아도 이미지가 크면 실제 객체는 크다 — 픽셀로 재야 한다."""
    from lib.detect.evaluate import size_of

    box = [0.0, 0.0, 0.04, 0.04]
    assert size_of(box, 500, 500) == "small"     # 20×20 = 400
    assert size_of(box, 1920, 1080) == "medium"  # 76.8×43.2 ≈ 3318
