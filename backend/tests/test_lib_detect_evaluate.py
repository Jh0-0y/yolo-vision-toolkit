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

    # (클래스, 점수, 붙은 정답 색인, 예측 색인) — 점수 높은 것이 먼저 claim 한다
    assert rows == [(0, 0.9, 1, 0), (0, 0.8, 0, 1)]


def test_indexed_match_marks_unmatched_predictions_with_minus_one():
    from lib.detect.evaluate import match_for_ap_indexed

    gt = [_g(0, 0.0, 0.0, 0.1, 0.1)]
    pred = [_p(0, 0.9, 0.8, 0.8, 0.9, 0.9)]  # 전혀 다른 자리

    assert match_for_ap_indexed(gt, pred, 0.5) == [(0, 0.9, -1, 0)]


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

    assert [(c, s) for c, s, _, _ in new] == [(c, s) for c, s, _ in old]
    assert [i >= 0 for _, _, i, _ in new] == [t for _, _, t in old]


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


def test_match_any_class_pairs_across_classes():
    """공을 선수로 본 것을 잡아내야 혼동행렬이 뜻을 갖는다."""
    from lib.detect.evaluate import match_any_class

    gt = [_g(0, 0.0, 0.0, 0.2, 0.2)]          # 실제로는 클래스 0
    pred = [_p(1, 0.9, 0.0, 0.0, 0.2, 0.2)]   # 클래스 1 로 예측

    matched, missed, spurious = match_any_class(gt, pred)

    assert matched == [(0, 1, 0.9)]
    assert missed == [] and spurious == []


def test_match_any_class_separates_misses_from_spurious():
    from lib.detect.evaluate import match_any_class

    gt = [_g(0, 0.0, 0.0, 0.1, 0.1), _g(1, 0.5, 0.5, 0.6, 0.6)]
    pred = [_p(0, 0.9, 0.0, 0.0, 0.1, 0.1), _p(1, 0.7, 0.9, 0.9, 1.0, 1.0)]

    matched, missed, spurious = match_any_class(gt, pred)

    assert matched == [(0, 0, 0.9)]
    assert missed == [1]              # 두 번째 정답은 아무도 못 잡음
    assert spurious == [(1, 0.7)]     # 두 번째 예측은 헛것


def test_confusion_puts_misses_and_spurious_in_the_background_lane():
    from lib.detect.evaluate import confusion_at

    matched = [(0, 0, 0.9), (0, 1, 0.8)]   # 하나는 맞고 하나는 0 을 1 로 봄
    missed = [0]                            # 놓친 0 하나
    spurious = [(1, 0.7)]                   # 헛것 1 하나

    out = confusion_at(0.5, matched, missed, spurious, [0, 1], {0: "ball", 1: "player"})

    assert out["labels"] == ["ball", "player", "background"]
    # 행=실제, 열=예측
    assert out["rows"][0] == [1, 1, 1]      # 실제 ball: ball 1, player 1, 놓침 1
    assert out["rows"][1] == [0, 0, 0]      # 실제 player: 없음
    assert out["rows"][2] == [0, 1, None]   # 배경을 player 로 본 헛것 1


def test_confusion_demotes_low_score_pairs_to_misses():
    """conf 를 올리면 잘려 나간 짝의 정답은 '놓침'이 된다 — 사라지면 안 된다."""
    from lib.detect.evaluate import confusion_at

    matched = [(0, 0, 0.9), (0, 0, 0.3)]

    high = confusion_at(0.5, matched, [], [], [0], {0: "ball"})
    low = confusion_at(0.2, matched, [], [], [0], {0: "ball"})

    assert low["rows"][0] == [2, 0]     # 둘 다 맞음
    assert high["rows"][0] == [1, 1]    # 하나는 맞고, 잘린 하나는 놓침


def test_confusion_row_totals_never_change_with_conf():
    """conf 가 무엇이든 '실제' 개수는 그대로다 — 정답이 사라질 리 없다."""
    from lib.detect.evaluate import confusion_at

    matched = [(0, 0, 0.9), (0, 1, 0.4), (1, 1, 0.6)]
    missed = [1]

    for c in (0.1, 0.5, 0.95):
        rows = confusion_at(c, matched, missed, [], [0, 1], {0: "a", 1: "b"})["rows"]
        assert sum(rows[0]) == 2   # 실제 a 는 항상 2개
        assert sum(rows[1]) == 2   # 실제 b 는 항상 2개


def test_unknown_class_false_alarms_are_never_hidden():
    """데이터셋에 없는 클래스(-1)의 헛것도 오검출이다 — 지우면 모델이 실제보다 깨끗해 보인다."""
    from lib.detect.evaluate import confusion_at

    # 계약을 지킨 경우: -1 이 class_ids 에 있으면 제 칸에 잡힌다
    ok = confusion_at(0.5, [], [], [(-1, 0.9)], [0, -1], {0: "ball", -1: "(not in dataset)"})
    assert ok["labels"] == ["ball", "(not in dataset)", "background"]
    assert ok["rows"][2][1] == 1          # 배경 행 · (not in dataset) 열
    assert ok["rows"][2][2] is None       # background × background 는 여전히 비어 있다

    # 계약을 어긴 경우라도 **잃지는 않는다**
    leaked = confusion_at(0.5, [], [], [(-1, 0.9)], [0], {0: "ball"})
    assert leaked["rows"][1][1] == 1

    # 셀 것이 없으면 그대로 비어 있다
    empty = confusion_at(0.5, [], [], [], [0], {0: "ball"})
    assert empty["rows"][1][1] is None


def test_counts_at_filters_by_confidence():
    from lib.detect.evaluate import counts_at

    flags = {0: [(0.9, True), (0.7, True), (0.4, False), (0.2, True)]}
    gt = {0: 3}

    high = counts_at(flags, gt, 0.5)
    low = counts_at(flags, gt, 0.1)

    assert high[0] == {"tp": 2, "fp": 0, "fn": 1}
    assert low[0] == {"tp": 3, "fp": 1, "fn": 0}


def test_counts_are_monotonic_in_confidence():
    """conf 를 올리면 TP·FP 는 줄고 FN 은 는다. 안 그러면 세는 규칙이 틀린 것이다."""
    from lib.detect.evaluate import CONF_STEPS, counts_at

    # 0..39 중 3의 배수가 14개이므로 정답으로 표시되는 것은 26개다
    flags = {0: [(1.0 - i / 40, i % 3 != 0) for i in range(40)]}
    gt = {0: 26}

    seq = [counts_at(flags, gt, c)[0] for c in CONF_STEPS]

    assert [s["tp"] for s in seq] == sorted((s["tp"] for s in seq), reverse=True)
    assert [s["fp"] for s in seq] == sorted((s["fp"] for s in seq), reverse=True)
    assert [s["fn"] for s in seq] == sorted(s["fn"] for s in seq)


def test_counts_include_classes_with_no_predictions():
    """예측이 하나도 없어도 정답이 있으면 전부 놓침으로 잡혀야 한다."""
    from lib.detect.evaluate import counts_at

    assert counts_at({}, {0: 5}, 0.5)[0] == {"tp": 0, "fp": 0, "fn": 5}


def test_conf_steps_span_the_useful_range():
    from lib.detect.evaluate import CONF_STEPS

    assert len(CONF_STEPS) == 19
    assert CONF_STEPS[0] == 0.05 and CONF_STEPS[-1] == 0.95


def test_counts_feed_the_existing_aggregate_unchanged():
    """스냅샷의 표는 지금 결과의 표와 같은 모양이어야 화면이 하나로 그린다."""
    from lib.detect.evaluate import aggregate, counts_at

    out = aggregate(counts_at({0: [(0.9, True)]}, {0: 2}, 0.5), {0: "ball"})

    assert out["overall"] == {"tp": 1, "fp": 0, "fn": 1,
                              "precision": 1.0, "recall": 0.5, "f1": 0.6667}
    assert out["per_class"][0]["name"] == "ball"
