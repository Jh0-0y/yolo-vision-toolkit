import pytest

from app.core.decision import DecisionConfig, judge_boxes, route_image, uncertainty_score
from app.core.ensemble import FusedBox


def fbox(cls=0, xyxy=(0.1, 0.1, 0.3, 0.3), score=0.9, sources=(("a", 0.9),)):
    return FusedBox(cls=cls, xyxy=xyxy, score=score, sources=list(sources))


CFG = DecisionConfig(conf_min=0.10, conf_confirm=0.60)


def test_confident_single_source_box_auto_ok():
    v = judge_boxes([fbox(score=0.9)], {0: ["a"]}, CFG)
    assert v[0].status == "auto_ok"


def test_low_confidence_flagged():
    v = judge_boxes([fbox(score=0.3)], {0: ["a"]}, CFG)
    assert v[0].status == "needs_review"
    assert v[0].reason == "low_conf"


def test_multi_source_class_needs_agreement():
    # class owned by two models but only one detected it → disagreement
    v = judge_boxes([fbox(score=0.9, sources=[("a", 0.9)])], {0: ["a", "b"]}, CFG)
    assert v[0].status == "needs_review"
    assert v[0].reason == "disagreement"

    # both models agree → ok
    v = judge_boxes(
        [fbox(score=0.9, sources=[("a", 0.9), ("b", 0.85)])], {0: ["a", "b"]}, CFG
    )
    assert v[0].status == "auto_ok"


def test_class_conflict_flags_lower_scoring_box():
    a = fbox(cls=0, score=0.9, xyxy=(0.1, 0.1, 0.3, 0.3))
    b = fbox(cls=1, score=0.7, xyxy=(0.11, 0.11, 0.31, 0.31), sources=[("b", 0.7)])
    v = judge_boxes([a, b], {0: ["a"], 1: ["b"]}, CFG)
    assert v[0].status == "auto_ok"
    assert v[1].status == "needs_review"
    assert v[1].reason == "class_conflict"


def test_per_class_threshold_override():
    cfg = DecisionConfig(conf_confirm=0.60, per_class_confirm={0: 0.95})
    v = judge_boxes([fbox(score=0.9)], {0: ["a"]}, cfg)
    assert v[0].status == "needs_review"


def test_route_image_buckets():
    ok = judge_boxes([fbox(score=0.9)], {0: ["a"]}, CFG)
    assert route_image(ok, CFG) == "confirmed"

    flagged = judge_boxes([fbox(score=0.3)], {0: ["a"]}, CFG)
    assert route_image(flagged, CFG) == "review"

    assert route_image([], CFG) == "review"
    assert route_image([], DecisionConfig(empty_policy="negative")) == "negative"


def test_uncertainty_score():
    flagged = judge_boxes([fbox(score=0.3)], {0: ["a"]}, CFG)
    assert uncertainty_score(flagged) == pytest.approx(0.7)
    ok = judge_boxes([fbox(score=0.9)], {0: ["a"]}, CFG)
    assert uncertainty_score(ok) == 0.0
