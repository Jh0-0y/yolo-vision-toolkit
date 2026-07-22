import pytest

from app.core.ensemble import Detection, FusedBox, fuse_class, iou, merge_detections


def det(cls=0, xyxy=(0.1, 0.1, 0.3, 0.3), score=0.9, model="a"):
    return Detection(cls=cls, xyxy=xyxy, score=score, model_id=model)


def test_iou_identical_and_disjoint():
    a = (0.1, 0.1, 0.3, 0.3)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, (0.5, 0.5, 0.7, 0.7)) == 0.0


def test_fuse_overlapping_boxes_from_two_models():
    d1 = det(score=0.9, model="a", xyxy=(0.10, 0.10, 0.30, 0.30))
    d2 = det(score=0.7, model="b", xyxy=(0.11, 0.11, 0.31, 0.31))
    fused = fuse_class([d1, d2], iou_thr=0.55)

    assert len(fused) == 1
    fb = fused[0]
    assert fb.agree_count == 2
    assert fb.score == pytest.approx(0.8)  # mean of member scores
    # coords are score-weighted toward the higher-confidence box
    assert fb.xyxy[0] < 0.11


def test_no_fusion_below_iou_threshold():
    d1 = det(model="a", xyxy=(0.1, 0.1, 0.3, 0.3))
    d2 = det(model="b", xyxy=(0.6, 0.6, 0.8, 0.8))
    fused = fuse_class([d1, d2], iou_thr=0.55)
    assert len(fused) == 2
    assert all(fb.agree_count == 1 for fb in fused)


def test_single_source_class_passes_through():
    dets = [det(cls=0, model="a", score=0.42)]
    out = merge_detections(dets, class_sources={0: ["a"]})
    assert len(out) == 1
    assert isinstance(out[0], FusedBox)
    assert out[0].score == pytest.approx(0.42)
    assert out[0].xyxy == dets[0].xyxy  # untouched


def test_merge_groups_by_class():
    # same location, different classes → must NOT fuse together
    d1 = det(cls=0, model="a")
    d2 = det(cls=1, model="b")
    out = merge_detections(d1_and_d2 := [d1, d2], class_sources={0: ["a", "b"], 1: ["a", "b"]})
    assert len(out) == len(d1_and_d2)
