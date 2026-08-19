"""타일 추론의 좌표 규칙과 경계 처리.

모델 없이 검증한다 — `collect` 는 ultralytics 를 모르고 좌표만 다룬다. 그래서
"타일 안에서 본 것을 원본 어디에 놓을 것인가"라는 규칙을 여기서 못 박을 수 있다.
"""

import pytest

from lib.detect.tiled import TiledParams, collect, merge_tile_detections, restore, tiles_for
from lib.detect.ensemble import Detection


def test_tiles_cover_the_image_with_the_last_one_clamped():
    """마지막 타일은 끝에 붙는다 — 안 그러면 오른쪽·아래가 안 보인다."""
    params = TiledParams(tile_size=640, stride=480)

    tiles = tiles_for(1920, 1080, params)

    xs = sorted({x for x, _ in tiles})
    ys = sorted({y for _, y in tiles})
    assert xs[0] == 0 and xs[-1] == 1920 - 640
    assert ys[0] == 0 and ys[-1] == 1080 - 640
    assert len(tiles) == len(xs) * len(ys)


def test_restore_maps_tile_pixels_to_original_normalized():
    # 타일이 (640, 240)에서 시작하고, 그 안 (10,20)-(50,60) 에 박스가 있다
    assert restore((10, 20, 50, 60), 640, 240, 1280, 720) == (
        650 / 1280, 260 / 720, 690 / 1280, 300 / 720,
    )


# ---------- 경계 처리 ----------
#
# 타일 안에서는 잘린 객체도 온전해 보인다 — "작은 공"인지 "잘린 공"인지 구분할 수
# 없다. 그래서 경계에 붙은 것은 버리고, 겹침 구간에서 옆 타일이 온전히 본 것을 쓴다.


def test_a_box_touching_an_inner_edge_is_dropped():
    params = TiledParams(tile_size=640, stride=480, border_margin_px=4)
    # 타일이 (0,0) 이고 오른쪽 경계(x=640)에 붙은 박스 — 오른쪽에 이미지가 더 있다
    boxes = [(0, 0, (600.0, 100.0, 638.0, 140.0), 0, 0.9)]

    assert collect(boxes, 1920, 1080, params, "m", {0: 7}) == []


def test_a_box_touching_the_image_edge_is_kept():
    """이미지 바깥 테두리는 잘린 게 아니라 원래 거기가 끝이다."""
    params = TiledParams(tile_size=640, stride=480, border_margin_px=4)
    # 1280 폭 이미지에서 마지막 타일은 x=640..1280 — 그 오른쪽 경계가 곧 이미지 끝
    boxes = [(640, 0, (600.0, 100.0, 638.0, 140.0), 0, 0.9)]

    out = collect(boxes, 1280, 1080, params, "m", {0: 7})

    assert len(out) == 1
    assert out[0].cls == 7


def test_unmapped_model_classes_are_dropped():
    """모델이 가진 클래스가 데이터셋에 없으면 라벨로 쓸 수 없다."""
    params = TiledParams(tile_size=640, stride=480)
    boxes = [(0, 0, (100.0, 100.0, 200.0, 200.0), 3, 0.9)]

    assert collect(boxes, 1920, 1080, params, "m", {0: 7}) == []


# ---------- 겹침 중복 ----------


def _det(x1, y1, x2, y2, score, cls=0):
    return Detection(cls=cls, xyxy=(x1, y1, x2, y2), score=score, model_id="m")


def test_the_same_object_seen_by_two_tiles_collapses_to_one():
    dets = [_det(0.1, 0.1, 0.2, 0.2, 0.8), _det(0.105, 0.1, 0.2, 0.205, 0.9)]

    out = merge_tile_detections(dets, iou_thr=0.5)

    assert len(out) == 1
    assert out[0].score == 0.9  # 점수 높은 쪽을 남긴다 — 평균하지 않는다


def test_different_classes_never_collapse():
    dets = [_det(0.1, 0.1, 0.2, 0.2, 0.8, cls=0), _det(0.1, 0.1, 0.2, 0.2, 0.9, cls=1)]

    assert len(merge_tile_detections(dets, iou_thr=0.5)) == 2


def test_separate_objects_survive():
    dets = [_det(0.1, 0.1, 0.2, 0.2, 0.9), _det(0.6, 0.6, 0.7, 0.7, 0.8)]

    assert len(merge_tile_detections(dets, iou_thr=0.5)) == 2


# ---------- 검증 ----------


@pytest.mark.parametrize(
    "params, hint",
    [
        (TiledParams(tile_size=640, stride=700), "stride"),  # 겹침이 음수
        (TiledParams(merge_iou=1.5), "merge_iou"),
        (TiledParams(border_margin_px=-1), "border_margin"),
    ],
)
def test_bad_params_are_reported(params, hint):
    assert any(hint in e for e in params.validate())


def test_good_params_pass():
    assert TiledParams(tile_size=640, stride=480).validate() == []
