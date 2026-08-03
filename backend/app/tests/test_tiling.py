"""Unit tests for 이미지 타일링 (domain/tiling).

1920×1080 기준 스펙 검증: 640/480 → 4×2 = 8타일, 겹침 160px, 마지막 타일 클램프,
라벨 자동 처리(min_visibility 클립/삭제, 기본 0.6).
"""

from app.domain.tiling import (
    TilingParams,
    clip_boxes_to_tile,
    tile_grid,
    tile_offsets,
    tile_stem,
)

P = TilingParams()  # 640 / 480 / 0.6


def test_default_grid_1920x1080_is_8_tiles():
    grid = tile_grid(1920, 1080, P)
    assert len(grid) == 8  # 가로 4 × 세로 2
    xs = sorted({x for _, _, x, _ in grid})
    ys = sorted({y for _, _, _, y in grid})
    assert xs == [0, 480, 960, 1280]  # 마지막은 1920-640=1280로 클램프
    assert ys == [0, 440]  # 1080-640=440로 클램프


def test_offsets_clamp_and_exact_fit():
    # 딱 떨어지는 경우 중복 없이
    assert tile_offsets(1280, 640, 640) == [0, 640]
    # 타일이 프레임보다 크면 원점 하나
    assert tile_offsets(500, 640, 480) == [0]


def test_overlap_property():
    assert P.overlap == 160  # 25% of 640


def test_validate_rejects_bad_params():
    assert TilingParams(stride=700).validate()  # stride > tile
    assert TilingParams(tile_size=0).validate()
    assert TilingParams(min_visibility=1.5).validate()
    assert not P.validate()


def _ball(cx_px: float, cy_px: float, size_px: float = 40):
    """1920×1080 이미지 위 정사각 박스 (정규화 xyxy)."""
    return (
        0,
        (
            (cx_px - size_px / 2) / 1920,
            (cy_px - size_px / 2) / 1080,
            (cx_px + size_px / 2) / 1920,
            (cy_px + size_px / 2) / 1080,
        ),
    )


def test_fully_inside_box_kept_and_renormalized():
    # 타일 (480, 440) 안 중앙 부근의 공
    boxes = [_ball(700, 700)]
    out = clip_boxes_to_tile(boxes, 1920, 1080, 480, 440, P)
    assert len(out) == 1
    cls, (x1, y1, x2, y2) = out[0]
    # 타일 좌표로 재정규화: (700-480±20)/640, (700-440±20)/640
    assert abs(x1 - (700 - 480 - 20) / 640) < 1e-9
    assert abs(x2 - (700 - 480 + 20) / 640) < 1e-9
    assert abs(y1 - (700 - 440 - 20) / 640) < 1e-9
    assert abs(y2 - (700 - 440 + 20) / 640) < 1e-9


def test_box_below_min_visibility_dropped():
    # 타일 오른쪽 경계(640)에 걸친 공: 절반만 보임 (0.5 < 0.6) → 삭제
    boxes = [_ball(640, 300)]
    out = clip_boxes_to_tile(boxes, 1920, 1080, 0, 0, P)
    assert out == []


def test_box_above_min_visibility_clipped():
    # 70%가 타일 안 → 유지 + 경계로 클립
    boxes = [_ball(640 - 8, 300, size_px=40)]  # 32px/40px = 80% 보임
    out = clip_boxes_to_tile(boxes, 1920, 1080, 0, 0, P)
    assert len(out) == 1
    _, (_, _, x2, _) = out[0]
    assert x2 == 1.0  # 타일 오른쪽 경계로 클립


def test_overlap_guarantees_full_copy_somewhere():
    # 경계에 걸려 한 타일에선 잘려도, 겹침(160px) 덕에 이웃 타일엔 온전히 담긴다
    boxes = [_ball(640, 300, size_px=40)]  # 타일0 경계에 정확히 걸침
    tile0 = clip_boxes_to_tile(boxes, 1920, 1080, 0, 0, P)
    tile1 = clip_boxes_to_tile(boxes, 1920, 1080, 480, 0, P)  # 480~1120 타일
    assert tile0 == []  # 반쪽 → 삭제
    assert len(tile1) == 1  # 온전히 포함
    _, (x1, y1, x2, y2) = tile1[0]
    assert abs((x2 - x1) * 640 - 40) < 1e-6  # 폭 40px 그대로 (클립 안 됨)


def test_min_visibility_override():
    # min_visibility 0.4면 절반 걸친 공도 유지된다
    loose = TilingParams(min_visibility=0.4)
    boxes = [_ball(640, 300)]
    assert len(clip_boxes_to_tile(boxes, 1920, 1080, 0, 0, loose)) == 1


def test_tile_stem_naming():
    assert tile_stem("game_00001", 2, 1) == "game_00001_r1c2"


# ---------------------------------------------------------------------------
# 타일 추론 병합 (detection.merge_tile_boxes)


def test_merge_tile_boxes_dedupes_overlap_region():
    from app.ml.trackcrop.detection import merge_tile_boxes

    # 겹침 구간에서 같은 공이 두 타일에 잡힘 — 온전한(고conf) 쪽이 남는다
    full = (600.0, 300.0, 640.0, 340.0, 0.9)
    half = (602.0, 301.0, 640.0, 339.0, 0.4)  # 경계 반쪽 (IoU 높음)
    other = (1200.0, 500.0, 1240.0, 540.0, 0.8)  # 다른 위치 공
    kept = merge_tile_boxes([half, full, other], iou_threshold=0.5)
    assert full in kept and other in kept and half not in kept


def test_merge_tile_boxes_keeps_distinct_boxes():
    from app.ml.trackcrop.detection import merge_tile_boxes

    a = (0.0, 0.0, 40.0, 40.0, 0.9)
    b = (100.0, 100.0, 140.0, 140.0, 0.3)
    assert len(merge_tile_boxes([a, b], iou_threshold=0.5)) == 2
