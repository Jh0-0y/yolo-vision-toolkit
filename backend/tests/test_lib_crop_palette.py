"""타깃 팔레트 — 서버 렌더와 프론트 캔버스가 같은 그림을 그리려면 정의가 하나여야
한다. 여기서는 그 하나가 두 소비자(cv2 BGR · API 응답)에게 같은 값을 주는지 본다."""

import pytest

from adaptive_crop.types import TargetType
from lib.crop import palette


def test_every_target_type_has_a_color():
    """라이브러리가 타입을 늘리면 색 없는 타입이 회색으로 조용히 뭉개진다 — 여기서 잡는다."""
    assert set(palette.TARGET_COLORS) == {t.value for t in TargetType}


def test_hex_to_bgr_reverses_the_channels():
    assert palette.hex_to_bgr("#ff0000") == (0, 0, 255)
    assert palette.hex_to_bgr("#00ff00") == (0, 255, 0)
    assert palette.hex_to_bgr("#0000ff") == (255, 0, 0)


def test_bgr_for_matches_the_hex_definition():
    for name, hexval in palette.TARGET_COLORS.items():
        assert palette.bgr_for(name) == palette.hex_to_bgr(hexval)


def test_an_unknown_type_falls_back_to_center_grey():
    assert palette.bgr_for("something_new") == palette.bgr_for("center")
    assert palette.bgr_for(None) == palette.bgr_for("center")


@pytest.mark.parametrize(
    "confidence,expected",
    [(1.0, 1.0), (0.0, palette.MIN_CONFIDENCE_ALPHA), (None, 1.0)],
)
def test_alpha_spans_from_the_floor_to_full(confidence, expected):
    assert palette.alpha_for(confidence) == pytest.approx(expected)


def test_alpha_never_drops_below_the_floor():
    """0 까지 떨어뜨리면 낮은 신뢰 구간이 "없는 구간" 으로 읽힌다."""
    assert palette.alpha_for(-5) == pytest.approx(palette.MIN_CONFIDENCE_ALPHA)
    assert palette.alpha_for(5) == pytest.approx(1.0)


def test_as_dict_carries_what_the_frontend_needs():
    """프론트가 색을 따로 적지 않으려면 이 응답만으로 충분해야 한다."""
    data = palette.as_dict()
    assert data["target_colors"] == palette.TARGET_COLORS
    assert data["target_line_color"] == palette.TARGET_LINE_COLOR
    assert data["dead_zone_color"] == palette.DEAD_ZONE_COLOR
    assert data["min_confidence_alpha"] == palette.MIN_CONFIDENCE_ALPHA


def test_as_dict_is_a_copy():
    """응답을 만지작거려도 원본 정의가 흔들리면 안 된다."""
    palette.as_dict()["target_colors"]["ball"] = "#000000"
    assert palette.TARGET_COLORS["ball"] != "#000000"
