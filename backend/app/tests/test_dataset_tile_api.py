"""타일링 엔드포인트의 순수 부분 — 이름 기본값과 파라미터 검증.

저장소에 HTTP 테스트 하네스가 없어(`TestClient` 도 공용 conftest 도 없다) 라우팅은
`openapi.json` 확인과 손 확인이 맡는다. 여기서는 HTTP 없이 부를 수 있는 것만 본다.
"""

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.dataset_tile import _derived_name, _params
from app.schemas.dataset import TileParamsIn


def test_derived_name_defaults_to_source_plus_tile_size():
    assert _derived_name("", "경기 1차", 640) == "경기 1차 (tiled 640)"


def test_derived_name_respects_an_explicit_name():
    assert _derived_name("  내 타일셋  ", "경기 1차", 640) == "내 타일셋"


def test_params_rejects_stride_bigger_than_tile():
    """겹침이 음수면 타일 경계에 걸린 객체가 양쪽에 반쪽씩 남는다."""
    with pytest.raises(HTTPException) as e:
        _params(TileParamsIn(tile_size=640, stride=700))
    assert e.value.status_code == 422


def test_params_rejects_negative_ratio_below_zero():
    with pytest.raises(HTTPException) as e:
        _params(TileParamsIn(negative_ratio=-0.5))
    assert e.value.status_code == 422


def test_params_passes_defaults_through():
    p = _params(TileParamsIn())
    assert (p.tile_size, p.stride, p.min_visibility) == (640, 480, 0.6)
    assert p.negative_ratio == 0.1
    assert p.keep_all_negatives is False
