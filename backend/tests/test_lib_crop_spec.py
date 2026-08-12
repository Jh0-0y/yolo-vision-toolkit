"""crop.json 계약 스키마(camelCase) 직렬화 검증.

좌표·규격은 adaptive-crop이 내고(`CropResult.to_dict`), 이 툴킷은 계약이 요구하는
id 컬럼을 얹어 파일로 쓴다(`lib.crop.plan.crop_plan_json`). 여기서 보는 것은 그
합성 결과다: schemaVersion·id 컬럼(플레이스홀더)·source/crop 중첩·
keyframes(videoOffsetMs/targetType UPPER)·summary(camelCase).
samples/debug는 내부 확장이라 include_internal=True에서만 담긴다.
"""

import json

from adaptive_crop import CropSpec, Keyframe, TargetSample, build_crop_result

from lib.crop.plan import crop_plan_json

WINDOW = CropSpec(608, 1080).resolve(1920, 1080)


def _result():
    samples = [
        TargetSample(0, 960.0, "ball", 0.9),
        TargetSample(100, 970.0, "ball_player", 0.85),
        TargetSample(200, 980.0, "center", 0.0),
    ]
    keyframes = [
        Keyframe(0, 656, "ball", 0.912345),
        Keyframe(200, 676, "center", 1.0),
    ]
    return build_crop_result(samples, keyframes, WINDOW)


def test_spec_shape_and_camel_case():
    d = json.loads(crop_plan_json(_result()))
    assert d["schemaVersion"] == 1
    # id 컬럼은 플레이스홀더로 존재해야 한다 (이 툴킷엔 게임/잡 체계가 없음)
    for key in ("jobId", "gameId", "clipCandidateId", "sourceContentOutputId", "sourceMediaAssetId"):
        assert isinstance(d[key], str) and d[key]
    assert d["source"] == {"width": 1920, "height": 1080, "durationMs": 200}
    assert d["crop"] == {"width": 608, "height": 1080, "y": 0, "interpolation": "LINEAR"}
    kf = d["keyframes"][0]
    assert set(kf) == {"videoOffsetMs", "x", "targetType", "confidence"}
    assert kf["targetType"] == "BALL"
    assert kf["confidence"] == 0.9123  # 4자리 반올림
    assert d["keyframes"][1]["targetType"] == "CENTER"
    assert set(d["summary"]) == {
        "keyframeCount", "ballTrackingCoverage", "playerTrackingCoverage", "fallbackCoverage",
    }
    # 스펙 외 내부 필드는 파일로 나가는 crop.json에 없어야 한다
    assert "samples" not in d and "debug" not in d
    # 키 이름에 snake_case가 남아있지 않아야 한다 (값의 언더스코어는 무관)
    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _keys(v)

    assert all("_" not in k for k in _keys(d))


def test_internal_fields_only_when_requested():
    d = _result().to_dict(include_internal=True)
    assert len(d["samples"]) == 3
    assert d["samples"][0]["target_center_x"] == 960.0  # 내부 확장은 내부 표기 유지
    assert d["debug"] == []


def test_crop_geometry_follows_the_source_resolution():
    """1920×1080 고정이 아니다 — 720p 소스면 창과 X 범위가 함께 줄어든다."""
    window = CropSpec(406, 720).resolve(1280, 720)
    result = build_crop_result(
        [TargetSample(0, 640.0, "ball", 0.9)], [Keyframe(0, 437, "ball", 0.9)], window
    )
    d = result.to_dict()
    assert d["source"]["width"] == 1280
    assert d["crop"]["width"] == 406
    assert window.x_max == 1280 - 406
