"""adaptive-crop 어댑터 — 툴킷의 요청 형태를 라이브러리 호출로 옮긴다.

크롭 좌표 계산(공·선수 추적 → 세로 크롭 X)은 전부 `adaptive_crop` 라이브러리 안에
있다. CropWorker와 **같은 로직**을 쓰기 위해 툴킷은 알고리즘을 따로 갖지 않는다 —
이 모듈이 하는 일은 "UI가 보내는 것"과 "라이브러리가 받는 것" 사이의 번역뿐이다:

    overrides(dict)      → ClipPlanConfig   (모르는 키·빈 값은 버린다)
    detectors(엔트리)     → 라이브러리 엔트리 (conf 기본값·IoU 이름·track_ids 정리)
    실제 영상 해상도       → CropSpec        (고정 1920/608 상수는 더 이상 없다)

크롭 창 폭은 렌더러(`lib.crop`)와 같은 규칙으로 정한다 — 좌표를 낸
쪽과 화면에 그리는 쪽이 어긋나면 크롭 박스가 실제 잘리는 영역과 달라진다.
"""

from __future__ import annotations

import json
from dataclasses import fields

from adaptive_crop import ClipPlanConfig, CropSpec, VideoInfo

from lib.crop.geometry import crop_width_for

# 공 검출 기본 임계값 — 낮게 잡고 트랙 단계에서 거른다 (라이브러리는 기본값을 두지 않는다).
DEFAULT_CROP_CONF = 0.10

_CONFIG_FIELDS = {f.name for f in fields(ClipPlanConfig)}

# 이 툴킷에는 게임/잡 식별자 체계가 없다 — crop.json 계약의 필수 id는 플레이스홀더로
# 채운다. 라이브러리는 비즈니스 필드를 모르므로(좌표만 낸다) 여기서 얹는다.
SPEC_PLACEHOLDER_IDS = {
    "jobId": "job_local",
    "gameId": "game_local",
    "clipCandidateId": "clip_local",
    "sourceContentOutputId": "cnto_local",
    "sourceMediaAssetId": "mast_local",
}


def resolve_clip_config(overrides: dict | None) -> ClipPlanConfig:
    """튜닝 오버라이드(dict) → ClipPlanConfig.

    알려진 필드·비어있지 않은 값만 반영한다. 라이브러리는 모르는 키에 TypeError를
    내지만, 프론트는 빈 노브를 None으로 보내거나 구버전 키를 남겨둘 수 있어서
    여기서 걸러낸다 (UI 오타가 500이 되지 않게).
    """
    if not overrides:
        return ClipPlanConfig()
    clean = {k: v for k, v in overrides.items() if k in _CONFIG_FIELDS and v is not None}
    return ClipPlanConfig(**clean)


def crop_spec_for(source_width: int, source_height: int) -> CropSpec:
    """소스 해상도에 맞는 세로 9:16 크롭 창 규격.

    폭은 `lib.crop.geometry.crop_width_for`와 같은 값이다 — 좌표(라이브러리)와 렌더링
    (cv2)이 같은 창을 가리키게 하려면 규칙이 하나여야 한다.
    """
    return CropSpec(crop_width_for(source_height, source_width), source_height)


def video_info_from_meta(meta: dict) -> VideoInfo:
    """라이브 세션 meta.json → VideoInfo (검출 당시 원본 영상 규격).

    좌표는 프리뷰가 아니라 **원본 해상도** 기준으로 계산해야 한다 — 프리뷰는
    트랜스코딩 과정에서 크기가 달라질 수 있고, 검출 좌표는 원본 기준이다.
    """
    return VideoInfo(
        width=int(meta["source_width"]),
        height=int(meta["source_height"]),
        fps=float(meta.get("fps") or 0.0),
        duration_ms=int(meta.get("duration_ms") or 0),
    )


def detector_entries(entries: list[dict], default_conf: float = DEFAULT_CROP_CONF) -> list[dict]:
    """툴킷 검출기 엔트리 → 라이브러리 엔트리.

    번역하는 것은 셋이다:

      conf        UI는 비워둘 수 있지만 라이브러리는 엔트리마다 필수 — 기본값을 채운다.
      merge_iou   라이브러리는 층을 나눠 부른다 (타일 경계 병합 = tile_merge_iou,
                  모델 간 병합 = build_detector의 model_merge_iou).
      track_ids   라이브러리는 track_id를 부여하는 검출기를 하나만 허용한다(둘이면
                  서로 다른 객체가 같은 id를 갖는다). 툴킷의 기존 규칙대로 **첫 full
                  엔트리**만 추적을 맡고 나머지는 predict로 돌린다.
    """
    out: list[dict] = []
    tracked_assigned = False
    for e in entries:
        mode = e.get("mode", "full")
        conf = e.get("conf")
        entry: dict = {
            "pt": e["pt"],
            "mode": mode,
            "conf": float(conf) if conf is not None else float(default_conf),
            "track_ids": mode == "full" and not tracked_assigned,
        }
        if entry["track_ids"]:
            tracked_assigned = True
        if mode == "tiled":
            entry["tile_size"] = int(e.get("tile_size", 640))
            entry["stride"] = int(e.get("stride", 480))
            entry["tile_merge_iou"] = float(e.get("merge_iou", 0.5))
        else:
            entry["imgsz"] = int(e.get("imgsz", 1920))
        out.append(entry)
    return out


def crop_plan_json(result, indent: int | None = 2) -> str:
    """CropResult → crop.json 문자열 (툴킷 계약: 플레이스홀더 id + 좌표·규격)."""
    return json.dumps(
        {**SPEC_PLACEHOLDER_IDS, **result.to_dict()}, indent=indent, ensure_ascii=False
    )
