---
title: 크롭 계획 · 튜닝 · crop.json
scope:
  - backend/app/workers/*.py
  - backend/app/api/v1/endpoints/lab_crops.py
  - backend/lib/crop/**
  - frontend/src/components/lab/**
applies_to: 크롭 런의 튜닝 노브·비율·산출물을 다룰 때
related:
  - ./adaptive-crop.md
  - ./adaptive-crop-upgrade.md
---

# 크롭 계획 · 튜닝 · crop.json

> 크롭 런 하나가 검출·좌표·영상 셋을 한 번에 낸다. 노브·비율·산출물을 다룰 때 읽는다.

## 호출 흐름

검출(비쌈, 모델 필요)과 좌표 계산(쌈, 순수 계산)은 라이브러리 안에서 분리되어 있지만,
**툴킷에서는 한 런 안에서 이어 돈다.** 경로가 하나뿐이라 갈래를 외울 것이 없다.

| 단계 | 하는 일 | validate |
|---|---|---|
| `workers/lab_crop_worker` ① | `detect_video` — 100ms 격자 검출 | — |
| `workers/lab_crop_worker` ② | `plan_from_detections` → `crop.json` | **`True`** — 파일로 나가는 결과라 규칙 위반이면 실패시킨다 |
| `workers/lab_crop_worker` ③ | 한 루프로 `crop.mp4`(깨끗) · `wide.mp4`(오버레이 or 원본 링크) | — |

산출물은 임시 폴더가 아니라 **연구실 아래 크롭 런**(`labs/{lab_id}/crops/{crop_id}/`)에 바로 쓴다
— 자리와 수명은 [데이터 배치](../data-layout.md) 를 본다.

**파일로 나가는 경로는 `validate=True` 를 유지한다.**

> 옛 Crop Draw 의 라이브 튜닝(검출 캐시 + 노브만 다시 계산)은 **없어졌다.** 노브를 바꾸면
> 검출부터 런 전체를 다시 돈다. 되살린다면 검출 캐시를 다시 두는 일부터다.

## 크롭 창 크기

크기는 **비율이 아니라 절대 px** 다. 런이 `crop_w`·`crop_h` 를 그대로 정하고(기본 608×1080 —
1080p 소스의 9:16 과 같은 값), 소스가 커진다고 창이 따라 커지지 않는다.

- 계산은 `lib/crop/geometry.crop_window_for(source_w, source_h, crop_w, crop_h)` 하나뿐이고,
  좌표 쪽 `crop_spec_for(...)` 도 같은 함수를 부른다. **양쪽에 같은 값을 넘겨야 한다** —
  어긋나면 크롭 박스가 실제 잘리는 영역과 달라진다.
- 돌려주는 것은 `(width, height, y)` 다. 소스보다 크면 **소스 크기로 clamp** 하고, 결과는 항상
  짝수(libx264/yuv420p)다.
- **세로도 잘린다.** 높이가 소스보다 낮으면 `y` 가 가운데를 잡아 위아래를 잘라낸다. 같으면 `y=0`
  이라 가로만 잘린다 — 그래서 608×1080 은 1080p 에서 예전 9:16 과 정확히 같은 창이다.

실제로 쓰인 창은 `crop.json` 의 `crop` 이 들고 있다. 요청값이 clamp 됐는지는 거기서 드러나므로
**따로 저장하지 않는다** — 목록·상세의 `applied_w`·`applied_h` 가 그 값이다.

## 타깃 타입의 색

`TargetSample` 은 `video_offset_ms`·`target_center_x`·`target_type`·`confidence` 넷뿐이다 —
**공과 선수의 혼합 비율은 내주지 않는다.** 대신 `ball_player` 구간의 혼합비는 그 런의
`ClipPlanConfig.ball_weight` 그대로이므로, 타입 색 + 런 설정으로 읽어 낸다.

색 정의는 `lib/crop/palette.py` **한 곳뿐**이다. 서버 렌더(`lib/crop/hud`)가 직접 쓰고,
프론트는 `GET /system/crop-palette` 로 받아 간다 — 양쪽에 따로 적으면 같은 런이 두 그림이 된다.

## 튜닝 노브

`TuningPanel.tsx` → `overrides` JSON → `resolve_clip_config` → `ClipPlanConfig`.
키 이름은 `ClipPlanConfig` 필드명과 1:1 이다. 노브를 비우면 그 키를 안 보내므로 라이브러리 기본값이 쓰인다 —
**placeholder 에 적는 `def` 값은 `ClipPlanConfig` 의 기본값과 같아야 한다**(다르면 UI 가 거짓말을 한다).

아직 UI 에 노출하지 않은 필드: `keyframe_epsilon_px` · `player_lost_hold_ms` · `absorb_point_scale` · `center_fallback_x`.

## crop.json

라이브러리의 `to_dict()` 는 **영상에서 계산된 값만** 낸다(`schemaVersion`·`source`·`crop`·`keyframes`·`summary`).
이 툴킷에는 게임/잡 식별자 체계가 없어 `crop_plan_json` 이 플레이스홀더 id(`jobId: "job_local"` 등)를 얹어 계약 형태를 맞춘다.

- 이 파일은 **외부 계약 스키마**라 키가 camelCase 다. 백엔드 DTO 의 snake_case 규칙이 여기엔 적용되지 않는다.
- `samples`·`debug` 는 스펙 외 내부 확장이라 파일에는 담기지 않는다(`include_internal=True` 일 때만). 워커는 메모리 안의 `CropResult` 에서 바로 읽어 그린다.
- `summary` 는 이 파일 안에 있다. 크롭 런 목록은 여기서 읽으므로 **커버리지를 따로 저장하지 않는다.**

## 화면은 브라우저에 아무것도 기억하지 않는다

런처는 시작만 하고 `crop_id` 를 받아 상세로 넘어간다. 진행률은 **상세페이지가** SSE 로 받는다
— `progress.jsonl` 을 처음부터 재생하므로 탭을 옮기거나 새로고침해도 이어진다.
전역 잡 카드는 쓰지 않는다 → [잡과 진행률](../conventions/jobs-and-progress.md)

## 보간과 계단

크롭 창의 중심 X 는 **선형 보간**(창이 부드럽게 움직여야 한다), 타깃 타입·신뢰도·디버그 박스는
**계단**(직전 샘플 유지)이다. 검출 박스도 계단이라 최대 한 샘플 간격(기본 100ms)만큼 늦다.
