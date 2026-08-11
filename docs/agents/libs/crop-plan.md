---
title: 크롭 계획 · 튜닝 · crop.json
scope:
  - backend/app/workers/*.py
  - backend/app/api/v1/endpoints/predict.py
  - frontend/src/components/lab/**
applies_to: 라이브 프리뷰·튜닝 노브·crop.json 산출물을 다룰 때
related:
  - ./adaptive-crop.md
  - ./adaptive-crop-upgrade.md
---

# 크롭 계획 · 튜닝 · crop.json

> 검출과 좌표 계산이 분리돼 있고, 경로마다 검증 강도가 다르다. 라이브 프리뷰나 산출물을 다룰 때 읽는다.

## 호출 흐름과 `validate`

검출(비쌈, 모델 필요)과 좌표 계산(쌈, 순수 계산)이 분리되어 있다.
**라이브 튜닝이 즉시 반영되는 것은 이 분리 덕분이다** — 노브를 돌려도 추론을 다시 하지 않는다.

| 경로 | 하는 일 | validate |
|---|---|---|
| `workers/live_worker.run_live` | `detect_video` → `detected.json`·`meta.json` 캐시 + preview.mp4 | — |
| `api/…/live/{id}/plan` | 캐시 + 현재 노브로 `plan_from_detections` (동기, 추론 없음) | `False` — 노브 조합 하나가 500이 되면 안 된다 |
| `workers/live_worker.run_live_render` | 같은 좌표를 preview.mp4 위에 굽는다 | `False` — 미리보기 |
| `workers/annotate_worker` | `detect_video` → `plan_from_detections` → crop.json + 영상 | **`True`** — 파일로 나가는 결과라 규칙 위반이면 실패시킨다 |

**파일로 나가는 경로는 `validate=True` 를 유지한다.** 미리보기용으로 껐다고 해서 산출 경로까지 끄지 않는다.

## 튜닝 노브

`TuningPanel.tsx` → `overrides` JSON → `resolve_clip_config` → `ClipPlanConfig`.
키 이름은 `ClipPlanConfig` 필드명과 1:1 이다. 노브를 비우면 그 키를 안 보내므로 라이브러리 기본값이 쓰인다 —
**placeholder 에 적는 `def` 값은 `ClipPlanConfig` 의 기본값과 같아야 한다**(다르면 UI 가 거짓말을 한다).

아직 UI 에 노출하지 않은 필드: `keyframe_epsilon_px` · `player_lost_hold_ms` · `absorb_point_scale` · `center_fallback_x`.

## crop.json

라이브러리의 `to_dict()` 는 **영상에서 계산된 값만** 낸다(`schemaVersion`·`source`·`crop`·`keyframes`·`summary`).
이 툴킷에는 게임/잡 식별자 체계가 없어 `crop_plan_json` 이 플레이스홀더 id(`jobId: "job_local"` 등)를 얹어 계약 형태를 맞춘다.

- 이 파일은 **외부 계약 스키마**라 키가 camelCase 다. 백엔드 DTO 의 snake_case 규칙이 여기엔 적용되지 않는다.
- `samples`·`debug` 는 스펙 외 내부 확장이라 파일에는 담기지 않는다. 라이브 오버레이용 `/plan` 응답에만 `include_internal=True` 로 담긴다.

## 알려진 차이

라이브 캔버스 오버레이는 100ms 검출 격자를 시간축 보간해 그리지만, 서버 렌더(`run_live_render`)의 검출 박스는
직전 샘플을 그대로 쓴다(step) — 두 화면의 박스가 최대 100ms 어긋난다. 크롭 창 자체는 양쪽 다 보간이라 일치한다.
