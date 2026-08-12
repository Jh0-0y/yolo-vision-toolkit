---
title: adaptive-crop
scope:
  - backend/lib/crop/plan.py
  - backend/app/workers/*.py
  - backend/lib/crop/**
  - frontend/src/components/lab/**
applies_to: 크롭 좌표 계산·어댑터·오버레이를 다룰 때
related:
  - ./crop-plan.md
  - ./adaptive-crop-upgrade.md
  - ../conventions/layer-boundaries.md
---

# adaptive-crop

> 세로 크롭 좌표 계산은 이 저장소에 없다. 외부 패키지가 하고 툴킷은 부르기만 한다. 크롭 코드를 만질 때 읽는다.

운영 CropWorker와 이 툴킷이 **같은 코드로 같은 좌표**를 내야 여기서 맞춘 튜닝을 믿을 수 있다.
둘 중 하나에만 알고리즘 사본이 있으면 "여기선 되는데 저기선 안 된다"가 반복된다.

## 경계

| | adaptive-crop | 툴킷 |
|---|---|---|
| 공·선수 검출, 궤적 조립, 크롭 X 좌표 | ✅ | ❌ **사본을 두지 않는다** |
| 크롭 창·데드존·박스 그리기, 세로 컷 | ❌ | ✅ `lib/crop/` · `liveOverlay.ts` |
| 잡 관리·진행률·취소·파일 저장 | ❌ | ✅ `workers/` · `services/` |
| 잡 ID·게임 ID 같은 비즈니스 필드 | ❌ | ✅ `crop_plan_json` 이 얹는다 |

**크롭 계산 로직을 이 저장소에 추가하지 않는다.** 알고리즘을 고쳐야 하면 adaptive-crop 에서 고치고 버전을 올린다 → [업그레이드](adaptive-crop-upgrade.md)

## 어댑터 — `lib/crop/plan.py`

라이브러리가 받는 형태와 툴킷/UI 가 쓰는 형태가 달라서 **번역만 하는 얇은 층**을 둔다.

| 함수 | 하는 일 |
|---|---|
| `resolve_clip_config(overrides)` | UI 노브 dict → `ClipPlanConfig`. 모르는 키·`None` 은 버린다(라이브러리는 오타에 `TypeError`) |
| `crop_spec_for(w, h)` | 실제 해상도 → `CropSpec`. 폭은 `lib.crop.geometry.crop_width_for` 와 **같은 규칙** |
| `video_info_from_meta(meta)` | 라이브 세션 `meta.json` → `VideoInfo` |
| `detector_entries(entries, conf)` | `conf` 기본값 채우기, `merge_iou` → `tile_merge_iou`, **첫 full 엔트리만** `track_ids=True` |
| `crop_plan_json(result)` | `CropResult` → crop.json 문자열 (계약용 플레이스홀더 id 를 얹는다) |

`track_ids` 를 어댑터가 정하는 이유: 라이브러리는 track_id 를 부여하는 검출기를 **하나만** 허용한다
(둘이면 서로 다른 객체가 같은 id 를 갖는다). UI 는 모델을 여러 개 고를 수 있어서 첫 full 엔트리에만 추적을 맡긴다.

## 좌표계 — 고정 해상도가 없다

좌표는 **검출 당시 원본 영상 해상도** 기준이다. 1920 고정이 아니라 `VideoInfo` 에서 나오므로
720p 든 4K 든 그대로 동작한다(1080p → 608px 창 / 720p → 404px 창).

프리뷰 영상이 다른 크기로 트랜스코딩됐다면 **그리는 쪽이** 스케일한다(`scale = preview_width / source_width`).
좌표 자체를 프리뷰 기준으로 계산하지 **않는다.**

## 다음

호출 흐름별 검증 강도, 튜닝 노브, crop.json 산출 형태는 [크롭 계획 · 튜닝 · crop.json](crop-plan.md) 에 있다.
