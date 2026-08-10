# 크롭 파이프라인 — adaptive-crop 통합

세로 크롭 좌표 계산은 이 저장소에 없다. 별도 라이브러리
[adaptive-crop](https://github.com/Jh0-0y/adaptive-crop)이 하고, 툴킷은 그것을 부른다.

CropWorker(운영)와 이 툴킷(실험)이 **같은 코드로 같은 좌표**를 내야 튜닝 결과를 믿을 수
있기 때문이다. 둘 중 하나에만 알고리즘 사본이 있으면 "여기선 잘 되는데 저기선 안 된다"가
반복된다.

```
adaptive-crop (외부 패키지)          이 저장소
─────────────────────────           ──────────────────────────────
검출 → 궤적 조립 → 좌표 계산    →   렌더링(cv2·캔버스) · 저장 · API · UI
                                     app/ml/crop.py 가 둘을 잇는다
```

## 경계 — 누가 무엇을 하나

| | adaptive-crop | 툴킷 |
|---|---|---|
| 공·선수 검출, 궤적 조립, 크롭 X 좌표 | ✅ | ❌ (사본을 두지 않는다) |
| 크롭 창·데드존·박스 **그리기**, 세로 컷 | ❌ (라이브러리는 렌더링을 안 한다) | ✅ `app/domain/crop_render.py`, `liveOverlay.ts` |
| 잡 관리·진행률·취소·파일 저장 | ❌ | ✅ `app/workers/`, `app/services/` |
| 잡 ID·게임 ID 같은 비즈니스 필드 | ❌ | ✅ `crop_plan_json`이 얹는다 |

**크롭 계산 로직을 이 저장소에 추가하지 말 것.** 알고리즘을 고쳐야 하면 adaptive-crop에서
고치고 버전을 올린다(아래 "업그레이드" 참고).

## 어댑터 — `app/ml/crop.py`

라이브러리가 받는 형태와 툴킷/UI가 쓰는 형태가 달라서, 번역만 하는 얇은 층을 둔다.

| 함수 | 하는 일 |
|---|---|
| `resolve_clip_config(overrides)` | UI 노브 dict → `ClipPlanConfig`. 모르는 키·`None`은 버린다(라이브러리는 오타에 `TypeError`) |
| `crop_spec_for(w, h)` | 실제 해상도 → `CropSpec`. 폭은 `crop_render.crop_width_for`와 **같은 규칙**(좌표와 그림이 어긋나지 않게) |
| `video_info_from_meta(meta)` | 라이브 세션 `meta.json` → `VideoInfo` (좌표는 항상 **원본 해상도** 기준) |
| `detector_entries(entries, conf)` | `conf` 기본값 채우기, `merge_iou` → `tile_merge_iou`, **첫 full 엔트리만** `track_ids=True` |
| `crop_plan_json(result)` | `CropResult` → crop.json 문자열 (계약용 플레이스홀더 id를 얹음) |

`track_ids`를 어댑터가 정하는 이유: 라이브러리는 track_id를 부여하는 검출기를 **하나만**
허용한다(둘이면 서로 다른 객체가 같은 id를 갖는다). 툴킷 UI는 모델을 여러 개 고를 수 있어서,
기존 규칙대로 첫 full 엔트리에만 추적을 맡기고 나머지는 predict로 돌린다.

## 호출 흐름

검출(비쌈, 모델 필요)과 좌표 계산(쌈, 순수 계산)이 분리되어 있다. 라이브 튜닝이 즉시
반영되는 것은 이 분리 덕분이다 — 노브를 돌려도 추론을 다시 하지 않는다.

| 경로 | 하는 일 | validate |
|---|---|---|
| `workers/live_worker.run_live` | `detect_video` → `detected.json`·`meta.json` 캐시 + preview.mp4 | — |
| `api/…/live/{id}/plan` | 캐시 + 현재 노브로 `plan_from_detections` (동기, 추론 없음) | `False` — 노브 조합 하나가 500이 되면 안 되므로 |
| `workers/live_worker.run_live_render` | 같은 좌표를 preview.mp4 위에 구워 render.mp4 | `False` — 미리보기 |
| `workers/annotate_worker` | `detect_video` → `plan_from_detections` → crop.json + 영상 | **`True`** — 파일로 나가는 결과라 규칙 위반이면 실패시킨다 |

## 좌표계 — 고정 해상도가 없다

좌표는 **검출 당시 원본 영상 해상도** 기준이다. 1920 고정이 아니라 `VideoInfo`에서 나오므로
720p든 4K든 그대로 동작하고, 크롭 창 폭·X 범위가 해상도에서 파생된다
(1080p → 608px 창, x_max 1312 / 720p → 404px 창, x_max 876).

프리뷰 영상이 다른 크기로 트랜스코딩됐다면 **그리는 쪽이** 스케일한다
(`scale = preview_width / source_width`). 좌표 자체는 절대 프리뷰 기준으로 계산하지 않는다.

## 튜닝 노브

UI(`TuningPanel.tsx`) → `overrides` JSON → `resolve_clip_config` → `ClipPlanConfig`.
키 이름은 `ClipPlanConfig` 필드명과 1:1이다. 노브를 비우면 그 키를 안 보내므로 라이브러리
기본값이 쓰인다 — **placeholder에 적는 `def` 값은 `ClipPlanConfig`의 기본값과 같아야 한다**
(다르면 UI가 거짓말을 한다).

전체 필드와 의미는 [adaptive-crop README의 ClipPlanConfig 절](https://github.com/Jh0-0y/adaptive-crop#튜닝--clipplanconfig)에 있다.
아직 UI에 노출하지 않은 필드: `keyframe_epsilon_px`, `player_lost_hold_ms`,
`absorb_point_scale`, `center_fallback_x`.

## crop.json

라이브러리의 `to_dict()`는 **영상에서 계산된 값만** 낸다(`schemaVersion`·`source`·`crop`·
`keyframes`·`summary`). 이 툴킷에는 게임/잡 식별자 체계가 없으므로 `crop_plan_json`이
플레이스홀더 id(`jobId: "job_local"` 등)를 얹어 계약 형태를 맞춘다.
`samples`·`debug`는 스펙 외 내부 확장이라 파일에는 담기지 않고, 라이브 오버레이용
`/plan` 응답에만 `include_internal=True`로 담긴다.

## 설치 · 업그레이드

비공개 저장소라 git 인증이 필요하다. 버전은 `backend/pyproject.toml`에 태그로 고정한다:

```toml
"adaptive-crop @ git+https://github.com/Jh0-0y/adaptive-crop@v0.1.0",
```

- **로컬**: SSH 키로 clone 중이면 한 번만 `git config --global url."git@github.com:".insteadOf "https://github.com/"`
- **Docker**: 토큰을 URL에 박지 말 것(`.dist-info/direct_url.json`에 남는다). `--secret id=gh_pat`로 주입 → [docker/backend.Dockerfile](../docker/backend.Dockerfile) 참고

업그레이드 절차:

1. adaptive-crop에서 고치고 태그를 올린다
2. `backend/pyproject.toml`의 태그를 바꾸고 `uv sync`
3. `ClipPlanConfig` 필드가 바뀌었으면 `TuningPanel.tsx`의 노브 목록·`def` 값과
   `client.ts`의 `TrackcropOverrides` 타입을 맞춘다
4. `uv run pytest` — 어댑터 계약 테스트(`test_crop_adapter.py`, `test_crop_spec.py`)가
   번역 규칙이 깨졌는지 잡는다

알고리즘 자체의 테스트는 adaptive-crop 저장소에 있다. 여기서 중복해 검증하지 않는다.

## 알려진 차이·주의

- 패키지 설치 시 `opencv-python-headless`가 함께 들어와 `cv2`가 그 버전으로 확정된다.
  워커는 GUI 함수를 쓰지 않아 문제없지만, `cv2.imshow` 류를 새로 쓰면 깨진다.
- 라이브 캔버스 오버레이는 100ms 검출 격자를 시간축 보간해 그리지만, 서버 렌더
  (`run_live_render`)의 검출 박스는 직전 샘플을 그대로 쓴다(step) — 두 화면의 박스가
  최대 100ms 어긋난다. 크롭 창 자체는 양쪽 다 보간이라 일치한다.
