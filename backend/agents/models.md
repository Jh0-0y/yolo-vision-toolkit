# Models Registry

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/models.py`, `app/models/__init__.py`(`ModelEntry`)

## 목적

학습된 `.pt` 업로드 또는 공식 사전학습 모델 다운로드로 모델을 등록·조회·다운로드·삭제한다.
클래스 이름/태스크는 ultralytics로 로드해 검증·추출한다.

## 요구사항

- `.pt`만 허용(업로드 시 확장자 검증).
- 등록 시 ultralytics로 로드해 유효성 검증 + `model.names`/`task` 추출(실패 시 422).
- 프로젝트 스코프(`project_id`) 또는 공유/legacy(`project_id=None`, 모든 프로젝트에 노출).
- 공식 카탈로그는 큐레이션된 families(yolo26/12/11) × sizes(n/s/m/l/x) 중 ultralytics `GITHUB_ASSETS_NAMES`에 존재하는 것만.
- 파일 삭제 시 DB row + 디스크 디렉터리 함께 제거.

## 데이터·저장 구조

- DB: `model_entry`(`ModelEntry`: id, name, project_id(index, None=공유), classes_json, task, created_at).
- 디스크: `model_dir(project_id, model_id)`(스코프 `projects/{pid}/models/{mid}`, 공유 `models/{mid}`) 각 디렉터리에 `model.pt` + `meta.json`. → [storage-layout.md](storage-layout.md).

## API (prefix `/models`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/models` | 목록(project_id 필터 시 해당+공유, 최신순) — `models.py:73` |
| GET | `/models/official` | 다운로드 가능 공식 카탈로그 — `models.py:85` |
| POST | `/models/official` | 공식 모델 다운로드·등록 (201, 실패 502) — `models.py:99` |
| POST | `/models` | `.pt` 업로드·등록 (201) — `models.py:120` |
| GET | `/models/{model_id}` | 단건 |
| GET | `/models/{model_id}/download` | model.pt 다운로드 |
| PATCH | `/models/{model_id}` | 이름 변경(DB+meta.json) |
| DELETE | `/models/{model_id}` | 삭제 (204, DB+디스크) — `models.py:191` |

## 처리 흐름

등록 공통 `_register`(`_load_names(pt)` ultralytics lazy import → `ModelEntry` insert → `shutil.move`로 pt를 `model_dir`로 이동 + meta.json).
- 업로드: `upload_model`이 1MB 청크로 임시 파일 기록 후 `_register(source="upload")`.
- 공식: `download_official`이 `run_in_threadpool(_download)`(`attempt_download_asset`) 후 `_register(source="official")`.
- training의 `register_weights`가 `_register(source="trained")` 재사용(weights 복사본 전달) → [training.md](training.md).
- 워커 없음(엔드포인트+DB만). ultralytics import는 함수 내부 lazy.

## 핵심 파일·함수

- `app/api/v1/endpoints/models.py:51` `_register`, `:43` `_load_names`, `:73` `list_models`, `:85` `official_catalog`, `:99` `download_official`, `:120` `upload_model`, `:191` `delete_model`
- `app/models/__init__.py:29` `ModelEntry`, `:11` `iso_utc`

## 엣지 케이스·주의

- `_register`는 입력 pt를 **move**(원본 소멸) — training `register_weights`는 복사본을 넘겨 런 weights 보존.
- `list_models`는 `or_(project_id==X, project_id IS NULL)`로 공유 모델 포함.
- source는 DB가 아닌 `meta.json`에서 읽음(`_to_out`); meta 없으면 기본 "upload".
- 삭제는 DB commit 후 `shutil.rmtree(..., ignore_errors=True)`.
- SQLite tz drop은 `iso_utc`로 보정.
