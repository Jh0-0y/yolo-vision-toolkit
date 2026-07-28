# Projects & Storage

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/projects.py`, `app/api/v1/endpoints/files.py`, `app/domain/thumbnails.py`

## 목적

프로젝트를 최상위 워크스페이스 단위로 CRUD하고, 각 프로젝트의 raw 이미지
업로드/목록/삭제/리뷰 플래그/통계를 제공한다. 프로젝트별 온디스크 버킷과 DB row의
생성·정리를 함께 관리한다.

## 요구사항

- 프로젝트 생성 시 표준 하위 디렉터리 스캐폴딩(`PROJECT_SUBDIRS`) + `project.json` 사이드카 작성.
- 프로젝트 삭제는 DB(`Project` + scoped `TrainRun`/`ModelEntry`) + 온디스크 디렉터리 + jobs 로그를 캐스케이드 정리. 공유(`project_id=None`) 리소스는 건드리지 않음.
- 이미지 업로드는 개별 이미지 및 zip(이미지만 추출) 지원, 확장자 화이트리스트(`IMAGE_EXTS`).
- 통합 이미지 목록: labeled/reviewed/cls 필터, created/name 정렬, 이름 검색(`q`), 페이지네이션, `names_only` 모드.
- Path traversal 방지(파일명 base만 취함, `.` prefix 거부).

## 데이터·저장 구조

- DB: `Project`(id, name, created_at). 삭제 캐스케이드로 `TrainRun`/`ModelEntry`(scoped).
- 디스크: `projects_dir/{project_id}/` 하위 `raw/`, `thumbs/`, `videos/`, `labels/`, `exports/` (`projects.py:41`). 사이드카: `project.json`, `classes.json`, `reviewed.json`.

## API (prefix `/projects`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/projects` | 프로젝트 목록 (created desc) — `projects.py:69` |
| POST | `/projects` | 프로젝트 생성 (201, 디렉터리 스캐폴딩) — `projects.py:75` |
| DELETE | `/projects/{project_id}` | 프로젝트 삭제 (204, DB+디스크 캐스케이드) — `projects.py:89` |
| POST | `/projects/{project_id}/images` | 이미지/zip 업로드 (201, `{added,skipped}`) — `projects.py:105` |
| POST | `/projects/{project_id}/dataset-zip` | 외부 YOLO zip 임포트 → [dataset-upload.md](dataset-upload.md) — `projects.py:237` |
| GET | `/projects/{project_id}/images` | 통합 이미지 목록/필터/정렬/페이지 — `projects.py:259` |
| DELETE | `/projects/{project_id}/images` | 벌크 이미지 삭제 (200, `{deleted}`) — `projects.py:339` |
| PUT | `/projects/{project_id}/images/{stem}/reviewed` | 리뷰 플래그 설정 — `projects.py:367` |
| GET | `/projects/{project_id}/stats` | 이미지/labeled/reviewed/classes 통계 — `projects.py:384` |

썸네일·원본 이미지 서빙은 별도 `files.py` 라우터(`raw_image`, `thumbnail`, path-traversal 가드 `_safe_name`).

## 처리 흐름

- 목록: `list_images` → `_raw_images`(raw/ 스캔) + `_labeled_stems`(labels/*.txt) + `read_reviewed`(domain/labels) + `label_classes`(cls 필터) + `read_boxes`.
- 삭제: `delete_images` → `read_reviewed`/`write_reviewed` 갱신 + label txt·thumb unlink.
- 통계: `project_stats` → classes.json + `_raw_images` + `_labeled_stems` + `read_reviewed`.
- 썸네일: `files.py` → `thumbnails.get_thumbnail`(256px JPEG, mtime 캐시).

## 핵심 파일·함수

- `app/api/v1/endpoints/projects.py:44` `_project_dir`, `:55` `_raw_images`, `:62` `_labeled_stems`, `:75` `create_project`, `:89` `delete_project`, `:259` `list_images`
- `app/domain/thumbnails.py:8` `get_thumbnail`
- `app/api/v1/endpoints/files.py:23` `raw_image`, `:31` `thumbnail`

## 엣지 케이스·주의

- `delete_images`는 label txt는 지우지만 `labels/*.meta.json` 사이드카는 **명시적으로 지우지 않음**(고아 meta 가능, `projects.py:358` 부근) — [label-editor.md](label-editor.md)의 `write_box_meta`(빈 meta 삭제)와 비대칭.
- `list_images`의 `cls=-1`은 "labeled but empty(negative)" 특수값(`projects.py:290`).
- 정렬 시 매 이미지 `st_mtime` 호출 → 대량 이미지에서 IO 비용.
- `app/domain/staging.py`, `app/domain/datasets.py`는 이름과 달리 **프로젝트 스토리지가 아니라 학습 데이터셋 스테이징/업로드 라이프사이클**용 → [training.md](training.md) 소속.
- SQLite tz drop은 `iso_utc`로 보정(`models/__init__.py:11`).
