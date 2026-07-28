# Classes

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/classes.py`, `app/domain/classes.py`

## 목적

프로젝트별 클래스(`classes.json`)를 add/rename/delete한다. YOLO 라벨이 위치 기반 id를
참조하므로 **contiguous·append-only id 공간**을 유지하고, 삭제는 프로젝트 전역 라벨 reindex를 동반한다.

## 요구사항

- **add**: 다음 contiguous id 부여, blank/중복(대소문자 무시) 거부.
- **rename**: 중복 이름 거부, 없는 id는 404.
- **delete**: 해당 클래스 박스 drop + 상위 id 전부 −1 shift(모든 라벨 파일) + `classes.json` contiguous 재작성 + meta 사이드카 라인 정렬 유지. 응답에 `removed_boxes` 경고 카운트 포함.

## 데이터·저장 구조

- DB: **없음**(Project 존재만 확인, 파일 기반).
- 디스크: `projects_dir/{pid}/classes.json`(`{"classes":[{"id","name","sources"}]}`), reindex 대상 `labels/*.txt`, 정렬 유지 대상 `labels/*.meta.json`.

## API (prefix `/projects/{project_id}/classes`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `.../classes` | 클래스 목록 — `classes.py:31` |
| POST | `.../classes` | 클래스 추가 (201) — `classes.py:37` |
| PATCH | `.../classes/{class_id}` | 이름 변경 — `classes.py:46` |
| DELETE | `.../classes/{class_id}` | 삭제 (`{ok,removed_boxes,classes}`) — `classes.py:59` |

## 처리 흐름

endpoint → `_require_project` → domain 호출.
- add: `add_class`(read_classes + 중복검사 → `write_classes` atomic).
- rename: `rename_class`(KeyError→404, ValueError→422).
- delete: `count_boxes_with_class`(경고 카운트) → `delete_class` → 각 라벨 `_reindex_label_file`(box drop + id shift + meta 정렬) → `write_classes`.

## 핵심 파일·함수

- `app/api/v1/endpoints/classes.py:37` `create_class`, `:46` `patch_class`, `:59` `remove_class`
- `app/domain/classes.py:42` `add_class`, `:59` `rename_class`, `:74` `_reindex_label_file`, `:99` `count_boxes_with_class`, `:117` `delete_class`

## 엣지 케이스·주의

- 삭제는 **파괴적·전역 연산** — 모든 라벨 파일 재작성. 파일별 atomic이나 전체 트랜잭션은 아님(중단 시 부분 정합성 위험).
- `_reindex_label_file`은 5열 미만/비정수 라인 skip → 재작성 시 그런 라인 소실.
- meta 정렬은 라인 index 기반 — txt와 meta 라인 수 불일치 시 `{}`로 채움.
- `classes.json` 손상 시 `read_classes`가 조용히 `[]` 반환 → add가 id=0부터 재시작(id 충돌 위험).
- `app/domain/class_registry.py`(`ClassRegistry`)는 **별개 모델** — 전역 union(dataset 임포트/모델 앙상블용). 프로젝트 로컬 dict인 `classes.py`와 혼동 주의 → [dataset-upload.md](dataset-upload.md).
