# Label Editor

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/labels.py`, `app/domain/labels.py`, `app/domain/yolo_io.py`

## 목적

라벨 에디터 페이지용 **per-image 라벨(박스) 읽기/저장**. YOLO txt(`labels/{stem}.txt`) +
meta 사이드카(score/status)를 에디터 형태 박스(`{id, cls, xyxy_n}`)로 변환한다.

## 요구사항

- 라벨 파일 존재 = "labeled"(빈 파일 = 의도된 negative).
- 박스 read 시 meta에서 score/status 부착.
- write는 atomic, **5열 YOLO 포맷 유지**(학습 호환), meta는 별도 사이드카.
- `xyxy_n` ↔ YOLO(cxcywh) 변환, 0..1 clamp.
- GET 응답에 classes.json + reviewed 플래그 + image_url 포함.
- stem path-traversal 검증.

## 데이터·저장 구조

- DB: **없음**(Project 존재만 확인, 파일 기반).
- 디스크: `labels/{stem}.txt`(YOLO 5열), `labels/{stem}.meta.json`(라인 정렬 score/status; 전부 비면 삭제), `reviewed.json`, `classes.json`, `raw/{stem}.*`(이미지 존재 확인).

## API (prefix `/projects/{project_id}/labels`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `.../labels/{stem}` | 박스 + classes + reviewed + image_url — `labels.py:49` |
| PUT | `.../labels/{stem}` | 박스 저장 (`{ok,count}`) — `labels.py:67` |

## 처리 흐름

- GET: `get_labels` → `_safe_stem` → `_image_name`(raw glob) → `read_boxes`(read_box_meta + read_label_file) + `_classes` + `read_reviewed`.
- PUT: `put_labels` → `_safe_stem` → 이미지 존재 확인 → `write_boxes` → `write_label_file`(xyxyn_to_yolo + clamp + atomic) + `write_box_meta`(전부 empty면 사이드카 삭제).

## 핵심 파일·함수

- `app/api/v1/endpoints/labels.py:49` `get_labels`, `:67` `put_labels`, `:29` `_safe_stem`, `:35` `_image_name`
- `app/domain/labels.py:36` `read_boxes`, `:57` `write_boxes`, `:69` `label_classes`, `:85` `read_reviewed`, `:103` `set_reviewed`
- `app/domain/yolo_io.py:40` `write_label_file`, `:63` `write_box_meta`, `:84` `read_label_file`, `:96` `write_data_yaml`, `:27` `atomic_write_text`
- schema `app/schemas/label.py` `BoxIn`, `LabelsIn`

## 엣지 케이스·주의

- reviewed 플래그는 GET에만 노출 — 라벨 저장이 reviewed를 자동 갱신하지 않음(리뷰는 `projects.py`의 별도 PUT `/images/{stem}/reviewed`).
- `read_label_file`은 정확히 5열만 파싱(≠5열 skip) — 세그멘테이션/키포인트 라벨 미지원.
- meta 사이드카는 라인 index로 txt와 정렬 — box 수 불일치 시 `{}` fallback.
- `write_box_meta`는 score·status 전부 없으면 사이드카를 **삭제**(고아 meta 방지). 단 [projects-and-storage.md](projects-and-storage.md)의 `delete_images`는 txt만 지워 비대칭.
- `BoxIn`의 reason/sources는 받지만 `write_boxes`는 score/status만 저장(reason/sources silent drop).
- write 시 xyxy 0..1 clamp — 화면 밖 박스는 경계로 잘림.
- `write_data_yaml`은 라벨 에디터가 아니라 export 경로에서 사용 → [export.md](export.md).
