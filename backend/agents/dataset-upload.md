# Dataset Upload (외부 YOLO zip 임포트)

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/projects.py`(`import_dataset_zip`), `app/domain/class_registry.py`, `app/domain/yolo_io.py`

## 목적

이미 라벨링된 외부 YOLO 포맷 데이터셋(zip: images + labels + data.yaml)을 프로젝트의
`raw/`·`labels/`로 임포트해 데이터셋 뷰에 노출한다. 로컬 클래스 id를 프로젝트 전역 id로 remap한다.

> **주의**: 이 기능의 실제 구현은 `endpoints/projects.py`의 `import_dataset_zip`이다.
> `app/domain/datasets.py`는 이름과 달리 **학습용 업로드 데이터셋 라이프사이클**(auto_delete/cleanup)용으로
> 이 임포트 경로와 무관 → [training.md](training.md).

## 요구사항

- `data.yaml`(list/dict `names` 양식) 필수 — 없으면 오류.
- 클래스를 `ClassRegistry`로 프로젝트 `classes.json`에 이름 정규화 병합(append-only).
- 라벨 파일의 class id를 local→global 매핑으로 재작성.
- 이미지 stem 기준 라벨 매칭, atomic write.
- `.zip`만 허용, 손상 zip은 오류.

## 데이터·저장 구조

- DB: **없음**(Project 존재만 확인).
- 디스크: 임시 `projects_dir/{pid}/.import_{hex}.zip`(finally 삭제) → `TemporaryDirectory` extractall → `raw/` + `labels/` + `classes.json`.
- `classes.json` 스키마: `{"classes":[{"id","name","sources"}]}`.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/projects/{project_id}/dataset-zip` | 라벨된 YOLO zip 임포트 (201, `{images,labeled,classes}`) — `projects.py:237` |

## 처리 흐름

`import_dataset_zip`(.zip 검증 → 임시 zip 저장) → `run_in_threadpool(_import_labeled_zip, ...)`
→ `_import_labeled_zip`(`projects.py:185`): `extractall` → `_read_yaml_names`(data.yaml rglob)
→ `ClassRegistry.from_dict`(기존 classes.json) or 새 registry → `registry.add_model("import", names)`로 local→global mapping
→ `labels/*.txt`를 stem으로 keying, 이미지 rglob 복사(`shutil.copyfile`) + `_remap_label_text`로 id remap + `atomic_write_text`
→ `registry.to_dict()` atomic write.

## 핵심 파일·함수

- `app/api/v1/endpoints/projects.py:237` `import_dataset_zip`, `:185` `_import_labeled_zip`, `:150` `_read_yaml_names`, `:169` `_remap_label_text`
- `app/domain/class_registry.py:28` `ClassRegistry.add_model`(이름 정규화 union), `:57` `from_dict`(contiguous id 검증)
- `app/domain/yolo_io.py:27` `atomic_write_text`

## 엣지 케이스·주의

- `_read_yaml_names`는 `data.yaml`/`data.yml`만 탐색, 첫 매치 사용. `names`는 dict/list 지원, 그 외는 빈 dict → 라벨 id 그대로 유지.
- `_remap_label_text`는 5열 미만/비정수 class 라인을 **silent drop** → 세그멘테이션 폴리곤 라벨은 소실.
- 이미지는 `shutil.copyfile` → raw/에 같은 이름 존재 시 **덮어씀**(파일명 충돌 위험). 라벨 매칭도 stem만 → 다른 하위 폴더 동일 stem 충돌 가능.
- meta.json 사이드카 생성 안 함(임포트 라벨은 score/status 없음).
- 기존 `classes.json`이 non-contiguous면 `from_dict`가 ValueError → 임포트 실패.
- 클래스 id 공간·reindex 규칙은 [classes.md](classes.md) 참고.
