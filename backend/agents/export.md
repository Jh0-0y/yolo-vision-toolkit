# Export

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/exports.py`, `app/services/export_manager.py`, `app/domain/export_build.py`

## 목적

라벨된 이미지를 train/val 스플릿 + `data.yaml`로 묶은 YOLO zip, 또는 원본 이미지만 담은 zip을
백그라운드 잡으로 빌드한다.

## 요구사항

- `kind="yolo"`는 라벨(`labels/{stem}.txt`) 존재 이미지만 대상. `kind="images"`는 모든 원본.
- 선택 파일 리스트(`names`)로 제한 가능.
- 동기 검증 후(대상 0장이면 422) 백그라운드 실행 + per-image SSE 진행.
- `val_split`>0이고 계산상 0이지만 이미지≥2면 최소 1장 val 보장. seed 고정 셔플로 재현 가능.
- 파일 IO만 → ThreadPoolExecutor(max_workers=2).

## 데이터·저장 구조

- DB: **없음**(export는 파일 기반, 메타는 `export.json`).
- 디스크: 입력 `projects_dir/{pid}/raw/`, `labels/`, `classes.json`. 출력 `exports/{export_id}/`(images|labels/train|val, data.yaml, export.json) + `exports/{export_id}.zip`. 진행 `jobs_dir/{export_id}/progress.jsonl`, 취소 `CANCEL`.

## API (prefix `/projects/{project_id}/exports`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `.../exports` | export 잡 시작 (201, 검증 실패 422) — `exports.py:44` |
| GET | `.../exports/{export_id}/events` | SSE 빌드 진행(start→copy→zip→done) — `exports.py:72` |
| POST | `.../exports/{export_id}/cancel` | export 취소 — `exports.py:97` |
| GET | `.../exports` | export 목록(export.json 스캔, 최신순) — `exports.py:104` |
| GET | `.../exports/{export_id}/download` | zip 다운로드(FileResponse) — `exports.py:121` |
| PATCH | `.../exports/{export_id}` | 이름 변경 |
| DELETE | `.../exports/{export_id}` | 삭제 (204) |

## 처리 흐름

`create_export` → `target_images(pdir, names, kind)` 동기 검증 → `export_manager.submit`
(`ThreadPoolExecutor`) → `_run` → `export_build.build_export`(이미지/라벨 `shutil.copy2` + `write_data_yaml` + `shutil.make_archive("zip")`). 종료/에러/취소 이벤트는 `_run`이 emit.
- 워커 방식: **ThreadPoolExecutor**(순수 파일 IO, 프로세스 아님).

## 핵심 파일·함수

- `app/api/v1/endpoints/exports.py:44` `create_export`, `:72` `export_events`, `:97` `cancel_export`, `:104` `list_exports`, `:121` `download_export`
- `app/services/export_manager.py:73` `submit`, `:30` `_run`, `:94` `cancel`
- `app/domain/export_build.py:60` `build_export`, `:38` `target_images`

## 엣지 케이스·주의

- 취소: 큐면 `future.cancel()`, 실행 중이면 CANCEL 센티넬 → 복사 루프 `_check_cancel()` → `ExportCancelled`.
- 진행 보고: `{phase:"start"|"copy"(copied/total)|"zip"}`; 종료 이벤트는 서비스 계층 책임.
- download/rename/delete에 path-traversal 가드(`"/" in id or startswith(".")` → 422).
- `data.yaml`의 val은 val 카운트 0이면 `images/train`을 가리킴.
- 진행·취소 계약은 [jobs-and-progress.md](jobs-and-progress.md), 출력물은 [training.md](training.md)의 학습 입력으로 이어짐.
