# Auto-Labeling

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/jobs.py`, `app/services/label_manager.py`, `app/workers/label_worker.py`, `app/ml/labeling.py`, `app/ml/ensemble.py`

## 목적

N개 모델을 이미지 폴더에 앙상블 추론(weighted-box fusion)하여 `labels/{stem}.txt` YOLO 라벨을
자동 생성한다. 사람이 이후 검수(review)하도록 **review 상태는 절대 설정하지 않는다**.

## 요구사항

- 여러 모델 detection을 WBF로 융합(`merge_detections`). `cfg.conf`는 융합 박스에 대한 **하드 필터**(리뷰 경계가 아님).
- 모델은 `DETECT_FLOOR=0.05`로 추론(저신뢰 후보를 WBF에 노출)한 뒤 `cfg.conf` 필터.
- 클래스별 최대 박스 수 캡(`max_boxes_per_class`, 미지정 시 `DEFAULT_MAX_PER_CLASS=300`).
- 재라벨링해도 클래스 id가 재번호되지 않도록 기존 `classes.json`에서 registry seeding.
- 빈 라벨 파일 = negative(정상). 자동 라벨된 이미지는 reviewed 플래그 해제.
- **GPU 1개 → 동시 1잡**(`max_workers=1`), 나머지는 큐 대기.

## 데이터·저장 구조

- DB: `job`(`kind="label"`, status queued/running/done/error/cancelled, `config_json`, `result_json`) — [jobs-and-progress.md](jobs-and-progress.md).
- 디스크: 입력 `projects_dir/{pid}/raw/`, 출력 `labels/{stem}.txt`(+ box-meta), `classes.json`, `reviewed`. 진행 `jobs_dir/{job_id}/progress.jsonl`, 취소 `CANCEL`.

## API

라벨링 잡 트리거·조회는 jobs 라우터를 사용 → [jobs-and-progress.md](jobs-and-progress.md).
- `POST /projects/{project_id}/jobs` — 라벨링 잡 생성·기동 (201)
- `GET /jobs/{job_id}/events` — SSE 진행 스트림, `POST /jobs/{job_id}/cancel` — 취소

## 처리 흐름

`jobs.py:create_job`(모델 검증·pt 경로 해석·Job insert) → `label_manager.submit_label_job`
(`ProcessPoolExecutor(max_workers=1, spawn)`) → 자식 `label_worker.run_label_job`
→ `ml/labeling.py:run_labeling`(배치 루프: 모델별 predict → `ensemble.merge_detections` → `_cap_per_class`
→ `write_label_file`/`write_box_meta`) → 완료 콜백 `_on_done`이 DB 갱신 + `reset_reviewed`.
- 워커 방식: **ProcessPoolExecutor(spawn)**, warm 유지 (subprocess.Popen 아님).

## 핵심 파일·함수

- `app/api/v1/endpoints/jobs.py:37` `create_job`
- `app/services/label_manager.py:34` `submit_label_job`, `:68` `_on_done`, `:90` `reset_reviewed`, `:104` `read_progress`
- `app/workers/label_worker.py:19` `run_label_job`
- `app/ml/labeling.py:102` `run_labeling`, `:88` `_cap_per_class`, `:77` `_load_registry`
- `app/ml/ensemble.py:60` `merge_detections`(per-class WBF)

## 엣지 케이스·주의

- 취소: 큐면 `future.cancel()`; 실행 중이면 CANCEL 센티넬 → 배치 루프 시작에서 `cancel_check()` → `JobCancelled`.
- 진행 보고: 배치 단위 `progress({phase:"inference"|"cancelled"|"done"|"error"})`.
- spawn pickle 제약: `run_label_job`은 모듈-레벨 + plain-dict 인자. 워커 규칙은 [`../CLAUDE.md`](../CLAUDE.md) 참고.
- `annotate_worker.py`는 자동 라벨링이 **아님** — Test 놀이터의 비디오 어노테이션 → [inference-and-test.md](inference-and-test.md).
- WBF·앙상블 세부는 [inference-and-test.md](inference-and-test.md)와 `ensemble.py` 공유.
