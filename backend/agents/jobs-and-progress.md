# Jobs & Progress (cross-cutting IPC)

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/jobs.py`, `app/services/label_manager.py`(공용 reader), `app/models/__init__.py`(Job)

## 목적

API 프로세스와 자식 워커 프로세스 간 **파일 기반 IPC**. DB 없이도 워커가 참여할 수 있도록
진행률 스트리밍(SSE)과 취소를 파일로 통일한다. 라벨링 잡만 DB에 영속되고,
test/video/export/train은 동일한 progress 계약을 재사용한다.

## 요구사항

- **진행률**: 워커가 `jobs_dir/{id}/progress.jsonl`에 `{ts, phase, ...}` 한 줄씩 append → API가 byte-offset tail 폴링 → SSE `progress` 이벤트. 부분(미완결) 줄은 다음 폴링까지 보류.
- **취소**: `jobs_dir/{id}/CANCEL` 센티넬 파일 touch → 워커가 루프마다 `cancel.exists()` 확인(협조적 취소). 큐 대기 중이면 `future.cancel()`로 즉시 취소.
- terminal phase(`done|error|cancelled`)에서 SSE 종료. 잡이 이벤트 없이 죽어도 idle 10폴링마다 DB status 확인해 종료 감지.

## 데이터·저장 구조

- DB 테이블 `job` (`models/__init__.py:50`): `id`(j_ prefix), `project_id`(index), `kind`(label|train), `status`(queued|running|done|error|cancelled), `config_json`, `result_json`, `error`, `created_at`, `finished_at`. **라벨링 잡만** DB에 기록.
- 디스크: `data/jobs/{id}/progress.jsonl`, `CANCEL`, 잡별 산출물(`result.json`, `images_manifest.json` 등).

## API (jobs.router — prefix 없음, `/api/v1` 하위 top-level)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/projects/{project_id}/jobs` | 라벨링 잡 생성·시작 (201) — `jobs.py:36` |
| GET | `/projects/{project_id}/jobs` | 프로젝트 잡 목록 — `jobs.py:84` |
| GET | `/jobs/{job_id}` | 잡 단건 — `jobs.py:92` |
| POST | `/jobs/{job_id}/cancel` | 잡 취소 — `jobs.py:100` |
| GET | `/jobs/{job_id}/events` | SSE 진행률 tail — `jobs.py:112` |

## 처리 흐름 (라벨링)

`jobs.py:create_job`(모델경로 해석·cfg 조립·Job insert) → `label_manager.submit_label_job`
(progress.jsonl touch, `ProcessPoolExecutor(max_workers=1, spawn)` submit, status=running, done callback)
→ 자식 `label_worker.run_label_job`(진행 append, CANCEL 확인) → `_on_done`이 결과를 DB에 반영.
SSE는 `jobs.py:job_events` → `asyncio.to_thread(read_progress, ...)` 0.5초 폴링.

## 핵심 파일·함수

- `app/services/label_manager.py:104` `read_progress` — **모든 SSE 엔드포인트 공용 tail reader**(byte-offset, 부분 줄 보류)
- `app/api/v1/endpoints/jobs.py:112` `job_events` — SSE 루프
- `app/models/__init__.py:50` `Job`, `:11` `iso_utc`(SQLite tz 처리)
- **progress writer(`_emit`) 재사용처**: `label_worker.py`, `train_runner.py`, `annotate_worker.py`, `compare_worker.py`, `export_manager.py`/`export_build.py`, `video_manager.py`
- **CANCEL 센티넬 재사용처**: `label_manager.py`, `test_jobs.py`, `video_manager.py`, `export_manager.py`

## 엣지 케이스·주의

- 진행률 offset은 **byte offset** — 각 SSE 소비자가 자기 offset 유지. 파일은 append-only(rotation/truncate 없음).
- 취소는 **협조적**(cooperative) — 워커 루프가 확인해야만 반영, 무한 블록 구간에선 지연.
- SSE 폴링 0.5초 — 파일 기반이라 다중 리더 안전하나 실시간성 ~0.5초 지연.
- 워커 함수는 spawn pickle 제약(모듈-레벨 + plain-dict 인자). 워커 패턴 상세는 [`../CLAUDE.md`](../CLAUDE.md) 참고.
