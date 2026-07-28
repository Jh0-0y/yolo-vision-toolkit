# Training

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/training.py`, `app/services/train_manager.py`, `app/workers/train_runner.py`, `app/domain/staging.py`, `app/domain/datasets.py`

## 목적

export/업로드 데이터셋으로 ultralytics 학습 subprocess를 기동·모니터·중단하고, 산출물을 조회하며,
학습된 weights를 모델 레지스트리로 재등록해 루프를 닫는다.

## 요구사항

- **동시 1런만**(`has_active()` → 409).
- 데이터셋 소스 2종: 프로젝트 export(`export:{pid}:{eid}`), 업로드 zip(`upload:{uid}`). 업로드는 `data.yaml` 정규화(경로 재해석, `path` 키 제거).
- per-epoch 메트릭 스트림(progress.jsonl), per-class 메트릭(최종/에폭별) 별도 파일.
- SIGTERM으로 중단 가능(별도 프로세스).
- 선택적 SSD 스크래치 staging(대용량 데이터셋을 빠른 디스크로 복사 후 학습, 종료 시 정리).
- `auto_delete` 업로드는 성공 시 원본 삭제.
- 학습 subprocess가 torch/CUDA를 API 프로세스에서 격리.

## 데이터·저장 구조

- DB: `train_run`(`TrainRun`: name, project_id, dataset_path, base_model_id, params_json, status queued/running/done/error/stopped, pid, metrics_json).
- 디스크: 업로드 데이터셋 `datasets_dir/{uid}/`(dataset.json, data.yaml). 런 산출물 `run_dir(project_id, run_id)`(config.json, train.log, results.csv, args.yaml, per_class.json, per_class_history.jsonl, weights/best.pt|last.pt, 플롯 png). 진행 `jobs_dir/{run_id}/progress.jsonl`. 선택 staging `ssd_cache_dir/{run_id}`.

## API (prefix `/training`)

데이터셋:
- `GET /training/datasets` — 학습 가능 데이터셋(export+upload, 최신순)
- `POST /training/datasets` — 데이터셋 zip 업로드·추출 (201)
- `PATCH /training/datasets/{dataset_id}` — auto_delete 토글(업로드만)
- `DELETE /training/datasets/{dataset_id}` — 삭제 (204, 사용중이면 409)

런:
- `POST /training/runs` — 런 생성·기동 (201, 동시 실행 시 409) — `training.py:317`
- `GET /training/runs`, `GET /training/runs/{run_id}` — 목록/단건
- `GET /training/runs/{run_id}/history` — epoch 이벤트만
- `GET /training/runs/{run_id}/per-class` · `/per-class-history` — per-class 메트릭
- `GET /training/runs/{run_id}/results` · `/results.csv` · `/args.yaml` · `/log` — 산출물 조회/다운로드
- `POST /training/runs/{run_id}/stop` — 중단, `DELETE /training/runs/{run_id}` — 삭제 (204, 활성이면 409)
- `GET /training/runs/{run_id}/events` — SSE 진행 스트림
- `GET /training/runs/{run_id}/artifacts` · `/artifacts/{name}` · `/weights/{which}` — 산출물·weights
- `POST /training/runs/{run_id}/register` — weights를 모델 레지스트리에 등록 (201) → [models.md](models.md)

## 처리 흐름

`create_run`(`_resolve_dataset_dir`로 data.yaml·base pt 확인, `has_active()` 체크 → TrainRun insert + config.json)
→ `train_manager.start`. ssd_cache 없으면 즉시 `_spawn`, 있으면 백그라운드 thread `_prepare_and_run`(staging 후 spawn).
`_spawn`은 **subprocess.Popen** `[sys.executable, "-m", "app.workers.train_runner", str(run_dir)]`(cwd=backend, env `YVT_DATA_DIR`/`YVT_DATASET_PATH_OVERRIDE`, stdout→train.log).
워커 `train_runner.main`이 ultralytics `model.train(...)` 실행, 콜백으로 progress.jsonl emit.
`_watch`가 `proc.wait()` 후 DB status 확정 + staging/auto_delete 정리.

## 핵심 파일·함수

- `app/api/v1/endpoints/training.py:317` `create_run`, `:294` `_resolve_dataset_dir`, `:182` `_extract_dataset`, `:134` `_normalize_data_yaml`, `:524` `stop_run`, `:630` `register_weights`
- `app/services/train_manager.py:39` `start`, `:94` `_spawn`, `:117` `stop`, `:135` `_watch`, `:202` `reconcile_on_boot`
- `app/workers/train_runner.py:44` `main`, `:78` `on_fit_epoch_end`
- `app/domain/staging.py`(SSD 스테이징), `app/domain/datasets.py`(업로드 라이프사이클)

## 엣지 케이스·주의

- **깨지기 쉬운 커플링**: `_spawn`의 `"-m", "app.workers.train_runner"`는 grep에 안 잡히는 문자열 — 워커 이동/리네임 시 함께 수정. `_extract_dataset`은 yaml 파일명이 반드시 `data.yaml`이어야 함. → [`../CLAUDE.md`](../CLAUDE.md)의 "깨지기 쉬운 커플링".
- 중단: `stop`이 `proc.terminate()`; 이전 API 수명의 프로세스는 저장된 `pid`로 `os.kill(SIGTERM)`. `_watch`가 exit code로 stopped/error 판정.
- staging 실패는 런을 죽이지 않고 원본 경로로 폴백. `_pending`을 반드시 clear해 `has_active()` 영구 True 방지.
- SIGTERM은 워커 finally를 건너뛰므로 staged 복사본 정리는 부모(`_watch`)가 수행.
- `reconcile_on_boot`: API 재시작 시 pid 생존 확인, 죽은 running 런 → error + 고아 staged dir 정리.
- per-class 캡처 실패는 런을 실패시키지 않음(try/except). results.csv/args.yaml은 ultralytics 버전에 따라 한 단계 하위 디렉터리 glob 폴백.
