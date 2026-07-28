# Storage Layout

> 상태: 구현됨 · 핵심 파일: `app/core/config.py`

## 목적

전 백엔드의 저장 경로·디바이스 정책·API 버전 프리픽스의 **단일 소스**.
`Settings`(env prefix `YVT_`)가 모든 경로를 CWD 비의존으로 계산한다(repo-root 앵커).

## 환경 변수 (`YVT_*`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `YVT_DATA_DIR` | `_REPO_ROOT/data` | 모든 데이터 루트 (`config.py:13`) |
| `YVT_SSD_CACHE_DIR` | `None` | 학습 시 데이터셋을 빠른 SSD로 스테이징. None이면 비활성 (`config.py:19`) |
| `YVT_DEVICE` | `"auto"` | 디바이스 정책 (auto → cuda>mps>cpu) (`config.py:21`) |
| `YVT_CORS_ORIGINS` | `localhost:5173,3000` | CORS 허용 오리진 (`config.py:22`) |
| `YVT_API_PREFIX` | `"/api/v1"` | API 버전 프리픽스 (`config.py:24`) |

## 경로 프로퍼티 (모두 `data_dir` 하위, `config.py:26-68`)

- `db_path` → `data/db.sqlite3` (SQLite, WAL 모드)
- `models_dir` → `data/models` — 공유/legacy 모델 풀 (`project_id=None`)
- `projects_dir` → `data/projects`
- `jobs_dir` → `data/jobs` — 진행률/취소/결과 IPC
- `runs_dir` → `data/runs` — 공유/legacy 학습 런
- `datasets_dir` → `data/datasets` — 업로드 학습 데이터셋
- `test_dir` → `data/test` — Test 놀이터 (transient)
- `model_dir(project_id, model_id)` — `project_id` 있으면 `projects/{pid}/models/{mid}`, 없으면 `models/{mid}` (`config.py:56`)
- `run_dir(project_id, run_id)` — 있으면 `projects/{pid}/runs/{rid}`, 없으면 `runs/{rid}` (`config.py:63`)
- `ensure_dirs()` — 부팅 시 기본 버킷 생성 (`config.py:70`)

## 디바이스 함수

- `resolve_device(device)` — `"auto"` → cuda(`"0"`)>mps>cpu (`config.py:104`)
- `get_device_info(device)` — `/system/device`용 리졸브 디바이스+가속기 상세 (`config.py:138`)
- `get_resource_info(device)` — psutil 기반 RAM/VRAM 라이브 압력 (Test 리소스 가드) (`config.py:83`)

## 디렉터리 트리

```
data/
├── db.sqlite3 (+ -wal, -shm)      # SQLite (WAL)
├── models/{model_id}/             # 공유/legacy 모델: model.pt + meta.json
├── runs/{run_id}/                 # 공유/legacy 학습 런
├── datasets/{dataset_id}/         # 업로드 학습 데이터셋: dataset.json, data.yaml
├── jobs/{id}/                     # IPC: progress.jsonl, CANCEL, result.json, images_manifest.json
│                                  #   id prefix로 종류 구분: 라벨 j_*, 학습 t_*, 영상 vid_*, export e_*, test=uuid hex
├── projects/{project_id}/         # 프로젝트 소유 리소스
│   ├── project.json               # 프로젝트 메타
│   ├── classes.json               # 클래스 레지스트리
│   ├── reviewed.json              # 리뷰 완료 stem 집합
│   ├── raw/                       # 원본 이미지 (예측 image_name 참조 대상)
│   ├── labels/                    # YOLO 라벨 .txt (+ .meta.json 사이드카)
│   ├── thumbs/                    # 썸네일
│   ├── videos/                    # 프로젝트 영상 (재샘플용 원본 + .json 메타)
│   ├── exports/{export_id}/       # export 산출물 + {export_id}.zip
│   ├── models/{model_id}/         # 프로젝트 스코프 모델: model.pt + meta.json
│   └── runs/{run_id}/             # 프로젝트 스코프 학습 런
└── test/                          # Test 놀이터 (transient, TTL sweep)
    ├── uploads/{uuid}{ext}        # 예측 업로드 (즉시 unlink)
    ├── annotate/{job_id}/         # source{ext}, out.mp4, crop.json (TTL 3600s)
    └── compare/{job_id}/          # upload.zip, dataset/ (TTL 3600s)
```

## 엣지 케이스·주의

- **모델·런은 `project_id` 유무로 경로가 분기**된다 — `model_dir`/`run_dir` 헬퍼가 유일한 진실. `project_id=None`(공유) 모델은 모든 프로젝트에서 가시.
- `jobs_dir`는 라벨/학습/영상/export/test가 **공유**한다(id prefix로 구분). video/export는 자체 task_dir이지만 공용 `read_progress`가 동작하도록 `jobs_dir`를 재사용 — [jobs-and-progress.md](jobs-and-progress.md) 참고.
- `data/test/`는 DB에 기록되지 않고 TTL sweep 대상 — 영구 저장 아님.
- repo-root 앵커 `_REPO_ROOT = parents[3]` (`config.py:7`): `config.py`의 디렉터리 깊이가 바뀌면 N을 맞춰야 한다 → [`../CLAUDE.md`](../CLAUDE.md)의 "깨지기 쉬운 커플링" 참고.
- `settings.api_prefix`는 라우터 마운트와 워커가 result에 심는 URL 문자열(`compare_worker.py`) 양쪽에서 참조 — 변경 시 저장된 result.json URL과 불일치 가능.
