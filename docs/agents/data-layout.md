---
title: 데이터 배치
scope: backend/app/**/*.py
applies_to: 파일을 읽고 쓰거나, DB에 넣을지 파일로 둘지 정할 때
related:
  - ./architecture.md
  - ./structure-dir.md
  - ./conventions/jobs-and-progress.md
---

# 데이터 배치

> 영구 데이터는 전부 `DATA_DIR` 아래에 있고, DB에 있는 것과 파일에 있는 것이 나뉜다. 파일을 읽고 쓸 때 읽는다.

## 트리

```
DATA_DIR/                     # 기본 <repo>/data — git 이 추적하지 않는다
├── db.sqlite3                # 메타데이터 (WAL 모드)
├── models/{model_id}/        # 공유 모델 풀 — model.pt + meta.json
├── projects/{project_id}/
│   ├── raw/                  # 원본 이미지
│   ├── thumbs/               # 썸네일
│   ├── videos/               # 업로드 영상
│   ├── labels/               # 이미지별 YOLO 라벨 (.txt)
│   ├── exports/              # 내보낸 데이터셋 zip
│   ├── crops/{crop_id}/      # 크롭 랩 산출물 — run.json · crop.json · out.mp4(만료됨)
│   ├── models/ · runs/       # 프로젝트 스코프 모델·학습 결과
│   ├── classes.json          # 클래스 레지스트리
│   ├── reviewed.json         # 검수 완료 플래그
│   └── project.json
├── jobs/{job_id}/            # progress.jsonl · CANCEL
├── datasets/                 # 업로드된 학습 데이터셋 전개
└── test/                     # 라이브 프리뷰·모델비교 캐시 — 순수 임시물, 자동 정리된다
```

## 경로는 반드시 `settings` 에서 파생시킨다

```python
from app.core.config import settings

pdir = settings.projects_dir / project_id        # ○
pt = settings.model_dir(project_id, model_id) / "model.pt"   # ○
```

- `settings.data_dir` · `models_dir` · `projects_dir` · `jobs_dir` · `runs_dir` · `datasets_dir` · `test_dir` 프로퍼티를 쓴다.
- 문자열로 경로를 조립하거나 CWD 기준 상대경로를 쓰지 **않는다.** `data_dir` 기본값은 저장소 루트에 앵커돼 있고, 배포에서는 다른 볼륨을 가리킨다.
- 모델·학습 결과는 **프로젝트 스코프와 공유 풀 두 자리**가 있다. 직접 조립하지 말고 `settings.model_dir(project_id, model_id)` · `settings.run_dir(...)` 를 쓴다 (`project_id=None` 이면 공유 풀).

## DB 인가 파일인가

| DB (SQLite) | 파일 |
|---|---|
| `Project` · `ModelEntry` · `Job` · `TrainRun` — 이 넷뿐 | 라벨 · 클래스 · 검수 플래그 · 내보내기 · 크롭 런 · 잡 진행상황 |

- 새 상태를 만들 때 **먼저 파일로 둘 수 있는지 본다.** 지금까지 DB는 "목록에 뜨고 상태가 바뀌는 것"만 담았다.
- DB 접근은 **API 프로세스에서만** 한다. 워커는 값을 받아 값을 돌려준다 → [계층 경계](conventions/layer-boundaries.md)
- 테이블에 컬럼을 추가하면 `db/session.py` 의 `_add_missing_columns` 에도 넣는다. `create_all` 은 기존 테이블을 바꾸지 못한다. 추가 컬럼은 **nullable** 이어야 기존 행이 살아남는다.

## 손대지 않는 것

- **`data/` 아래 실제 데이터를 검증용으로 지우거나 덮어쓰지 않는다.** git 이 추적하지 않아 되돌릴 수 없다.
- 임시 산출물이 필요하면 `settings.test_dir` 아래에 만든다(자동 정리 대상).

## 크롭 런

`projects/{project_id}/crops/{crop_id}/` 하나가 랩에서 돌린 크롭 잡 하나다. `crop_id` 는 **잡 id 로도 그대로** 쓰므로 진행률은 `jobs/{crop_id}/progress.jsonl` 에 있다.

| 파일 | 수명 |
|---|---|
| `run.json` | 영구. 잡을 **던지기 전에** 쓴다 — 실패한 시도도 목록에 남는다 |
| `crop.json` | 영구. 커버리지 요약도 이 안에 있어 따로 저장하지 않는다 |
| `out.mp4` | `VIDEO_TTL_SEC` 뒤 삭제. `run.json` 에 `video_expired` 만 남는다 |
| `source.*` | 잡이 끝나면 삭제 |

**상태는 저장하지 않는다.** `running` / `done` / `error` 는 `progress.jsonl` 의 마지막 이벤트에서 파생한다 → [잡과 진행률](conventions/jobs-and-progress.md)

자리·정리·상태 파생은 전부 `services/crop_runs.py` 하나가 안다. 경로를 직접 조립하지 않는다.
