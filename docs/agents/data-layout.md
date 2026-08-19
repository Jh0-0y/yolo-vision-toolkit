---
title: 데이터 배치
scope: backend/app/**/*.py
applies_to: 파일을 읽고 쓰거나, DB에 넣을지 파일로 둘지 정할 때
related:
  - ./architecture.md
  - ./structure-dir.md
  - ./conventions/jobs-and-progress.md
  - ./conventions/datasets.md
---

# 데이터 배치

> 영구 데이터는 전부 `DATA_DIR` 아래에 있고, DB에 있는 것과 파일에 있는 것이 나뉜다. 파일을 읽고 쓸 때 읽는다.

## 트리

공간이 둘이다. **학습실**(`projects/`)은 데이터셋·라벨·학습을 갖고, **연구실**(`labs/`)은
영상과 크롭 런만 갖는다. 둘은 **모델 풀만 공유**한다 — 영상은 각자 갖는다(학습용 장면
샘플과 크롭할 경기는 실제로 다른 영상이다).

```
DATA_DIR/                     # 기본 <repo>/data — git 이 추적하지 않는다
├── db.sqlite3                # 메타데이터 (WAL 모드)
├── models/{model_id}/        # 공유 모델 풀 — model.pt + meta.json
├── projects/{project_id}/    # 학습실 — 프로젝트는 껍데기다
│   ├── project.json
│   ├── models/ · runs/       # 프로젝트 스코프 모델·학습 결과
│   └── datasets/{dataset_id}/    # 데이터셋 하나가 자기 것을 전부 갖는다
│       ├── dataset.json      # 이름 · 생성일
│       ├── raw/              # 이미지
│       ├── thumbs/           # 썸네일 — 요청할 때 만든다
│       ├── labels/{stem}.txt # YOLO 라벨 (+ {stem}.meta.json — 박스별 score·status)
│       ├── classes.json      # **이 데이터셋만의** 클래스
│       ├── reviewed.json     # {stem: true}
│       └── splits.json       # {stem: "train"|"val"|"test"}
├── labs/{lab_id}/            # 연구실
│   ├── lab.json              # 사람이 읽는 표시(이름) — 진실은 DB 행이다
│   ├── videos/{video_id}.mp4 # 원본 + {video_id}.json 사이드카 (프레임 추출 없음)
│   └── crops/{crop_id}/      # 크롭 런 — run.json · crop.json · wide.mp4 · crop.mp4
├── jobs/{job_id}/            # progress.jsonl · CANCEL
└── test/                     # 모델비교 캐시 — 순수 임시물, 자동 정리된다
```

**데이터셋끼리는 아무것도 공유하지 않는다.** 같은 영상을 두 데이터셋에 쓰려면 두 번
추출한다. 그래서 삭제도 통째로고, 한쪽을 고쳐도 다른 쪽이 흔들리지 않는다.

## 경로는 반드시 `settings` 에서 파생시킨다

```python
from app.core.config import settings

pdir = settings.projects_dir / project_id        # ○
pt = settings.model_dir(project_id, model_id) / "model.pt"   # ○
```

- `settings.data_dir` · `models_dir` · `projects_dir` · `labs_dir` · `jobs_dir` · `runs_dir` · `datasets_dir` · `test_dir` 프로퍼티를 쓴다 (연구실 하나는 `settings.lab_dir(lab_id)`).
- 문자열로 경로를 조립하거나 CWD 기준 상대경로를 쓰지 **않는다.** `data_dir` 기본값은 저장소 루트에 앵커돼 있고, 배포에서는 다른 볼륨을 가리킨다.
- 모델·학습 결과는 **프로젝트 스코프와 공유 풀 두 자리**가 있다. 직접 조립하지 말고 `settings.model_dir(project_id, model_id)` · `settings.run_dir(...)` 를 쓴다 (`project_id=None` 이면 공유 풀).

## DB 인가 파일인가

| DB (SQLite) | 파일 |
|---|---|
| `Project` · `LabProject` · `ModelEntry` · `Job` · `TrainRun` — 이 다섯뿐 | 라벨 · 클래스 · 검수 플래그 · 내보내기 · 연구실 영상 · 크롭 런 · 잡 진행상황 |

- 새 상태를 만들 때 **먼저 파일로 둘 수 있는지 본다.** 지금까지 DB는 "목록에 뜨고 상태가 바뀌는 것"만 담았다.
- DB 접근은 **API 프로세스에서만** 한다. 워커는 값을 받아 값을 돌려준다 → [계층 경계](conventions/layer-boundaries.md)
- 테이블에 컬럼을 추가하면 `db/session.py` 의 `_add_missing_columns` 에도 넣는다. `create_all` 은 기존 테이블을 바꾸지 못한다. 추가 컬럼은 **nullable** 이어야 기존 행이 살아남는다.

## 손대지 않는 것

- **`data/` 아래 실제 데이터를 검증용으로 지우거나 덮어쓰지 않는다.** git 이 추적하지 않아 되돌릴 수 없다.
- 임시 산출물이 필요하면 `settings.test_dir` 아래에 만든다(자동 정리 대상).

## 연구실 영상

`labs/{lab_id}/videos/` 에 원본과 사이드카가 나란히 있다. **프레임 추출은 하지 않는다** — 학습
재료가 아니라 크롭해서 내보낼 대상이다. 확장자는 사이드카(`{video_id}.json`)가 들고 있으므로
**추측하지 않는다**(컨테이너가 여러 가지다).

## 크롭 런

`labs/{lab_id}/crops/{crop_id}/` 하나가 크롭 런 하나다. `crop_id` 는 **잡 id 로도 그대로** 쓰므로
진행률은 `jobs/{crop_id}/progress.jsonl` 에 있다.

| 파일 | 담는 것 |
|---|---|
| `run.json` | 설정 스냅샷. 잡을 **던지기 전에** 쓴다 — 실패한 시도도 목록에 남는다 |
| `crop.json` | 좌표. 커버리지 요약도 이 안에 있어 따로 저장하지 않는다 |
| `wide.mp4` | 가로 영상 — 그리기 옵션이 있으면 오버레이 렌더본, 없으면 원본 하드링크 |
| `crop.mp4` | 세로 크롭 영상. 오버레이를 굽지 않는다 — 산출물이지 관찰용이 아니다 |

**셋은 전부 필수다.** 선택이 아니다. **TTL 은 없다** — 지우는 것은 사용자뿐이다.

`wide.mp4` 는 원본과 완전히 분리된다. 그릴 것이 없을 때 통째로 복사하면 런마다 원본만 한 용량이
들어서, **하드링크로 만들고 실패하면(다른 볼륨) 복사로 폴백**한다. 링크든 복사든 아카이브에서
원본을 지워도 런은 온전하다.

**상태는 저장하지 않는다.** `running` / `done` / `error` 는 `progress.jsonl` 의 마지막 이벤트에서 파생한다 → [잡과 진행률](conventions/jobs-and-progress.md)

자리·상태 파생·하드링크는 전부 `services/lab_crop_runs.py` 하나가 안다. 경로를 직접 조립하지 않는다.
