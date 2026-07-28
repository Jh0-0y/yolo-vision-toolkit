# Backend 아키텍처 & 컨벤션

이 문서는 `backend/`에서 작업할 때 **반드시 따르는 구조·규칙**이다. 다른 프로젝트에도 그대로 복사해 템플릿으로 쓸 수 있게 작성되어 있다.

> **기능별 상세는 [`agents/`](agents/) 참고.** 이 문서는 *계층·컨벤션 규칙*("어디에 무엇을 두는가")만 담는다. 개별 기능의 요구사항·처리 흐름·엣지케이스("무엇을·어떻게")는 [`agents/README.md`](agents/README.md)의 문서 목록에서 찾는다. 작업 순서: 이 문서로 규칙을 잡고 → 해당 기능 문서로 진입.

## 대원칙 — 계층 경계

> **API·services = 비즈니스·상태·결정·DB / workers = 순수 계산(DB·결정 없음) / ml·domain = 순수 함수(프로세스·DB 모름).**

- `torch`/`ultralytics`/CUDA는 **`workers/`와 `ml/`에서만** import한다. 그것도 **함수 안에서 lazy import** — 모듈 최상위에서 절대 import하지 않는다. API 프로세스가 CUDA를 로드하면 안 되기 때문이다.
- 워커는 "다른 런타임(별도 프로세스)"이지 요청 경로의 한 계층이 아니다. 그래서 최상위 패키지로 격리한다.
- DB(SQLite)는 **API 프로세스에서만** 만진다. 워커는 값을 받아 계산하고 값을 돌려줄 뿐, DB를 모른다.

## 디렉터리 구조

```
app/
├── main.py            # ASGI 엔트리 (위치 고정 — app.main:app 이 하드코딩됨)
├── core/
│   └── config.py      # Settings(env: YVT_*), resolve_device, get_device_info, api_prefix
├── db/                # SQLModel 엔진/세션/마이그레이션 (get_session, session_scope, init_db)
├── models/            # SQLModel ORM 테이블 정의
├── schemas/           # Pydantic 요청/응답 DTO (리소스별 파일)
├── domain/            # 순수 파일 IO·도메인 로직 (torch·DB 없음): yolo_io, labels, classes, video, thumbnails …
├── ml/                # 모델 컴퓨트 (ultralytics lazy, DB 없음): ensemble, labeling, predict …
├── services/          # 오케스트레이터 (API 프로세스 상주). 워커 lifecycle + DB 뒤처리 + 비즈니스 결정
│   └── *_manager.py   # 싱글톤 (job_manager, train_manager, …)
├── workers/           # 자식 프로세스 엔트리 (torch/CUDA 여기서만). pickle 가능한 모듈-레벨 함수
└── api/
    └── v1/
        ├── router.py       # /api/v1 집약 라우터 — main.py 는 이것 하나만 include
        └── endpoints/      # 리소스별 라우터 (얇게: 검증 → service 호출 → 응답)
```

## "새 코드 어디에?" 결정표

| 만드는 것 | 위치 | 예 |
|---|---|---|
| HTTP 라우트 | `api/v1/endpoints/<resource>.py` | 새 엔드포인트 |
| 요청/응답 DTO | `schemas/<resource>.py` | `RunCreate`, `ModelOut` |
| 워커 오케스트레이션·잡 관리 | `services/<x>_manager.py` | `InferManager` |
| 자식 프로세스에서 도는 코드 | `workers/<x>_worker.py` / `<x>_runner.py` | 추론·학습 엔트리 |
| 모델 연산(추론/앙상블) | `ml/*.py` | `predict.py` |
| 파일/라벨/클래스 도메인 로직 | `domain/*.py` | 라벨 파일 read/write |
| 설정·device 정책 | `core/config.py` | 새 env 설정 |
| DB 테이블 | `models/__init__.py` | 새 SQLModel |

## 워커 패턴 (services ↔ workers)

- **매니저(services)** 가 워커 프로세스를 소유·기동하고 결과를 DB에 반영한다. 두 가지 방식이 공존:
  - `ProcessPoolExecutor(max_workers=1, spawn)` — 풀 워커가 작업 사이에 **살아있음** (warm 캐시에 유리). 예: `services/label_manager.py`.
  - `subprocess.Popen(["-m", "app.workers.<mod>", …])` — 일회성 프로세스. 예: `services/train_manager.py` → `app.workers.train_runner`.
- **워커(workers)** 함수는 반드시 **모듈-레벨 + picklable**(plain-dict 인자). spawn 인터프리터에서 재-import 되므로 최상위 부작용 금지.
- 취소는 **CANCEL 센티넬 파일**, 진행상황은 **`progress.jsonl` tail**로 주고받는다(파일 기반 IPC).

## API 컨벤션

- **버전**: 모든 경로는 `/api/v1/...`. 프리픽스의 단일 소스는 `settings.api_prefix` (라우터 + 백엔드가 내보내는 URL 문자열 모두 이걸 참조). 무버전 `GET /api/health`(인프라 프로브)만 예외.
- **라우터**: 각 엔드포인트 라우터는 **리소스 sub-prefix만** 보유(`prefix="/models"`). `api/v1/router.py`가 `settings.api_prefix`를 붙여 집약한다. `main.py`는 이 집약 라우터 하나만 include.
- **URL 명명**: 컬렉션은 복수명사(`/projects`, `/models`, `/runs`). 다단어 세그먼트는 kebab-case(`per-class`, `dataset-zip`). 프로젝트 소유 리소스는 중첩(`/projects/{project_id}/labels`). 범-프로젝트 카탈로그는 top-level + `?project_id=` 필터.
- **HTTP 메서드/상태코드**:
  - GET → 200 · **생성 POST → 201** · PUT/PATCH → 200 · **단일 삭제 DELETE → 204(무바디)**.
  - 계산 결과를 돌려주는 벌크/삭제는 200 + 요약 바디(예: 이미지 벌크삭제 `DELETE /images` → `{deleted}`, 클래스 삭제 → `{removed_boxes,…}`).
  - 비-CRUD 상태전이는 `POST /{id}/{verb}` 패턴: `cancel`, `stop`, `register`, `resample`.
- **엔드포인트는 얇게**: 검증 → service/domain 호출 → DTO 응답. 무거운 로직은 계층으로 내린다.

## 깨지기 쉬운 커플링 (파일 이동·리네임 시 주의)

1. **ASGI 경로 `app.main:app`** — `dev.sh`, `docker/backend.Dockerfile`, `README.md`, `.claude/launch.json`에 하드코딩. `app/main.py`는 위치를 옮기지 말고, 재구성은 `app/` **내부**에서만.
2. **워커 서브프로세스 모듈 문자열** — `services/train_manager.py`의 `"-m", "app.workers.train_runner"`. import 문이 아니라 문자열이라 grep에 안 잡힌다. 워커 이동 시 이 문자열도 같이 수정.
3. **spawn pickle 제약** — `ProcessPoolExecutor`에 넘기는 워커 함수와 그 의존은 fresh 인터프리터에서 import 가능해야 한다.
4. **`core/config.py`의 repo-root 앵커** — `_REPO_ROOT = Path(__file__).resolve().parents[N]`. `config.py`의 디렉터리 깊이가 바뀌면 `N`을 맞춰야 data_dir이 깨지지 않는다.

## 검증 루틴

- 백엔드: `uv run python -m py_compile $(git ls-files 'app/**/*.py')` → `uv run pytest` → `uv run uvicorn app.main:app --app-dir backend --port 8010` 부팅 → `/api/health`·`/api/v1/system/device` 200, `/openapi.json` 전 경로 `/api/v1`.
- 프론트: `npx tsc --noEmit && npx oxlint src`.
- 학습/추론 등 서브프로세스 경로는 **실제 1회 실행**으로만 검증됨(모듈 문자열은 런타임에만 발현).
