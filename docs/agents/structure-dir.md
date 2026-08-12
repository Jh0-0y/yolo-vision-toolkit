---
title: 디렉터리 구조
scope: "**"
applies_to: 파일·모듈을 새로 만들거나 기존 코드를 찾을 때
related:
  - ./architecture.md
  - ./conventions/naming.md
  - ./data-layout.md
---

# 디렉터리 구조

> 저장소 트리와 새 파일을 둘 자리. 파일을 만들거나 찾을 때 읽는다.

```
backend/
├── pyproject.toml           의존성·pytest 설정 (uv)
├── cli.py                   웹 UI 없이 폴더 단위 오토라벨링
├── scripts/                 일회성 마이그레이션 스크립트
├── lib/                     순수 기능 — app·infra 를 import 하지 않는다
│   ├── formats.py           IMAGE_EXTS · VIDEO_EXTS — 어디서든 읽는 공용 어휘
│   ├── video/               probe(규격 읽기) · to_h264 · require_ffmpeg
│   ├── media/               extract(프레임 추출) · tiling · thumbnails
│   ├── crop/                geometry(좌표 조회) · window · hud · highlight · cut
│   └── labels/              io(라벨 파일) · store(프로젝트 라벨) · classes · registry
├── infra/                   시스템 배관 — 기능이 아니다
│   └── jobs/                progress.jsonl · CANCEL · JobDir
├── tests/                   lib·infra 테스트 (app 테스트는 app/tests/)
└── app/
    ├── main.py              create_app() · CORS · lifespan
    ├── core/config.py       settings · resolve_device · get_device_info
    ├── db/                  엔진·세션·경량 마이그레이션
    ├── models/__init__.py   DB 테이블 4개 (Project · ModelEntry · Job · TrainRun)
    ├── schemas/             요청·응답 DTO (리소스별 파일)
    ├── api/v1/
    │   ├── router.py        라우터 배선 단 한 곳
    │   └── endpoints/       리소스별 HTTP 경로
    ├── services/            *_manager.py — 잡 수명·프로세스 풀·DB 갱신
    ├── workers/             *_worker.py · train_runner.py — 별도 프로세스 엔트리
    ├── ml/                  모델을 쓰는 순수 계산
    ├── domain/              모델을 쓰지 않는 순수 계산
    └── tests/               test_*.py

frontend/src/
├── main.tsx                 QueryClient · MantineProvider · BrowserRouter
├── App.tsx                  라우트 정의
├── api/client.ts            서버 통신 단일 진입 (경로 함수 + 응답 타입)
├── pages/                   라우트 1개 = 파일 1개
├── layouts/                 프로젝트 셸
├── components/<영역>/       화면 전용 컴포넌트·훅 (dataset · editor · lab · test · upload)
├── components/*.tsx         여러 화면이 쓰는 공용 컴포넌트
├── stores/                  zustand — 화면 간 공유 상태
└── global.css

docker/                      배포 이미지 3종 (backend · frontend · nginx)
scripts/build.sh             GHCR 이미지 빌드·푸시
dev.sh                       로컬 백엔드+프론트 동시 실행
data/                        런타임 데이터 (git 추적 안 함) → data-layout.md
```

## 새 파일을 둘 자리

| 만들 것 | 자리 | 같이 해야 하는 일 |
|---|---|---|
| 여러 기능이 쓰는 순수 계산 | `lib/<주제>/` | `tests/test_lib_<주제>.py` — `app/`·`infra/` import 금지 |
| 잡·프로세스 배관 | `infra/<주제>/` | `tests/test_infra_<주제>.py` — `app/` import 금지 |
| HTTP 엔드포인트 | `app/api/v1/endpoints/<리소스>.py` | `router.py` 의 import 목록과 include 목록 **양쪽**에 등록 |
| 요청·응답 DTO | `app/schemas/<리소스>.py` | 엔드포인트의 `response_model` 로 연결 |
| DB 테이블 | `app/models/__init__.py` (새 파일 만들지 않는다) | — |
| 별도 프로세스 엔트리 | `app/workers/<도메인>_worker.py` | 모듈 최상단 함수 + picklable 인자 |
| 테스트 | `app/tests/test_<대상>.py` | — |
| 화면(라우트) | `frontend/src/pages/<이름>Page.tsx` | `App.tsx` 에 라우트 추가 |
| 화면 전용 컴포넌트·훅 | `frontend/src/components/<영역>/` | — |
| 서버 호출 | `frontend/src/api/client.ts` 에 함수 + 응답 인터페이스 | 컴포넌트에서 `fetch` 직접 호출 금지 |

## 자리가 애매하면 묻는다

`services/` vs `workers/`, `domain/` vs `ml/` 의 경계는 **아직 확정되지 않았다**([아키텍처](architecture.md) 참고).
새 모듈이 이 둘 중 하나로 애매하면 **추측해서 두지 말고 사용자에게 묻는다.**

## 이행 중이다

`lib/` 과 `infra/` 는 `app/domain/`·`app/ml/` 의 잡탕 상태를 걷어내려고 새로 만든 자리다.
**아직 옮기지 않은 것이 많다** — `app/domain/`·`app/ml/` 는 그대로 살아 있고, 지금까지 옮긴 건
영상 프로브·H.264 인코딩(`lib/video/`), 진행률·취소(`infra/jobs/`), 크롭 렌더(`lib/crop/`),
라벨·클래스(`lib/labels/`)뿐이다.

무관한 변경에서 나머지를 함께 옮기지 않는다. 새 코드는 새 자리에 두고, 기존 코드는 그 기능을
손볼 때 같이 옮긴다.
