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
│   ├── device.py            "auto" -> 실제 장치 (설정은 모른다)
│   ├── detect/              predict · labeling · ensemble · evaluate
│   ├── video/               probe(규격 읽기) · to_h264 · require_ffmpeg
│   ├── media/               extract(프레임 추출) · tiling · thumbnails
│   ├── crop/                plan(어댑터) · geometry · window · hud · highlight · cut · palette
│   ├── labels/              io · store · classes · split · import_yolo · dataset_export · registry
│   ├── fsutil.py            link_or_copy — 하드링크, 안 되면 복사
│   └── train/               dataset(data.yaml 찾기) · results(산출물 읽기) · staging
├── infra/                   시스템 배관 — 기능이 아니다
│   └── jobs/                progress.jsonl · CANCEL · JobDir
├── tests/                   lib·infra 테스트 (app 테스트는 app/tests/)
└── app/
    ├── main.py              create_app() · CORS · lifespan
    ├── core/config.py       settings · resolve_device · get_device_info
    ├── db/                  엔진·세션·경량 마이그레이션
    ├── models/__init__.py   DB 테이블 5개 (Project · LabProject · ModelEntry · Job · TrainRun)
    ├── schemas/             요청·응답 DTO (리소스별 파일)
    ├── api/v1/
    │   ├── router.py        라우터 배선 단 한 곳
    │   └── endpoints/       리소스별 HTTP 경로 (predict/ 는 계열별 패키지)
    │                        데이터셋은 넷으로 나뉜다 — datasets(자리) · dataset_import
    │                        · dataset_images · dataset_export
    ├── services/            *_manager.py — 잡 수명·프로세스 풀·DB 갱신
    ├── workers/             *_worker.py · train_runner.py — 별도 프로세스 엔트리
    └── tests/               test_*.py

frontend/src/
├── main.tsx                 QueryClient · MantineProvider · BrowserRouter
├── App.tsx                  라우트 정의
├── api/                     서버 통신 — client.ts 는 재수출 배럴, 구현은 리소스별 파일
│   ├── client.ts            화면은 이것만 import 한다
│   ├── http.ts              fetch/XHR 을 아는 유일한 파일
│   ├── <리소스>.ts           datasets · projects · labs · labCrops · labels · classes · training …
│   └── test/                predict · compare
├── pages/                   라우트 1개 = 파일 1개 (Lab* 은 연구실, 나머지는 학습실)
├── layouts/                 공간별 셸 — ProjectLayout(학습실) · LabLayout(연구실)
├── components/<영역>/       화면 전용 컴포넌트 (dataset · editor · lab · test · train · upload)
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
| 리소스 하나가 300줄을 넘을 때 | `endpoints/<리소스>/` 패키지 + `__init__.py` 에서 묶기 | `router.py` 는 그대로 — 밖에서 보면 모듈 하나다 |
| 요청·응답 DTO | `app/schemas/<리소스>.py` | 엔드포인트의 `response_model` 로 연결 |
| DB 테이블 | `app/models/__init__.py` (새 파일 만들지 않는다) | — |
| 별도 프로세스 엔트리 | `app/workers/<도메인>_worker.py` | 모듈 최상단 함수 + picklable 인자 |
| 테스트 | `app/tests/test_<대상>.py` | — |
| 화면(라우트) | `frontend/src/pages/<이름>Page.tsx` | `App.tsx` 에 라우트 추가 |
| 화면 전용 컴포넌트·훅 | `frontend/src/components/<영역>/` | — |
| 서버 호출 | `frontend/src/api/client.ts` 에 함수 + 응답 인터페이스 | 컴포넌트에서 `fetch` 직접 호출 금지 |

## 자리가 애매하면 묻는다

`services/` vs `workers/` 의 경계는 **아직 확정되지 않았다**([아키텍처](architecture.md) 참고).
새 장시간 잡의 자리가 애매하면 **추측해서 두지 말고 사용자에게 묻는다.**

`domain/` vs `ml/` 은 해소됐다 — 두 폴더 모두 사라졌고, 기준은 **모델을 쓰는지가 아니라 import
방향**이다: `app/`·`infra/` 를 import 하지 않으면 `lib/`.

## 이행 중이다

`lib/` 과 `infra/` 는 `app/domain/`·`app/ml/` 의 잡탕 상태를 걷어내려고 만든 자리다.
**두 폴더 모두 비워져 사라졌다** — 순수 계산은 전부 `lib/` 의 주제 패키지에 있고, `app/` 에는
HTTP·DB·프로세스 관리만 남았다.

백엔드와 프론트 모두 한 차례 정리를 마쳤다. 다음에 손댈 곳은 그때 가장 큰 파일을 보고
정한다 — 지금 기준으로는 `TrainRunDetailPage.tsx`(약 600줄)와 `LabelEditorPage.tsx`(약 490줄)다.

무관한 변경에서 나머지를 함께 옮기지 않는다. 새 코드는 새 자리에 두고, 기존 코드는 그 기능을
손볼 때 같이 옮긴다.
