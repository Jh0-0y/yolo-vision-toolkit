# AGENTS.md — 에이전트 진입점

이 파일은 코드 에이전트가 이 저장소에서 작업을 시작할 때 읽는 **지도**다.
"무엇을 어디서 찾는지"를 라우팅하고, 상세는 각 문서로 위임한다.

## 이 프로젝트는 무엇인가

YOLO 기반 **오토라벨링 + 학습 툴킷**. 이미지/영상/외부 데이터셋을 입력받아
오토라벨링 → 리뷰 → export → 학습 → 모델 레지스트리 등록의 선순환을 한 웹앱에서 돈다.

- 백엔드: Python 3.12 · FastAPI · SQLModel(SQLite) · Ultralytics(YOLO) · SSE · uv
- 프론트: React 19 · TypeScript · Vite · Mantine · TanStack Query · Zustand · Konva · Recharts

## 전체 파이프라인

```
[입력 A] 이미지/zip ─┐
[입력 B] 동영상 ──── 프레임추출 ─┴─→ raw/ → 오토라벨링 → 리뷰 → export ─┐
                                                                        │
[입력 C] 외부 데이터셋 zip ──── 검증/정규화 ──→ raw+labels ─────────────┤
                                                                        ▼
                                                          학습 (소스 무관) → .pt
                                                                        │
                                                          모델 레지스트리 등록 ─┐
                                                                        ▲       │
                                                                        └───────┘ (선순환)
```

## 문서 구조 — 어디를 읽을까

에이전트는 **먼저 아키텍처 규칙([backend/CLAUDE.md](backend/CLAUDE.md))을 읽고**,
작업 대상 기능의 상세 문서([backend/agents/](backend/agents/))로 진입한다.

| 문서 | 무엇을 담나 |
|---|---|
| [backend/CLAUDE.md](backend/CLAUDE.md) | **계층 아키텍처·컨벤션·규칙** — "어디에 무엇을 두는가"(디렉터리 결정표, 워커 패턴, API 컨벤션, 깨지기 쉬운 커플링) |
| [backend/agents/](backend/agents/) | **기능별 요구사항·처리 흐름·엣지케이스** — "무엇을·어떻게" |
| README.md | 사용자용 기능 설명·실행/배포 방법·환경 변수 |

> 원칙: CLAUDE.md(규칙)와 agents/(기능)는 **내용을 분리**한다. 기능 문서는 계층 규칙을 반복하지 않고 CLAUDE.md를 링크로 참조한다.

## 기능 → 문서 라우팅

| 하려는 작업 | 문서 |
|---|---|
| 프로젝트·이미지 업로드·데이터셋 갤러리 | [projects-and-storage](backend/agents/projects-and-storage.md) |
| 동영상 → 프레임 추출 | [video-ingest](backend/agents/video-ingest.md) |
| 외부 YOLO zip 임포트 | [dataset-upload](backend/agents/dataset-upload.md) |
| 클래스 add/rename/delete·reindex | [classes](backend/agents/classes.md) |
| 라벨 에디터(박스 read/write) | [label-editor](backend/agents/label-editor.md) |
| 오토라벨링(앙상블) | [auto-labeling](backend/agents/auto-labeling.md) |
| 데이터셋 export(zip 빌드) | [export](backend/agents/export.md) |
| 학습 런 | [training](backend/agents/training.md) |
| 모델 등록/다운로드 | [models](backend/agents/models.md) |
| 예측·모델 비교·영상 어노테이션·mAP | [inference-and-test](backend/agents/inference-and-test.md) |
| 9:16 크롭 좌표 산출 | [track-crop](backend/agents/track-crop.md) |
| 진행률·취소(SSE·CANCEL, 파일 IPC) | [jobs-and-progress](backend/agents/jobs-and-progress.md) |
| `data/` 저장 구조·경로·설정 | [storage-layout](backend/agents/storage-layout.md) |

## 프론트엔드

`frontend/src/pages/*`가 각 기능에 1:1로 대응(Upload/Dataset/LabelEditor/Classes/Models/Train/TrainingHistory/TrainRunDetail/Exports/Test/Home).
API 호출은 `frontend/src/api/client.ts`에 집약, 전역 상태는 `stores/`(Zustand). 백엔드 계약이 진실의 원천이며 프론트는 얇은 소비자다.

## 검증 루틴

- 백엔드: `uv run python -m py_compile $(git ls-files 'app/**/*.py')` → `uv run pytest` → `uv run uvicorn app.main:app --app-dir backend --port 8010` 부팅 → `/api/health`·`/api/v1/system/device` 200, `/openapi.json` 전 경로 `/api/v1`.
- 프론트: `npx tsc --noEmit && npx oxlint src`.
- 학습/추론 등 서브프로세스 경로는 **실제 1회 실행**으로만 검증됨(워커 모듈 문자열은 런타임에만 발현).

상세 규칙과 근거는 [backend/CLAUDE.md](backend/CLAUDE.md)를 참고.
