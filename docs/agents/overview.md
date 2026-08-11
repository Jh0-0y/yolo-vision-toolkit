---
title: 프로젝트 개요 & 스택
scope: "**"
applies_to: 이 저장소를 처음 다루거나 언어·라이브러리 버전을 확인할 때
related:
  - ./architecture.md
  - ./build-run.md
---

# 프로젝트 개요 & 스택

> YOLO 데이터셋을 만들고·라벨링하고·학습하고·검증하는 웹 툴킷. 스택과 버전을 확인할 때 읽는다.

- 저장소는 **백엔드(`backend/`)와 프론트엔드(`frontend/`) 두 덩어리**다. 배포는 nginx가 프론트를 서빙하고 `/api` 를 백엔드로 프록시한다.
- **로그인·인증이 없다.** 신뢰할 수 있는 내부 네트워크 전용으로 설계됐다. 인증을 전제한 코드를 쓰지 않는다.
- 작업 단위는 **프로젝트**다. 이미지·라벨·클래스·모델·학습 기록이 프로젝트별로 격리된다.

## 버전 (고정된 것)

| | |
|---|---|
| Python | `>=3.12,<3.13` — 상한이 있다. 3.13 문법을 쓰지 않는다 |
| 패키지 관리 | **uv** (`backend/pyproject.toml`, `uv.lock`). pip 로 설치하지 않는다 |
| 백엔드 | FastAPI · SQLModel(SQLite) · sse-starlette · pydantic-settings · Ultralytics 8.3 |
| 크롭 계산 | `adaptive-crop` — **비공개 저장소, 태그로 고정.** [adaptive-crop](libs/adaptive-crop.md) 참고 |
| 프론트 | React 19 · TypeScript ~6.0 · Vite 8 |
| UI | Mantine 9 (`@mantine/core` · `charts` · `dropzone` · `notifications`) · Tabler Icons |
| 상태·통신 | TanStack Query 5 · Zustand 5 · React Router 7 |
| 캔버스·차트 | Konva / react-konva (라벨 에디터) · Recharts |
| 린터 | **oxlint** (ESLint 아님) |

## 가속기

`YVT_DEVICE` 기본값은 `auto` 이고 `cuda > mps > cpu` 순으로 고른다(`app/core/config.py:resolve_device`).
맥은 MPS, 윈도우·리눅스는 CUDA로 잡힌다. **디바이스를 코드에 하드코딩하지 않는다** — 항상 `resolve_device()` 를 거친다.
