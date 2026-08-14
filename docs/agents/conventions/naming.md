---
title: 명명 규칙
scope:
  - backend/app/**/*.py
  - frontend/src/**/*.{ts,tsx}
applies_to: 파일·클래스·함수 이름을 짓거나 바꿀 때
related:
  - ../structure-dir.md
  - ./backend-api-route.md
---

# 명명 규칙

> 이름만 보고 어느 계층인지 알 수 있어야 한다. 이름을 지을 때 읽는다.

## 백엔드

| 대상 | 규칙 | 예 |
|---|---|---|
| 엔드포인트 파일 | 리소스 **복수형** snake_case | `projects.py` · `videos.py` · `labels.py` |
| 서비스 | `<도메인>_manager.py` + 모듈 하단에 소문자 싱글턴 | `train_manager.py` → `train_manager` |
| 워커 | `<도메인>_worker.py` + 엔트리 함수 `run_<작업>` | `label_worker.py` → `run_label_job` |
| DB 테이블 | 파스칼 **단수**, `models/__init__.py` 안에 | `Project` · `Job` · `ModelEntry` · `TrainRun` |
| 요청·응답 DTO | `<리소스><역할>` | `ProjectCreate` · `ProjectOut` · `ModelPatch` · `ClassIn` |
| 테스트 | `test_<대상>.py` | `test_tiling.py` |
| 내부 전용 함수 | `_` 접두사 | `_require_project` · `_sanitize_stem` |

DTO 역할 접미사는 이미 쓰이는 것을 따른다: `Create`(생성 요청) · `Out`(응답) · `In`(단순 입력) ·
`Patch`(부분 수정) · `Params`(설정 묶음). **새 접미사를 만들지 않는다.**

## 프론트

| 대상 | 규칙 | 예 |
|---|---|---|
| 페이지 | `<이름>Page.tsx` | `DatasetPage.tsx` |
| 컴포넌트 | PascalCase `.tsx` | `BBoxCanvas.tsx` · `JobIndicator.tsx` |
| 훅 | `use<이름>.ts` | `useLiveJob.ts` |
| 순수 모듈 | camelCase `.ts` | `colors.ts` · `metrics.ts` |
| 스토어 | `<도메인>Store.ts` + `use<도메인>Store` | `jobStore.ts` → `useJobStore` |

## 필드 이름은 백엔드 표기를 그대로 쓴다

API 응답은 **snake_case** 이고, 프론트 타입도 그대로 받는다(`data_dir` · `created_at` · `job_id`).
받아서 camelCase 로 바꾸지 않는다 — 두 표기가 섞이면 어느 쪽이 서버 값인지 알 수 없게 된다.

예외는 **crop.json 하나**다. 외부 계약 스키마라 camelCase 로 나간다 → [adaptive-crop](../libs/adaptive-crop.md)

## 주석과 docstring은 한국어로 쓴다

- 새로 쓰는 주석·docstring 은 **한국어**로 쓴다.
- 식별자(변수·함수·클래스명)와 로그·예외 메시지는 **영어**다.
- 기존 영어 주석을 한국어로 바꾸지 않는다. 그 파일을 크게 고칠 때가 아니면 그대로 둔다.
- 주석은 **왜** 를 적는다. 무엇을 하는지는 코드가 말한다.
