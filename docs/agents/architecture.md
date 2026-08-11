---
title: 아키텍처
scope: "**"
applies_to: 새 코드를 어느 프로세스·계층에 둘지 정할 때
related:
  - ./structure-dir.md
  - ./conventions/layer-boundaries.md
  - ./conventions/jobs-and-progress.md
---

# 아키텍처

> 프로세스가 셋이고, 그 경계가 이 저장소의 가장 강한 제약이다. 새 코드의 자리를 정할 때 읽는다.

## 프로세스 셋

| 프로세스 | 하는 일 | 절대 하지 않는 것 |
|---|---|---|
| **API** (uvicorn) | HTTP · DB(SQLite) · 잡 제출 · SSE 스트리밍 | **torch/CUDA를 로드하지 않는다** |
| **워커** (`spawn` 프로세스 풀) | 추론·학습·인코딩 등 무거운 계산 | **DB에 접근하지 않는다** |
| **브라우저** | 화면·업로드·SSE 수신 | 백엔드 없이 상태를 만들지 않는다 |

- 둘은 **파일로만 통신한다.** 진행상황은 `progress.jsonl`, 취소는 `CANCEL` 센티넬 파일 → [잡과 진행률](conventions/jobs-and-progress.md)
- 워커에 넘기는 인자는 **picklable** 해야 한다 — 모듈 최상단 함수 + 평범한 dict.
- 이 경계를 깨는 방법은 하나뿐이다: 모듈 최상단에서 `torch`·`ultralytics` 를 import 하는 것. → [계층 경계](conventions/layer-boundaries.md)

## 백엔드 계층과 의존 방향

```
api/v1/endpoints  →  services  →  workers  ─┐
       │                                     ├→  ml  ·  domain
       └──────────────→  schemas  ·  models  ┘
```

| 계층 | 담는 것 |
|---|---|
| `api/v1/endpoints/` | HTTP 경로. 입력 검증, `HTTPException`, DTO 변환 |
| `api/v1/router.py` | **라우터 배선 단 한 곳.** 새 엔드포인트는 여기 두 목록에 등록해야 살아난다 |
| `services/` | 잡 수명 관리(프로세스 풀 소유), DB 갱신, 싱글턴 매니저 |
| `workers/` | 별도 프로세스에서 도는 엔트리 함수. `torch`·`ultralytics` 는 여기 함수 안에서만 |
| `ml/` | 모델을 쓰는 순수 계산 (앙상블·평가·추론·크롭 어댑터) |
| `domain/` | 모델을 쓰지 않는 순수 계산 (파일 IO·이미지·라벨·렌더) |
| `schemas/` | 요청·응답 DTO (pydantic `BaseModel`) |
| `models/` | DB 테이블 (SQLModel `table=True`) — `__init__.py` 한 파일 |
| `db/` | 엔진·세션·경량 마이그레이션 |
| `core/` | 설정(`settings`)·디바이스 해석 |

- `ml/`·`domain/` 은 **프로세스도 DB도 HTTP도 모른다.** 값을 받아 값을 돌려준다.
- DB 접근은 `api/`·`services/` 에서만 한다.

## 아직 확정되지 않은 경계

다음 두 가지는 **규칙이 정해져 있지 않다.**

- `services/` vs `workers/` — 새 장시간 잡의 기본 자리
- `domain/` vs `ml/` — 둘 다 순수 계산인데 무엇으로 가르는지

**새 모듈의 자리가 이 둘 중 하나로 애매하면 추측하지 말고 사용자에게 묻는다.**
기존 파일을 흉내 내 아무 데나 두면 경계가 더 흐려진다.

## 프론트

- 서버 통신은 `src/api/client.ts` **하나만** 거친다 → [프론트 데이터 접근](conventions/frontend-data.md)
- 페이지 이동에도 살아남아야 하는 장시간 작업은 `stores/jobStore.ts` 가 소유한다.
