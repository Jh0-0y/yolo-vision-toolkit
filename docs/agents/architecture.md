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
api/v1/endpoints  →  services  →  workers
       │
       └──────────────→  schemas  ·  models

                    app/  ──▶  infra/     (잡 배관)
                      └──▶  lib/          (순수 기능)
```

`app/` 밖의 두 패키지는 **위를 향해 import 하지 않는다.**

| 패키지 | 담는 것 | 금지 |
|---|---|---|
| `lib/` | 순수 기능 (검출·크롭·라벨·영상·미디어·학습파일) | `app/`·`infra/` import. 잡·DB·HTTP 를 몰라야 한다 |
| `infra/` | 잡 배관 (진행률·취소·잡 디렉터리) | `app/` import. `settings` 도 모르므로 경로를 인자로 받는다 |

- `lib/` 가 진행률을 알려야 하면 **콜백으로 받는다** (`emit`·`on_progress`·`cancel_check`).
  그래야 웹 없이 CLI·배치에서 같은 계산을 쓸 수 있다.
- **주제 패키지끼리는 서로의 로직을 import 하지 않는다** — `lib/crop` 이 `lib/detect` 를 부르지 않고,
  검출 결과를 **인자로 받는다**. 연결은 호출자(파이프라인)가 한다. 그래서 층을 나눌 필요가 없고
  순환도 안 생긴다.
- 예외는 **로직 없는 공용 어휘** 하나뿐이다 — `lib/formats.py`(확장자 상수) 같은 최상위 모듈은
  어디서든 읽는다. 어휘가 흩어지면 "이 화면만 .tiff 를 못 받는" 어긋남이 생긴다.
- 한 패키지 **안에서는** 자유롭게 나눈다. `lib/crop/` 은 `geometry` 를 바닥에 두고
  `window`·`hud`·`highlight`·`cut` 이 그 좌표를 받아 쓴다 — 시각(ms)을 아는 건 `geometry` 뿐이라
  보간이 프레임당 한 번만 돈다.

| 계층 (`app/` 안) | 담는 것 |
|---|---|
| `api/v1/endpoints/` | HTTP 경로. 입력 검증, `HTTPException`, DTO 변환 |
| `api/v1/router.py` | **라우터 배선 단 한 곳.** 새 엔드포인트는 여기 두 목록에 등록해야 살아난다 |
| `services/` | 잡 수명 관리(프로세스 풀 소유), DB 갱신, 싱글턴 매니저 |
| `workers/` | 별도 프로세스에서 도는 엔트리 함수. 계산은 `lib/` 에서 가져다 **조립만** 한다 |

워커의 골격은 하나다 — **무엇을 켤지 정하는 곳을 한 군데로 모으고, 루프는 하나만 둔다.**
`annotate_worker` 가 그 예다: 궤적을 구하고 → 프레임 소스를 고르고 → 스테이지를 쌓고 →
한 루프로 돌린다. 새 오버레이는 스테이지 하나를 더하는 일이지 분기를 늘리는 일이 아니다.
| `schemas/` | 요청·응답 DTO (pydantic `BaseModel`) |
| `models/` | DB 테이블 (SQLModel `table=True`) — `__init__.py` 한 파일 |
| `db/` | 엔진·세션·경량 마이그레이션 |
| `core/` | 설정(`settings`)·디바이스 해석 |

- **계산은 `app/` 에 두지 않는다.** 옛 `app/domain/`·`app/ml/` 은 둘 다 비워져 사라졌고,
  순수 계산은 전부 `lib/` 의 주제 패키지에 있다. `app/` 에 남은 건 HTTP·DB·프로세스 관리뿐이다.
- DB 접근은 `api/`·`services/` 에서만 한다.

## 아직 확정되지 않은 경계

- `services/` vs `workers/` — 새 장시간 잡의 기본 자리는 **정해져 있지 않다.**
  애매하면 추측하지 말고 사용자에게 묻는다.

> `domain/` vs `ml/` 은 더 이상 문제가 아니다. 두 폴더를 없애고 내용을 `lib/` 의 주제 패키지로
> 보내면서 경계 자체가 사라졌다. **모델을 쓰는지는 기준이 아니다** — `lib/detect/` 는 ultralytics 를
> 쓴다. 기준은 하나다: **`app/`·`infra/` 를 import 하지 않으면 `lib/`.**

## 프론트

- 서버 통신은 `src/api/client.ts` **하나만** 거친다 → [프론트 데이터 접근](conventions/frontend-data.md)
- 페이지 이동에도 살아남아야 하는 장시간 작업은 `stores/jobStore.ts` 가 소유한다.
