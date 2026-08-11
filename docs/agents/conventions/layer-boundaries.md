---
title: 계층 경계
scope: backend/app/**/*.py
applies_to: 백엔드 코드를 쓰거나 import 를 추가할 때
related:
  - ../architecture.md
  - ./jobs-and-progress.md
  - ../data-layout.md
---

# 계층 경계

> API 프로세스는 torch를 로드하지 않고, 워커는 DB를 모른다. 백엔드 코드를 만질 때마다 읽는다.

## `torch` · `ultralytics` 는 **절대** 모듈 최상단에서 import 하지 않는다

**항상 쓰는 함수 안에서 import 한다.** 예외 없다.

```python
def run_labeling(cfg, progress, cancel_check):
    from ultralytics import YOLO      # ○ 함수 안
    ...
```

최상단에 두면 그 모듈을 import 하는 순간 **API 프로세스가 CUDA 컨텍스트를 잡는다.**
워커가 쓸 VRAM 을 API 가 미리 먹고, uvicorn 기동이 몇 초씩 느려진다.

`cv2` 와 `adaptive_crop` 도 같은 기준을 따른다 — **무거운 진입점은 함수 안**,
타입·상수만 쓰는 곳은 최상단이어도 된다(`app/ml/crop.py` 의 `ClipPlanConfig` 등).

## DB는 API 프로세스에서만

- `Session`·`select`·`session_scope` 는 `api/` 와 `services/` 에서만 쓴다.
- **워커는 값을 받아 계산하고 값을 돌려준다.** 워커 안에서 DB를 열지 않는다.
- 잡 결과를 DB에 반영하는 것은 부모 프로세스(`services/`)의 완료 콜백이 한다.

## 워커 엔트리는 picklable 해야 한다

프로세스 풀은 `spawn` 컨텍스트로 뜬다. 그래서:

- 엔트리는 **모듈 최상단 함수**여야 한다. 클로저·람다·인스턴스 메서드를 제출하지 않는다.
- 인자는 **평범한 dict·str·int** 로 넘긴다. `Path` 는 문자열로 바꿔 넘기고 워커 안에서 되살린다.
- 워커가 던지는 예외도 picklable 이어야 한다. 커스텀 예외에 복잡한 객체를 담지 않는다.

## `ml/` · `domain/` 은 순수 계산이다

- 프로세스·DB·HTTP·FastAPI 를 **모른다.** `HTTPException` 을 던지지 않는다.
- 진행상황이 필요하면 `progress` 콜백을, 취소가 필요하면 `cancel_check` 콜백을 **인자로 받는다.** 파일을 직접 들여다보지 않는다.
- 둘의 차이: `ml/` 은 모델을 쓰는 계산, `domain/` 은 모델을 쓰지 않는 계산. **이 경계가 애매한 새 모듈은 사용자에게 묻는다.**

## 예외를 던지는 자리

| 계층 | 던지는 것 |
|---|---|
| `api/` | `HTTPException(404, "...")` · `HTTPException(422, "...")` |
| `services/` · `workers/` · `ml/` · `domain/` | 자기 예외 또는 표준 예외. **`HTTPException` 은 여기서 던지지 않는다** |

워커가 실패하면 `progress.jsonl` 에 `{"phase": "error", "msg": ...}` 를 남기고 다시 raise 한다 — 스트림을 보는 쪽과 부모 프로세스 양쪽이 알아야 한다.
