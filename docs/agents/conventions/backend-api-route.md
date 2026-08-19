---
title: 백엔드 API 라우트
scope: backend/app/**/*.py
applies_to: HTTP 엔드포인트를 추가하거나 고칠 때
related:
  - ./naming.md
  - ./layer-boundaries.md
  - ./frontend-data.md
  - ./datasets.md
---

# 백엔드 API 라우트

> 새 경로는 `router.py` 에 등록해야 살아난다. 엔드포인트를 만들 때 읽는다.

## 등록 — 잊으면 404 다

새 파일 `app/api/v1/endpoints/<리소스>.py` 를 만들고 `app/api/v1/router.py` 의 **두 목록 모두**에 넣는다.
import 목록에만 넣고 include 목록을 빠뜨리면 경로가 뜨지 않는다.

```python
router = APIRouter(prefix="/projects/{project_id}/videos", tags=["videos"])
```

- **`prefix` 에 `/api/v1` 을 붙이지 않는다.** `api_router` 가 `settings.api_prefix` 로 붙인다.
- 경로를 문자열로 하드코딩하지 않는다. 프리픽스의 단일 소스는 `settings.api_prefix` 뿐이다.
- 프로젝트에 속한 리소스는 `/projects/{project_id}/<리소스>` 아래 둔다.
- `tags` 는 리소스명 하나.

## 리소스가 커지면 패키지로 나눈다

한 리소스가 300줄을 넘고 안에서 **서로 관계없는 계열**로 갈리면 파일 대신 패키지를 만든다.
`predict/` 가 그 예다 — 단발추론 · annotate · live · compare 넷이 `/predict` 아래 함께 산다.

```
endpoints/predict/
├── __init__.py    router = APIRouter(); 하위 라우터를 include 만 한다
├── common.py      계열들이 함께 쓰는 조각
└── <계열>.py       router = APIRouter(prefix="/predict", tags=["predict"])
```

- **접두사·태그는 하위 라우터가 든다.** 묶는 `__init__.py` 의 라우터는 비워 둔다 —
  `@router.post("")` 처럼 경로가 빈 라우트는 접두사 없는 라우터에 실을 수 없다.
- `router.py` 는 **그대로 둔다.** 밖에서 보면 여전히 모듈 하나이므로 등록도 한 줄이다.
- 나눠도 **경로와 OpenAPI 태그는 그대로여야 한다.** 나눈 뒤 `/openapi.json` 을 비교해 확인한다.

## 규칙

- 응답은 **`schemas/` 의 DTO** 를 `response_model` 로 지정한다. dict 를 그대로 돌려주지 않는다.
- 생성은 `status_code=201`, 본문 없는 삭제는 204.
- 세션은 `session: Session = Depends(get_session)` 로 받는다. 직접 열지 않는다.
- 프로젝트 스코프 엔드포인트는 **맨 먼저 프로젝트 존재를 확인한다.**

```python
if session.get(Project, project_id) is None:
    raise HTTPException(404, "Project not found")
```

## 오류 코드

| 코드 | 언제 |
|---|---|
| `404` | 리소스가 없다 |
| `422` | 입력이 틀렸다 — 값 범위, 참조 대상 없음, 선택 안 함 |

`HTTPException(404, "Project not found")` 처럼 **위치 인자**로 쓴다. 메시지는 영어 한 문장.

## 무거운 일은 엔드포인트에서 하지 않는다

- 추론·학습·인코딩은 `services/` 의 매니저에 제출하고 **즉시 응답한다** → [잡과 진행률](jobs-and-progress.md)
- `torch`·`ultralytics` 를 엔드포인트에서 import 하지 않는다 → [계층 경계](layer-boundaries.md)

## 프론트까지가 한 작업이다

경로를 추가·변경했으면 `frontend/src/api/client.ts` 에 대응 함수와 응답 인터페이스를 **같은 변경에서** 넣는다 → [프론트 데이터 접근](frontend-data.md)
