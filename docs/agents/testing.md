---
title: 테스트 · 검증
scope: "**"
applies_to: 코드를 고친 뒤 검증하거나 테스트를 추가할 때
related:
  - ./build-run.md
  - ./workflow/pr.md
---

# 테스트 · 검증

> 무엇을 고쳤든 아래 명령으로 확인한 뒤에 "됐다"고 말한다. 코드를 고친 뒤 읽는다.

## 고쳤으면 반드시 돌린다

```bash
# 백엔드를 고쳤다면
cd backend && uv run pytest

# 프론트를 고쳤다면 — 둘 다
cd frontend && npm run build && npm run lint
```

- `npm run build` 는 `tsc -b` 를 포함한다. **타입 오류는 여기서만 잡힌다.**
- `npm run lint` 는 oxlint 다. ESLint 설정을 추가하지 않는다.
- 돌리지 않고 완료를 보고하지 않는다.

## 백엔드 테스트

자리가 둘이다 (`pyproject.toml` 의 `testpaths` 가 이 둘만 본다).

| 대상 | 자리 | 이름 |
|---|---|---|
| `app/` 안의 것 | `backend/app/tests/` | `test_<대상>.py` |
| `lib/`·`infra/` | `backend/tests/` | `test_lib_<주제>.py` · `test_infra_<주제>.py` |

- 두 폴더 모두 **`__init__.py` 가 있어야 한다** — 없으면 pytest 가 `backend/` 를 sys.path 에 넣지 않아 `import app` 이 깨진다.
- 한 파일씩 돌리려면 `cd backend && uv run pytest app/tests/test_tiling.py -q`.
- **순수 계산(`domain/`·`ml/`)을 우선 테스트한다.** 프로세스·GPU가 필요한 경로는 테스트가 어렵다.
- 크롭 **알고리즘** 테스트는 `adaptive-crop` 저장소가 담당한다. 여기에는 **어댑터 계약 테스트만** 둔다
  (`test_crop_adapter.py` · `test_crop_spec.py`) → [adaptive-crop](libs/adaptive-crop.md)

## 수동 확인이 필요할 때

`./dev.sh` 로 띄운 뒤 확인한다.

- 헬스: `curl -s http://localhost:8010/api/health`
- 경로 목록: `curl -s http://localhost:8010/openapi.json`
- Swagger UI: `http://localhost:8010/docs`

`data/` 아래 실제 프로젝트 데이터를 검증용으로 지우거나 덮어쓰지 않는다 → [데이터 배치](data-layout.md)
