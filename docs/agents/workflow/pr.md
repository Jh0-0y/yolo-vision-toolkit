---
title: PR
scope: "**"
applies_to: develop 을 main 에 올릴 때
related:
  - ./branching.md
  - ./commit-messages.md
  - ../testing.md
---

# PR

> `develop` → `main` 은 PR 로만 간다. 올리기 전에 검증을 통과시킨다.

- **base 는 `main`, head 는 `develop`.**
- CI 가 없다. **아래 검증은 사람 대신 반드시 직접 돌린다.**

## 올리기 전 검증

```bash
cd backend && uv run pytest
```

```bash
cd frontend && npm run build && npm run lint
```

- 백엔드만 고쳤어도 프론트 타입이 깨질 수 있다(응답 필드 변경). **API 를 건드렸으면 양쪽 다 돌린다.**
- 실패한 채로 PR 을 올리지 않는다. 고칠 수 없으면 무엇이 실패했는지 그대로 보고한다.

## PR 본문

- 무엇을 왜 바꿨는지, 그리고 **어떻게 확인했는지**를 적는다.
- 규칙을 바꿨다면 `docs/agents/` 도 같은 PR 에 들어 있어야 한다. 코드보다 뒤처진 규칙 문서는 능동적 위험이다.

## 확인 없이 하지 않는 것

- **PR 생성·병합**은 사용자 지시가 있을 때만 한다.
- `git push --force` · `git reset --hard` · 브랜치 삭제는 **명시 승인 전에 실행하지 않는다.**
