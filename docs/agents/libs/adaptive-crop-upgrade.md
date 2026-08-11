---
title: adaptive-crop 설치 · 업그레이드
scope: backend/pyproject.toml
applies_to: adaptive-crop 을 설치하거나 버전을 올릴 때
related:
  - ./adaptive-crop.md
  - ./crop-plan.md
  - ../build-run.md
---

# adaptive-crop 설치 · 업그레이드

> 비공개 저장소이고 태그로 고정돼 있다. 설치가 막히거나 버전을 올릴 때 읽는다.

## 고정 방식

버전은 `backend/pyproject.toml` 에 **git 태그로 못박는다.** 브랜치를 가리키지 않는다.

```toml
"adaptive-crop @ git+https://github.com/Jh0-0y/adaptive-crop@0.0.1-rc1-20260811",
```

## 설치 인증

- **로컬**: SSH 키로 clone 중이면 한 번만
  `git config --global url."git@github.com:".insteadOf "https://github.com/"`
- **Docker**: 토큰을 URL 에 박지 **않는다** — `.dist-info/direct_url.json` 에 남는다.
  `--secret id=gh_pat` 로 주입한다. `docker/backend.Dockerfile` 참고.

## 업그레이드 절차

1. adaptive-crop 저장소에서 고치고 **태그를 올린다.**
2. `backend/pyproject.toml` 의 태그를 바꾸고 `cd backend && uv sync`.
3. `ClipPlanConfig` 필드가 바뀌었으면 **세 곳을 맞춘다**:
   - `TuningPanel.tsx` 의 노브 목록과 placeholder `def` 값
   - `client.ts` 의 `TrackcropOverrides` 타입
   - 새 필드를 UI 에 노출하지 않기로 했다면 [adaptive-crop](adaptive-crop.md) 의 "노출하지 않은 필드" 목록에 추가
4. `cd backend && uv run pytest` — 어댑터 계약 테스트(`test_crop_adapter.py` · `test_crop_spec.py`)가 번역 규칙이 깨졌는지 잡는다.

**버전 변경은 사용자 승인을 받고 한다.** 튜닝 기본값이 함께 바뀌면 지금까지 맞춰 둔 결과가 달라진다.

## 딸려 오는 것

설치 시 `opencv-python-headless` 가 함께 들어와 `cv2` 가 그 버전으로 확정된다.
워커는 GUI 함수를 쓰지 않아 문제없지만, `cv2.imshow` 류를 새로 쓰면 깨진다.

알고리즘 자체의 테스트는 adaptive-crop 저장소가 담당한다. 여기서 중복해 검증하지 않는다.
