---
title: 빌드 · 실행
scope: "**"
applies_to: 앱을 띄우거나 의존성을 설치·변경할 때
related:
  - ./testing.md
  - ./overview.md
  - ./libs/adaptive-crop-upgrade.md
---

# 빌드 · 실행

> 로컬 실행과 배포 이미지 빌드는 완전히 다른 경로다. 앱을 띄우거나 의존성을 만질 때 읽는다.

## 로컬 개발

```bash
# 최초 1회
cd backend && uv sync && cd ../frontend && npm install && cd ..

# 백엔드 + 프론트 동시 실행 (Ctrl+C 한 번으로 둘 다 종료)
./dev.sh
```

- 백엔드 **8010** · 프론트 **5173**. 프론트가 `/api` 를 8010 으로 프록시한다.
- 따로 띄우려면 `cd backend && uv run uvicorn app.main:app --reload --port 8010` / `cd frontend && npm run dev`.
- `uv sync` 가 **비공개 저장소**에서 `adaptive-crop` 을 받아오므로 GitHub 인증이 되어 있어야 한다 → [adaptive-crop 설치·업그레이드](libs/adaptive-crop-upgrade.md)

## 의존성

- 백엔드는 **uv 로만** 만진다: `uv add <패키지>` / `uv remove`. `pip install` 하지 않는다.
- 프론트는 `npm install <패키지>`.
- **의존성 추가·제거·버전 변경은 사용자 승인을 받고 한다.**

## 배포 (윈도우 GPU PC)

역할이 나뉜다.

| 파일 | 역할 |
|---|---|
| `scripts/build.sh` | 이미지 빌드 + GHCR 푸시. 항상 `linux/amd64` |
| `docker-compose.yml` | **배포 전용 — `build:` 가 없다.** GHCR 이미지를 pull 해서 실행만 한다 |

- compose 구성은 **GPU를 무조건 요구한다.** GPU 없는 호스트(맥 포함)에서는 뜨지 않는다 — 의도된 설계다.
- 맥에서 docker 로 띄우려고 compose 를 고치지 않는다. 맥은 `dev.sh` 전용이다.
- 외부에 여는 포트는 프론트 하나(기본 3000). nginx 가 `/api` 를 내부 8000 으로 프록시한다.

## 환경 변수

백엔드 설정은 `YVT_` 프리픽스 환경변수 또는 `.env` 로 덮어쓴다(`app/core/config.py`의 `Settings`).
새 설정값이 필요하면 **`Settings` 에 필드를 추가하고 그것만 읽는다** — `os.environ` 을 직접 읽지 않는다.

| 자주 쓰는 것 | 기본값 |
|---|---|
| `YVT_DATA_DIR` | `<repo>/data` — 모든 영구 데이터의 뿌리 |
| `YVT_SSD_CACHE_DIR` | `None` — 학습 스테이징용 빠른 스크래치. 없으면 스테이징 생략 |
| `YVT_DEVICE` | `auto` (`cuda > mps > cpu`). `cpu` \| `mps` \| `0` 지정 가능 |
| `YVT_API_PREFIX` | `/api/v1` — 버전 프리픽스의 단일 소스 |
| `YVT_CORS_ORIGINS` | `localhost:5173`, `localhost:3000` |

경로는 `settings.data_dir` 등 **프로퍼티로 파생**된다. 경로를 문자열로 조립하지 않는다 → [데이터 배치](data-layout.md)
