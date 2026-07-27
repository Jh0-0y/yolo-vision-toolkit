# yolo-vision-toolkit

YOLO 기반 객체 탐지 툴박스 — 앙상블 자동 라벨링 + YOLO 범용 학습을 독립 모듈로 제공

## 구성

| 모듈 | 설명 | 상태 |
|---|---|---|
| 오토라벨링 툴킷 | 학습된 YOLO 모델 1~N개로 앙상블 추론 → 이미지별 라벨 기록 → 사람 검수(박스 에디터, 검수 완료 체크) → train/val 데이터셋 내보내기 | ✅ 완료 |
| 학습 툴킷 | 웹 UI에서 데이터셋(내보내기 또는 zip 업로드)·베이스 모델 선택 → Ultralytics 학습 실행 → 실시간 mAP/loss 차트 → best.pt 레지스트리 등록 | ✅ 완료 |

전체 루프: **라벨링 → 검수 → 내보내기 → 학습 → 학습된 모델로 다시 라벨링**. 기준 모델은 yolo26n이며, 모델 레지스트리에서 공식 모델(yolo26/12/11 계열)을 바로 내려받거나 직접 학습한 `.pt`를 업로드할 수 있습니다.

> 구버전(confirmed/review 버킷) 데이터를 쓰던 경우 일회성 마이그레이션이 필요합니다: `cd backend && uv run python scripts/migrate_buckets.py` (미리보기는 `--dry-run`).

## 개발 환경 실행 (Mac)

```bash
# 최초 1회 의존성 설치
cd backend && uv sync && cd ../frontend && npm install && cd ..

# 백엔드 + 프론트엔드 한 번에 실행 (Ctrl+C로 둘 다 종료)
./dev.sh
# 백엔드 http://localhost:8010 · 프론트 http://localhost:5173 (API는 8010으로 프록시)
```

<details>
<summary>따로 실행하려면</summary>

```bash
# 백엔드 (Python 3.12, uv)
cd backend && uv run uvicorn app.main:app --port 8010

# 프론트엔드 (별도 터미널)
cd frontend && npm run dev
```
</details>

## 오토라벨링 CLI

```bash
cd backend
uv run python cli.py label \
  --models modelA.pt modelB.pt \
  --images ./raw_images \
  --out ./output \
  --conf 0.4
```

- 모델별 클래스가 달라도 클래스명 기준으로 union됩니다 (A={person,car}, B={car,dog} → {person,car,dog}).
- 복수 모델이 공유하는 클래스는 Weighted Box Fusion으로 병합됩니다.
- 출력: `output/labels/*.txt` (이미지별 YOLO 라벨, 빈 파일 = 네거티브), `output/classes.json` (글로벌 클래스 레지스트리)

## 배포 — 윈도우 GPU PC (Docker) + 외부 접속

> **Docker 구성은 윈도우(WSL2) + NVIDIA GPU 전용입니다.** GPU를 무조건 붙이도록 되어 있어 GPU 없는 호스트에선 컨테이너가 뜨지 않습니다. 맥이나 CPU 머신은 Docker 대신 위의 `./dev.sh` 로컬 실행을 쓰세요(가속: 맥=MPS, 윈도우=CUDA, 없으면 CPU 자동 선택).

사전 준비(1회): Docker Desktop 설치 → Settings에서 **WSL2 기반 엔진** 활성화 → 최신 NVIDIA 드라이버 설치 → `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`로 GPU 인식 확인.

방법은 둘 중 하나:

**A) 서버에서 직접 빌드** (소스를 서버에 두는 경우)
```bash
git clone <repo> && cd yolo-vision-toolkit
cp .env.example .env   # DATA_DIR·CACHE_DIR 경로 수정
docker compose up -d --build
```

**B) GHCR 이미지로 배포** (내 PC에서 굽고, 서버는 받아서 실행 — 소스 불필요)
```bash
# ── 내 PC(맥 포함): 이미지 굽고 GHCR에 올리기 ──
# compose 에 platform: linux/amd64 가 박혀 있어 맥(arm64)에서 빌드해도
# 윈도우(amd64)용 이미지가 나온다. (첫 빌드는 에뮬레이션이라 다소 느림)
# TAG 로 버전 지정 — 매 배포마다 다른 값을 주면 된다(생략 시 latest).
echo <GITHUB_TOKEN> | docker login ghcr.io -u Jh0-0y --password-stdin  # PAT: write:packages
TAG=v1.2.3 docker compose build
TAG=v1.2.3 docker compose push

# ── 서버: docker-compose.yml + .env 두 파일만 두고 ──
cp .env.example .env   # DATA_DIR·CACHE_DIR 경로 + TAG(실행할 버전) 수정
docker login ghcr.io -u Jh0-0y --password-stdin   # 비공개 패키지일 때만
docker compose pull
docker compose up -d
```
> 이미지 태그는 `TAG`(`.env` 또는 명령어 앞에 지정)로 정해집니다 — 이름 자체는 `docker-compose.yml`의 `image:`(`ghcr.io/jh0-0y/yvt-backend|frontend:${TAG}`)로 고정. 서버는 `.env`의 `TAG`에 적은 버전을 pull 합니다. 소스 없이 뜨려면 `pull` 을 먼저 돌린 뒤 `up` 하세요(빌드 시도 방지).

- 접속: **http://<PC주소>:3000** (프론트 nginx가 `/api`를 백엔드로 프록시하므로 열어야 할 포트는 **3000 하나**입니다. 8000은 호스트 로컬 전용.)
- 외부 접속: 공유기 포트포워딩(외부포트 → PC:3000) + 윈도우 방화벽에서 3000 인바운드 허용.
- 저장 경로는 `.env`에서 지정합니다(`cp .env.example .env`): `DATA_DIR`(영구 저장 — HDD 권장, 백업은 이 폴더만), `CACHE_DIR`(학습용 SSD 캐시 — 학습 시 데이터셋을 여기로 복사해 돌리고 끝나면 자동 삭제). 스테이징이 불필요하면 둘을 같은 경로로 두면 됩니다.

> ⚠️ **보안**: 현재 로그인/인증 기능이 없습니다. 인터넷에 직접 노출하면 누구나 데이터·모델에 접근하고 학습을 실행할 수 있으니, 가급적 VPN(Tailscale 등)이나 신뢰할 수 있는 네트워크 안에서만 열어두세요.

## 테스트

```bash
cd backend && uv run pytest
```
