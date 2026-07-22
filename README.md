# yolo-vision-toolkit

YOLO 기반 객체 탐지 툴박스 — 앙상블 자동 라벨링 + YOLO 범용 학습을 독립 모듈로 제공

## 구성

| 모듈 | 설명 | 상태 |
|---|---|---|
| 오토라벨링 툴킷 | 학습된 YOLO 모델 1~N개로 앙상블 추론 → confirmed / review 버킷 분리 → 사람 검증(박스 에디터) → train/val 데이터셋 내보내기 | ✅ 완료 |
| 학습 툴킷 | 웹 UI에서 데이터셋·베이스 모델 선택 → Ultralytics 학습 실행 → 실시간 mAP/loss 차트 → best.pt 레지스트리 등록 | ✅ 완료 |

전체 루프: **라벨링 → 리뷰 → 내보내기 → 학습 → 학습된 모델로 다시 라벨링**. 기준 모델은 yolo26n이며, 모델 레지스트리에서 공식 모델(yolo26/12/11 계열)을 바로 내려받거나 직접 학습한 `.pt`를 업로드할 수 있습니다.

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
  --conf-confirm 0.60
```

- 모델별 클래스가 달라도 클래스명 기준으로 union됩니다 (A={person,car}, B={car,dog} → {person,car,dog}).
- 복수 모델이 공유하는 클래스는 Weighted Box Fusion으로 병합되고, 모델 간 합의 수(agree_count)가 판정에 사용됩니다.
- 출력: `output/confirmed/{images,labels}` (자동 확정), `output/review/*.json` (사람 검증 대상), `output/classes.json` (글로벌 클래스 레지스트리)

## 배포 — 윈도우 GPU PC (Docker) + 외부 접속

사전 준비(1회): Docker Desktop 설치 → Settings에서 **WSL2 기반 엔진** 활성화 → 최신 NVIDIA 드라이버 설치 → `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`로 GPU 인식 확인.

```bash
git clone <repo> && cd yolo-vision-toolkit

# GPU (윈도우 + WSL2 + NVIDIA)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# CPU만 있는 머신은
docker compose up -d --build
```

- 접속: **http://<PC주소>:3000** (프론트 nginx가 `/api`를 백엔드로 프록시하므로 열어야 할 포트는 **3000 하나**입니다. 8000은 호스트 로컬 전용.)
- 외부 접속: 공유기 포트포워딩(외부포트 → PC:3000) + 윈도우 방화벽에서 3000 인바운드 허용.
- 데이터는 `./data` 볼륨에 저장됩니다. 백업은 이 폴더만 챙기면 됩니다.
- 대용량 이미지 폴더는 `.env`의 `EXTRA_DATA_DIR`로 컨테이너에 읽기전용 마운트할 수 있습니다.

> ⚠️ **보안**: 현재 로그인/인증 기능이 없습니다. 인터넷에 직접 노출하면 누구나 데이터·모델에 접근하고 학습을 실행할 수 있으니, 가급적 VPN(Tailscale 등)이나 신뢰할 수 있는 네트워크 안에서만 열어두세요.

## 테스트

```bash
cd backend && uv run pytest
```
