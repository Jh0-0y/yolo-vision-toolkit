# AGENTS.md

`yolo-vision-toolkit` 의 에이전트 공용 가이드. 상세 규칙은 [`docs/agents/`](docs/agents/) 아래 토픽별로 나뉘어 있어, 작업에 필요한 문서만 열면 된다.

각 토픽 문서는 frontmatter(`title`, `scope`, `applies_to`, `related`)와 한 줄 TL;DR 로 시작한다. Map 에서 항목을 찾아 문서를 열고, TL;DR 로 적용 여부를 판단한 뒤 읽는다. 맞는 항목이 없으면 [프로젝트 개요](docs/agents/overview.md) 와 [아키텍처](docs/agents/architecture.md) 부터 본다.

## Map

### 개요
- [프로젝트 개요 & 스택](docs/agents/overview.md) — 무엇을 하는 앱인지 + 언어·라이브러리 버전; 처음 들어오거나 버전을 확인할 때
- [아키텍처](docs/agents/architecture.md) — 두 공간(연구실·학습실) · 프로세스 셋 · 백엔드 계층과 의존 방향; 새 코드의 자리를 정할 때
- [디렉터리 구조](docs/agents/structure-dir.md) — 저장소 트리와 새 파일을 둘 자리; 파일을 만들거나 찾을 때

### 코드 컨벤션
- [계층 경계](docs/agents/conventions/layer-boundaries.md) — torch lazy import · DB는 API 프로세스만 · 워커 picklable; **백엔드 코드를 만질 때마다**
- [명명 규칙](docs/agents/conventions/naming.md) — 파일·클래스·DTO·훅 이름과 주석 언어; 이름을 지을 때
- [백엔드 API 라우트](docs/agents/conventions/backend-api-route.md) — router.py 등록 · 프리픽스 · 오류 코드; 엔드포인트를 추가할 때
- [데이터셋 규약](docs/agents/conventions/datasets.md) — 데이터셋이 전부 갖는다 · 수치 파생 · 분할 안정성 · 내보내기 3종 · 학습 토큰; 학습실 코드를 만질 때
- [잡과 진행률](docs/agents/conventions/jobs-and-progress.md) — progress.jsonl · CANCEL · SSE; 오래 걸리는 작업을 만들 때
- [프론트 데이터 접근](docs/agents/conventions/frontend-data.md) — client.ts 단일 진입 · Query/스토어 경계; 서버 데이터를 다룰 때

### 라이브러리
- [adaptive-crop](docs/agents/libs/adaptive-crop.md) — 크롭 좌표 계산의 경계와 어댑터 규칙, 좌표계; 크롭 코드를 만질 때
- [크롭 계획 · 튜닝 · crop.json](docs/agents/libs/crop-plan.md) — 호출 흐름 · 크롭 크기(px) · 타깃 색 · 튜닝 노브 · 산출 형태; 크롭 런을 다룰 때
- [adaptive-crop 설치 · 업그레이드](docs/agents/libs/adaptive-crop-upgrade.md) — 비공개 저장소 인증과 태그 올리는 절차; 설치가 막히거나 버전을 올릴 때

### 운영
- [빌드 · 실행](docs/agents/build-run.md) — dev.sh · uv · docker 역할분담 · 환경변수; 앱을 띄우거나 의존성을 만질 때
- [테스트 · 검증](docs/agents/testing.md) — 고친 뒤 반드시 돌릴 명령; 코드를 고친 뒤
- [데이터 배치](docs/agents/data-layout.md) — DATA_DIR 트리 · DB인가 파일인가 · 손대지 않을 것; 파일을 읽고 쓸 때

### 워크플로
- [브랜치](docs/agents/workflow/branching.md) — develop 에서 개발, main 은 PR 로만; 작업을 시작할 때
- [커밋 메시지](docs/agents/workflow/commit-messages.md) — 허용 접두사 여섯 개와 형식; 커밋할 때
- [PR](docs/agents/workflow/pr.md) — base 브랜치와 올리기 전 검증; develop 을 main 에 올릴 때

## 작업별 경로

- **HTTP 엔드포인트 추가** → [백엔드 API 라우트](docs/agents/conventions/backend-api-route.md) → [명명 규칙](docs/agents/conventions/naming.md) → [계층 경계](docs/agents/conventions/layer-boundaries.md) → [프론트 데이터 접근](docs/agents/conventions/frontend-data.md)
- **오래 걸리는 작업 추가** → [잡과 진행률](docs/agents/conventions/jobs-and-progress.md) → [계층 경계](docs/agents/conventions/layer-boundaries.md) → [데이터 배치](docs/agents/data-layout.md)
- **화면 추가·수정** → [프론트 데이터 접근](docs/agents/conventions/frontend-data.md) → [명명 규칙](docs/agents/conventions/naming.md) → [테스트 · 검증](docs/agents/testing.md)
- **크롭 관련 수정** → [adaptive-crop](docs/agents/libs/adaptive-crop.md) → [크롭 계획 · 튜닝 · crop.json](docs/agents/libs/crop-plan.md) → [adaptive-crop 설치 · 업그레이드](docs/agents/libs/adaptive-crop-upgrade.md)
- **데이터셋·검수·분할·내보내기·학습 수정** → [데이터셋 규약](docs/agents/conventions/datasets.md) → [데이터 배치](docs/agents/data-layout.md) → [잡과 진행률](docs/agents/conventions/jobs-and-progress.md)
- **작업 시작 · 마무리** → [브랜치](docs/agents/workflow/branching.md) → [테스트 · 검증](docs/agents/testing.md) → [커밋 메시지](docs/agents/workflow/commit-messages.md) → [PR](docs/agents/workflow/pr.md)

## 규칙 적용 원칙

이 규칙들은 **새로 쓰는 코드와 대폭 수정하는 기존 코드**에 적용된다. 아직 규칙을 따르지 않는 기존 코드를 무관한 변경에서 함께 리팩터링하지 않는다 — 발견하면 알리고 넘어간다.

예시가 기존 코드에 없는 형태를 보여주더라도, 그 예시는 새 코드가 지향할 목표이지 기존 코드를 소급 수정하라는 지시가 아니다.

**확정되지 않은 것은 추측하지 않는다.** `services/` vs `workers/` 의 경계처럼 문서가 "아직 정해지지 않았다"고 적어 둔 것은 실제로 정해지지 않은 것이다. 비슷해 보이는 파일을 흉내 내지 말고 사용자에게 묻는다.

## 사람용 문서와의 관계

`README.md` 와 `docs/` 아래 다른 문서는 **사람이 읽는 문서**다. 규칙의 근거로 삼지 않고, 이 문서들에서 링크하지도 않는다. 에이전트가 알아야 할 것은 전부 `docs/agents/` 안에 있다.

다만 **바꿀 때는 양쪽을 함께 고친다.** 실행 명령·포트·환경변수·디렉터리 구조·계층 규칙을 변경하면 `README.md` 와 `docs/agents/` 를 **같은 커밋에서** 갱신한다. 둘은 링크로 이어져 있지 않아 자동으로 동기화되지 않는다.

## 문서 추가·수정

새 토픽 문서를 만들면 **같은 커밋 안에서 세 군데**를 고친다.

1. 위 **Map 에 한 줄** — `- [제목](경로) — 무엇을 담는지; 언제 읽는지`
2. 알아볼 수 있는 작업이 트리거라면 **작업별 경로에 한 줄**
3. **`related:` 를 양방향으로** — 새 문서가 이웃을 가리키고, 이웃도 새 문서를 가리키게

등록되지 않은 문서는 존재하지 않는 문서다. "나중에 링크하겠다"로 남기지 않는다.

문서 하나는 **한 토픽, 60줄 이내**를 목표로 한다. 넘으면 토픽이 두 개이거나 튜토리얼로 번진 것이다.
