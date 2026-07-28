# Backend Feature Docs (`agents/`)

이 폴더는 **기능(feature) 단위**로 "무엇을 요구하고 어떻게 동작하는가"를 정리한다.
에이전트가 작업 전 해당 기능 문서를 먼저 읽고 코드로 진입하는 것을 목표로 한다.

## 문서 분리 원칙

- **[`../CLAUDE.md`](../CLAUDE.md)** = 아키텍처·계층·컨벤션 규칙 ("어디에 무엇을 두는가"). 횡단 규칙은 여기 하나에만 있다.
- **`agents/*.md`** (이 폴더) = 기능별 요구사항·처리 흐름·엣지케이스 ("무엇을·어떻게").
- 계층 규칙(워커 패턴, API 컨벤션, 깨지기 쉬운 커플링 등)은 이 폴더에서 **반복하지 않고** `../CLAUDE.md`를 링크로 참조한다.

## 문서 목록

| 문서 | 기능 | 주요 계층 |
|---|---|---|
| [projects-and-storage.md](projects-and-storage.md) | **Projects & Storage** — 프로젝트/이미지 업로드/데이터셋 갤러리 | endpoints/projects, domain |
| [video-ingest.md](video-ingest.md) | **Video Ingest** — 동영상 → 프레임 추출 | endpoints/videos, services, domain |
| [dataset-upload.md](dataset-upload.md) | **Dataset Upload** — 외부 YOLO zip 임포트 | endpoints/projects, domain/class_registry |
| [classes.md](classes.md) | **Classes** — 클래스 레지스트리 CRUD | endpoints/classes, domain/classes |
| [label-editor.md](label-editor.md) | **Label Editor** — per-image 라벨 read/write | endpoints/labels, domain/labels·yolo_io |
| [auto-labeling.md](auto-labeling.md) | **Auto-Labeling** — 앙상블 자동 라벨링 | endpoints/jobs, ml, workers |
| [export.md](export.md) | **Export** — YOLO 데이터셋 zip 빌드 | endpoints/exports, services, domain |
| [training.md](training.md) | **Training** — ultralytics 학습 런 | endpoints/training, services, workers |
| [models.md](models.md) | **Models Registry** — 모델 등록/다운로드 | endpoints/models |
| [inference-and-test.md](inference-and-test.md) | **Inference & Test** — 예측/비교/어노테이션/mAP 놀이터 | endpoints/predict, services, workers, ml |
| [track-crop.md](track-crop.md) | **Track & Crop** — 9:16 크롭 좌표 산출 | ml/trackcrop |
| [jobs-and-progress.md](jobs-and-progress.md) | **Jobs & Progress** — 파일 기반 IPC (진행률·취소) | 횡단 |
| [storage-layout.md](storage-layout.md) | **Storage Layout** — `data/` 디렉터리 구조·설정 | core/config |

## 각 기능 문서 템플릿

```
# <Feature Name>            ← 영어 기능명
> 상태 · 핵심 파일 (endpoint / service / worker / ml·domain)

## 목적            — 이 기능이 해결하는 것
## 요구사항         — 보장해야 하는 것
## 데이터·저장 구조  — DB 테이블 / 디스크 버킷
## API             — 엔드포인트 표
## 처리 흐름         — endpoint → service → worker → ml/domain
## 핵심 파일·함수     — 진입점·주요 함수 (file:line)
## 엣지 케이스·주의   — 실패 모드, 관련 문서 링크
```

> 규약: **제목/기능명은 영어, 본문은 한국어.** 코드 참조는 `app/...py:line`(backend 기준 상대경로) 형식.
> 계층·워커 패턴·API 컨벤션은 [`../CLAUDE.md`](../CLAUDE.md)에서 확인.
