# 설계: 동영상 → 오토라벨링 입력 (프레임 추출)

> 상태: 구현됨 · 최종 수정 2026-07-22
>
> 구현: `app/core/video.py`(추출), `app/jobs/video_manager.py`(스레드풀 러너),
> `app/api/videos.py`(업로드/SSE/resample/cancel),
> `frontend/src/components/videos/VideoUploadModal.tsx`(UI)

## 목표

동영상을 업로드하면 프레임을 샘플링해 `raw/`에 이미지로 저장하고, 그
이후는 **기존 오토라벨링 파이프라인을 그대로 재사용**한다. 탐지 데이터셋의
원천은 대부분 영상(CCTV·드론·촬영 클립)이므로 입력단에 프레임 추출 단계를
하나 더 붙이는 것이 핵심.

## 전제 — GPU를 쓰지 않는다

프레임 추출은 CPU/IO 작업이다. 오토라벨링·학습이 공유하는 GPU 직렬화
게이트(`ProcessPoolExecutor(max_workers=1)`, `app/jobs/runner.py`)를 타면
안 되고, **별도 백그라운드(threadpool)** 로 돌려 GPU 작업을 막지 않는다.

## 저장 구조

프로젝트 폴더에 `videos/` bucket을 추가한다.

```
projects/{project_id}/
├── videos/            # 원본 동영상 보관 (재샘플링용)
│   └── {video_id}.mp4
├── raw/               # 추출된 프레임 (기존과 동일)
│   ├── clipA_000000.jpg
│   ├── clipA_000030.jpg
│   └── ...
└── thumbs/            # 기존 썸네일 로직 재사용
```

원본을 `videos/`에 남기는 이유: 샘플링 파라미터를 바꿔 **재업로드 없이
다시 추출**하기 위함.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/api/projects/{id}/videos` | 동영상 업로드 + 추출 시작 (비동기) → `task_id` |
| `GET` | `/api/projects/{id}/videos/{video_id}/events` | SSE 진행률 (`progress.jsonl` 패턴 재사용) |
| `POST` | `/api/projects/{id}/videos/{video_id}/resample` | 다른 파라미터로 재추출 |

## 파라미터

```jsonc
{
  "target_fps": 2,        // 초당 몇 장 뽑을지 (원본 fps 무관, 핵심 옵션)
  "max_frames": 2000,     // 폭발 방지 상한
  "start_sec": 0,         // 구간 지정 (선택)
  "end_sec": null,
  "dedup": true,          // 인접 유사 프레임 제거
  "dedup_threshold": 0.92 // 유사도 임계값 (선택)
}
```

## 처리 단계

1. 업로드 → `videos/{video_id}.mp4` 저장 (확장자 검증: mp4/mov/avi/mkv/webm)
2. `cv2.VideoCapture`로 열어 `CAP_PROP_FPS` 읽음 → `step = round(source_fps / target_fps)`
3. **스트리밍 읽기**(전 프레임을 메모리에 올리지 않음) — `step` 간격마다 저장
4. `dedup`이 켜져 있으면 직전 저장 프레임과 **프레임 차분(mean abs diff)**
   또는 perceptual hash 비교 → 임계값 이하이면 스킵
5. `max_frames` 도달 시 중단
6. 저장된 프레임에 대해 **기존 썸네일 생성 로직 호출** → raw 인덱싱

## 설계 근거

- **`target_fps` 방식**: 사용자는 원본이 몇 fps인지 모른다. "초당 N장"이
  직관적이며, step은 내부에서 계산.
- **dedup 필수**: 영상은 인접 프레임이 거의 동일 → 안 거르면
  near-duplicate가 데이터셋 다양성을 망치고 라벨링 노동만 늘린다.
- **cv2 사용**: ultralytics 의존성으로 **이미 설치됨** → 추가 의존성 0.
  (ffmpeg가 있으면 더 빠르나 외부 의존성이라 선택적 최적화)

## 엣지 케이스

- 가변 프레임레이트(VFR): `CAP_PROP_FPS` 부정확 가능 → 타임스탬프 기반 폴백
- 긴 영상: `max_frames` + 스트리밍으로 메모리/디스크 보호
- 손상 파일: 열기 실패 시 422
- 파일명 충돌: `{video_stem}_{frame_idx:06d}.jpg`로 유일성 확보

## 기존 코드 연결점

- `app/api/projects.py:88` `upload_images` 옆에 형제 엔드포인트로 추가
- 추출 이후는 완전히 기존 흐름(raw 인덱싱·썸네일·오토라벨링) 재사용 →
  파이프라인 뒷단은 손대지 않음

## 열린 질문

- dedup 방식: 프레임 차분(간단·빠름) vs perceptual hash(견고·약간 느림)
- 원본 동영상 보관 기간/용량 정책 (재샘플링 편의 vs 디스크)
