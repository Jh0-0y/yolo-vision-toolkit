# Track & Crop

> 상태: 구현됨 · 핵심 파일: `app/ml/trackcrop/` 패키지 · 유일 호출처: `app/workers/annotate_worker.py`

## 목적

가로형(1920×1080) 스포츠 영상에서 공·선수를 추적해 세로형(9:16, 608×1080) Crop 창의 X 좌표
궤적을 계산한다. 이 모듈 자체는 **좌표(keyframes + samples)만** 산출한다(순수 계산, 서버/S3/Job 의존 없음).
그 좌표로 실제 프레임을 그리거나(오버레이) 잘라내는(컷) 소비 로직은 `app/domain/crop_render.py`에 있고,
`annotate_worker`가 `crop_output`("label"=오버레이 / "video"=세로 컷)에 따라 호출한다.

## 요구사항

- **순수 계산** — 파일 IO는 영상 경로 읽기만, DB·네트워크 없음.
- 100ms 격자 순차 샘플링(seek 반복 없음, 프레임 누락 없음).
- 추적은 ByteTrack 대신 predict + 자체 최근접 track_id 부여(10fps 빠른 공의 IoU 연계 실패 회피). 시간 연속성은 `BallTracker` 등속 예측이 담당.
- 모델 클래스 자동 매핑: 이름에 "ball"/"person"/"player" 포함 → object_type. **ball 클래스 없으면 로드 실패.**
- 결과 자체 검증 9규칙(crop 규격, keyframe≥2, 첫=0ms, 마지막=duration, 오름차순, X 범위 0..1312, 정수, 최대속도, NaN/inf 없음).

## 파이프라인 스테이지 (`pipeline.py:analyze_video` = 공개 엔트리)

1. `sample_frames` — 100ms 격자 프레임 추출 (`sampling.py`)
2. `Detector.track` — YOLO predict + `_IdAssigner` 최근접 track_id (`detection.py`)
3. `resolve_targets` — 시점별 Target X 결정: 공 → 공+선수 가중중심(0.7/0.3) → 선수군집(중앙값) → 직전 유지(≤1500ms) → 중앙 960 (`target.py`, `tracking.py`)
4. `stabilize` — outlier 제거 → gap 선형보간 → 속도제한 → 이동평균(창5) (`stabilization.py`)
5. `reduce_keyframes` — Crop X 변환(`to_crop_x`: center−304, clamp 0..1312) + RDP 단순화(epsilon 8px) (`keyframe.py`)
6. `build_crop_result` + `validate_crop_result` — CropResult 조립·검증 (`result.py`)

## 데이터·저장 구조

- DB 없음. 결과는 `CropResult` dataclass(`types.py`) → `to_json()`. annotate 워커가 `crop.json`으로 디스크에 기록.

## 핵심 파일·함수

- `app/ml/trackcrop/pipeline.py:25` `analyze_video(video_path, *, detector=None, model_path, device, imgsz, conf, validate)` — Detector 재사용 가능(모델 1회 로딩)
- `app/ml/trackcrop/detection.py:27` `_IdAssigner`, `:55` `Detector`
- `app/ml/trackcrop/tracking.py:21` `BallTracker`, `:66` `select_ball`, `:94` `player_group_center`
- `app/ml/trackcrop/target.py:25` `resolve_targets`
- `app/ml/trackcrop/stabilization.py` `remove_outliers`/`interpolate_gaps`/`limit_speed`/`smooth`
- `app/ml/trackcrop/keyframe.py:16` `to_crop_x`, `:26` `reduce_keyframes`/`_rdp_indices`
- `app/ml/trackcrop/result.py:13` `build_crop_result`, `:72` `validate_crop_result`
- `app/ml/trackcrop/constants.py` — SOURCE 1920×1080, CROP 608×1080, X_MAX 1312, SAMPLING 100ms, CENTER 960, weights 0.7/0.3
- `app/ml/trackcrop/types.py`(TargetType/Detection/TargetSample/Keyframe/CropResult), `errors.py`(ErrorCode/TrackCropError)

## 엣지 케이스·주의

- `constants.py`가 **1920×1080 하드코딩** → 다른 해상도 입력은 좌표가 소스 픽셀 기준이라 소비 측 remap 필요(annotate 워커가 실제 프레임폭으로 remap). → [inference-and-test.md](inference-and-test.md).
- `validate=True`(기본) 시 비-1080p·짧은 클립은 검증 실패 ValueError → annotate는 `validate=False`로 우회.
- ball 클래스 없는 모델은 `Detector.__init__`에서 즉시 예외.
- RDP 재귀 — 매우 긴 영상에서 깊은 재귀 가능성.
- 유일 호출처는 `annotate_worker.py`(crop_tracking 토글 시 lazy import). 계층 규칙상 `ml/`은 torch lazy·DB 미의존 → [`../CLAUDE.md`](../CLAUDE.md).
