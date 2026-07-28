# Inference & Test (playground)

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/predict.py`, `app/services/infer_manager.py`, `app/services/test_jobs.py`, `app/workers/infer_worker.py`, `app/workers/compare_worker.py`, `app/workers/annotate_worker.py`, `app/ml/predict.py`, `app/ml/ensemble.py`, `app/ml/evaluate.py`

## 목적

학습된 모델을 **데이터셋을 건드리지 않고** 실험하는 놀이터 — 단일/배치 이미지 예측,
다중 모델 A/B 앙상블 비교, 영상 어노테이션(추적/크롭 오버레이), 업로드한 YOLO 테스트셋에 대한
mAP·P/R/F1 평가. **DB에 아무것도 쓰지 않는다**(전부 transient `data/test/`).

## 요구사항

- 예측은 `DETECT_FLOOR=0.05`까지 모든 박스 반환 → UI 슬라이더가 **재추론 없이** 클라이언트 필터링. `cfg.conf`는 슬라이더 초기값(하드 필터 아님).
- 다중 모델 시 per-class WBF 융합; 단일 모델 소유 클래스는 통과, 2+ 모델 공유 클래스만 융합.
- **워엄 모델 레지던시**: 반복 테스트/스크러빙 시 가중치 재로딩 회피, 다중 모델 동시 상주(A/B·앙상블). 180초 idle 후 reaper가 워커·CUDA 컨텍스트 회수.
- 학습 시작 시 추론 워커 evict(학습에 클린 디바이스 양보). Apple MPS에서 학습 활성 시 테스트 추론은 CPU로 강등.
- compare의 mAP는 display conf와 독립적으로 전체 예측셋(detector floor까지)에서 COCO 방식 계산; P/R/F1은 display conf 부분집합.
- 예측 클래스는 데이터셋 클래스에 **이름(정규화)으로** 매핑; 미정의 클래스는 드롭하지 않고 id=−1 FP로 유지.

## 데이터·저장 구조

- DB: **없음**. compare 결과는 `jobs_dir/{job_id}/result.json` + `images_manifest.json`.
- 디스크: 이미지 업로드 `test/uploads/{uuid}{ext}`(예측 후 unlink), 어노테이션 `test/annotate/{job_id}/`(source, out.mp4, crop.json), 비교 `test/compare/{job_id}/`(upload.zip, dataset/). TTL 3600초, 새 잡 시작 시 sweep. → [storage-layout.md](storage-layout.md).

## API (prefix `/predict`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/predict` | 단일 이미지 예측(multipart) — `predict.py:70` |
| GET | `/predict/residents` | 상주 모델 목록 — `predict.py:117` |
| POST | `/predict/residents/{model_id}` | 모델 워엄 로드 (201) — `predict.py:122` |
| DELETE | `/predict/residents/{model_id}` | 모델 언로드 (204) — `predict.py:133` |
| POST | `/predict/annotate` | 영상 어노테이션 잡 시작 (201) — `predict.py:141` |
| GET | `/predict/annotate/{job_id}/events` · `/result` · `/crop` | SSE / mp4(Range) / crop.json |
| POST | `/predict/compare` | 모델 비교 잡 시작 (201, YOLO 테스트셋 zip) — `predict.py:237` |
| GET | `/predict/compare/{job_id}/events` · `/result` · `/images/{idx}` | SSE / 결과 JSON / 이미지 서빙 |

## 처리 흐름

- 단일 예측: `predict.py:predict` → `infer_manager.predict`(device 정책·레지던시) → `infer_worker.run_predict`(워엄 모델) → `ml/predict.py:predict_image`(모델별 predict → `ClassRegistry` 매핑 → `ensemble.merge_detections`).
- 어노테이션: `start_annotate` → `test_job_manager.submit_annotate`(video 풀, ProcessPool spawn) → `annotate_worker.run_annotate`(ByteTrack `model.track` + 별도 `trackcrop.analyze_video` 패스 → mp4v → ffmpeg H.264). 크롭 출력형식은 `crop_output`(Form): `"label"`=풀프레임에 9:16 사각형 오버레이(기본, object_tracking 박스와 합성) / `"video"`=프레임을 세로 9:16으로 실제 컷 → 박스·사각형 없는 깔끔한 세로 클립(object_tracking 무시). 컷/그리기 지오메트리는 `app/domain/crop_render.py`.
- 비교: `start_compare`(`_extract_compare_dataset` unzip) → `submit_compare`(eval 풀) → `compare_worker.run_compare`(per-image `predict_image` + `evaluate` mAP/P/R/F1).

## 핵심 파일·함수

- `app/services/infer_manager.py:39` executor/reaper 라이프사이클, `:107` load/predict/residents
- `app/workers/infer_worker.py:23` `_RESIDENT`, `:37` load_model, `:83` run_predict
- `app/ml/predict.py:31` `predict_image`
- `app/ml/ensemble.py:60` `fuse_class`/`merge_detections`
- `app/ml/evaluate.py:83` `average_precision`/`map_from_accumulated`(101-point COCO), `:15` `match_frame`
- `app/workers/compare_worker.py:81` `run_compare`
- `app/workers/annotate_worker.py:53` `run_annotate`, `:245` `_to_h264` — 크롭 컷/오버레이는 `app/domain/crop_render.py`(`crop_width_for`/`build_trajectory`/`center_at`/`draw_window`/`cut_window`)로 분리
- `app/services/test_jobs.py:56` `TestJobManager`(단일워커 풀 "video"/"eval")

## 엣지 케이스·주의

- **ffmpeg 필수** — 없으면 annotate 잡 실패(unplayable mp4v 방지). annotate 소스 영상은 `finally`에서 항상 unlink.
- annotate/compare 잡은 DB에 없음 → job_id 유효성은 `.isalnum()` 검사로만(path traversal 차단).
- reaper가 워엄 워커를 죽여도 `_residents` 미러는 mirror-only(워커를 깨우지 않음).
- compare: data.yaml에 클래스명 없으면 모든 예측이 FP + warning.
- tracking/crop은 본질상 single-model — 여러 모델 선택 시 첫 번째만 사용.
- crop 오버레이 좌표는 trackcrop의 소스 픽셀 기준 → annotate 워커가 실제 프레임폭으로 remap → [track-crop.md](track-crop.md).
- 워커 패턴·취소·진행 계약은 [`../CLAUDE.md`](../CLAUDE.md), [jobs-and-progress.md](jobs-and-progress.md).
