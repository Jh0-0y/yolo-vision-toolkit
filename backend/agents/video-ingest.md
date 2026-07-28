# Video Ingest

> 상태: 구현됨 · 핵심 파일: `app/api/v1/endpoints/videos.py`, `app/services/video_manager.py`, `app/domain/video.py`

## 목적

동영상을 업로드해 프레임을 샘플링하여 프로젝트 `raw/`로 추출한다. 추출 이후는
**기존 라벨링 파이프라인을 그대로 재사용**(이미지처럼 취급). 진행 상황은 SSE로 스트리밍.

## 요구사항

- `target_fps`/`max_frames`/`start_sec`/`end_sec`/`dedup`/`dedup_threshold` 파라미터로 샘플링 제어.
- 프레임 dedup — 연속 저장 프레임 간 32×32 그레이 평균차 유사도 ≥ threshold이면 skip.
- 순차 프레임 넘버링(`{stem}_00001.jpg`, 저장 순서 기준).
- 취소(CANCEL 센티넬), 재샘플(`resample`), 진행 SSE.
- 확장자 화이트리스트(`VIDEO_EXTS`), fps/frames 양수 검증.
- **CPU/IO 전용** — GPU executor와 분리된 ThreadPoolExecutor(max_workers=2)에서 실행해 라벨링·학습을 막지 않음.

## 데이터·저장 구조

- DB: **없음**(Project 존재만 확인, 순수 파일 기반).
- 디스크: `projects_dir/{pid}/videos/{video_id}.{ext}`(원본), `{video_id}.json`(메타: id/filename/ext/stem/params), 추출 프레임 → `projects_dir/{pid}/raw/`.
- 진행/취소: `jobs_dir/{video_id}/progress.jsonl` + `CANCEL` (공용 `read_progress` 재사용).

## API (prefix `/projects/{project_id}/videos`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `.../videos` | 업로드+추출 시작 (201, `{video_id,filename,status}`) — `videos.py:64` |
| POST | `.../videos/{video_id}/resample` | 새 파라미터로 재추출 — `videos.py:108` |
| POST | `.../videos/{video_id}/cancel` | 추출 취소 (`{cancelled}`) — `videos.py:142` |
| GET | `.../videos/{video_id}/events` | SSE 진행 스트림 — `videos.py:149` |

## 처리 흐름

`upload_video`(검증 `_params`/`VIDEO_EXTS` → 파일·메타 저장) → `video_manager.submit`
(task_dir 준비, `ThreadPoolExecutor.submit(extract_frames, ...)`) → `domain/video.py:extract_frames`
(cv2 디코드 → 샘플링/dedup → `cv2.imwrite` → progress.jsonl phase 이벤트). SSE는 `video_events` → `read_progress` 폴링.

## 핵심 파일·함수

- `app/api/v1/endpoints/videos.py:64` `upload_video`, `:36` `_sanitize_stem`, `:42` `_params`, `:149` `video_events`
- `app/services/video_manager.py:34` `submit`, `:59` `cancel`, `:68` `is_active`, `:17` `task_dir`
- `app/domain/video.py:40` `extract_frames`, `:34` `_similarity`, `:19` `ExtractParams`

## 엣지 케이스·주의

- 취소: 큐 상태면 `future.cancel()`, 실행 중이면 CANCEL 파일 touch → 워커가 매 프레임 확인(`video.py:91`).
- resample은 in-flight면 409. 단 `upload_video`는 중복 방지 없음(같은 파일 재업로드 시 새 video_id).
- dedup은 **연속 저장 프레임 간만** 비교 → 전역 중복 제거 아님.
- `_sanitize_stem`으로 stem 정규화 → 다른 비디오가 같은 sanitized stem이면 raw/에서 프레임 파일명 충돌(덮어씀) 가능.
- `video_manager`는 프로세스 상주 싱글톤 → 서버 재시작 시 `_futures` 상태 소실(progress.jsonl만 남음).
- 가변 프레임레이트(VFR): `CAP_PROP_FPS` 부정확 가능. 손상 파일은 열기 실패 시 오류.
- 취소·진행 계약 상세는 [jobs-and-progress.md](jobs-and-progress.md) 참고.
