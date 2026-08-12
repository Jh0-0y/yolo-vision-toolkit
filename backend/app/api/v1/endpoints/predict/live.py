"""라이브 프리뷰 — 검출은 한 번만, 크롭 좌표는 튜닝할 때마다 다시 계산.

비싼 절반(모델 추론)은 잡으로 한 번 돌려 검출을 캐시하고, 싼 절반(좌표 계산)은
`/plan` 이 동기로 처리한다. 그래서 노브를 돌릴 때마다 재추론 없이 크롭 박스가
즉시 갱신된다. 캐시로 오버레이 영상을 구워 내려받는 것도 여기 있다.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import (
    detectors_cfg,
    job_event_stream,
    model_pt,
)
from app.core.config import settings
from app.db import get_session
from app.schemas.predict import TestJobStart
from app.services.test_jobs import sweep_old_live, test_job_manager
from lib.formats import VIDEO_EXTS

router = APIRouter(prefix="/predict", tags=["predict"])


def _live_dir(detect_id: str) -> Path:
    """Resolve a live-preview cache dir, blocking path traversal via the id shape."""
    if not detect_id.isalnum():  # uuid4().hex
        raise HTTPException(422, "Invalid detect id")
    return settings.test_dir / "live" / detect_id


@router.post("/live", response_model=TestJobStart, status_code=201)
async def start_live(
    model_ids: str = Form(...),
    conf: float | None = Form(None),  # None → crop 검출 기본값 0.10
    imgsz: int = Form(1920),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    sampling_interval_ms: int | None = Form(None),  # None → 기본값 (100ms)
    detectors: str = Form("[]"),  # 검출기 엔트리 목록 (JSON)
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Run the expensive detection pass ONCE and cache the per-sample detections +
    a browser-playable preview. The crop trajectory is then computed on demand via
    `/live/{id}/plan` (cheap, no inference) as the user tunes parameters."""
    ext = Path(file.filename or "v").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type: {ext}")
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, model_pt(session, mid, project_id)) for mid in ids]

    await run_in_threadpool(sweep_old_live)  # keep test_dir/live tidy
    job_id = uuid.uuid4().hex
    work = settings.test_dir / "live" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"source{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    cfg = {
        "work": str(work),
        "source": str(src),
        "specs": specs,
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
        "sampling_interval_ms": sampling_interval_ms,
        "detectors": detectors_cfg(session, detectors, project_id),
    }
    await run_in_threadpool(test_job_manager.submit_live, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/live/{job_id}/events")
async def live_events(job_id: str):
    return await job_event_stream(job_id)


@router.get("/live/{job_id}/video")
def live_video(job_id: str):
    """The browser-playable H.264 preview the client plays back under the overlay."""
    preview = _live_dir(job_id) / "preview.mp4"
    if not preview.exists():
        raise HTTPException(404, "Preview not ready")
    return FileResponse(preview, media_type="video/mp4")  # FileResponse handles Range


@router.get("/live/{job_id}/result")
def live_result(job_id: str):
    """Detection cache + geometry meta for the client: video dimensions/fps/duration
    and the raw per-sample detections (for the ByteTrack overlay). The crop trajectory
    itself comes from `/plan` so tuning changes need no re-fetch of this payload."""
    work = _live_dir(job_id)
    meta_path = work / "meta.json"
    det_path = work / "detected.json"
    if not meta_path.exists() or not det_path.exists():
        raise HTTPException(404, "Detection result not ready")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    detections = json.loads(det_path.read_text(encoding="utf-8"))
    return {"detect_id": job_id, **meta, "detections": detections}


@router.post("/live/{detect_id}/render", response_model=TestJobStart, status_code=201)
async def start_live_render(detect_id: str, body: dict = Body(default={})):
    """캐시된 검출 + 현재 튜닝으로 오버레이 영상을 렌더한다 (추론 없음).

    body: {overrides: {...}, toggles: {obj_boxes, draw_crop_box, show_dead_zone,
    show_center_line, show_highlight}}. 진행은 /live/{job_id}/events로 구독."""
    work = _live_dir(detect_id)
    if not (work / "detected.json").exists() or not (work / "preview.mp4").exists():
        raise HTTPException(404, "Detection cache not found")
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(422, "overrides must be an object")
    toggles = body.get("toggles") or {}
    if not isinstance(toggles, dict):
        raise HTTPException(422, "toggles must be an object")

    job_id = uuid.uuid4().hex
    cfg = {"work": str(work), "overrides": overrides, "toggles": toggles}
    await run_in_threadpool(test_job_manager.submit_live_render, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/live/{detect_id}/render-video")
def live_render_video(detect_id: str):
    """렌더된 오버레이 영상 다운로드."""
    render = _live_dir(detect_id) / "render.mp4"
    if not render.exists():
        raise HTTPException(404, "Rendered video not ready")
    return FileResponse(render, media_type="video/mp4", filename="crop_overlay.mp4")


@router.post("/live/{detect_id}/plan")
async def live_plan(detect_id: str, body: dict = Body(default={})):
    """Recompute the crop trajectory from cached detections + tuning overrides.

    This is the cheap half of the pipeline (no model/torch) — it runs synchronously
    so the client can call it on every knob change and redraw the crop box at once.
    """
    work = _live_dir(detect_id)
    det_path = work / "detected.json"
    meta_path = work / "meta.json"
    if not det_path.exists() or not meta_path.exists():
        raise HTTPException(404, "Detection cache not found")
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(422, "overrides must be an object")
    collect_debug = bool(body.get("collect_debug", False))

    def _compute() -> dict:
        # adaptive_crop imports cv2/numpy but not torch on this path (좌표 계산만).
        from adaptive_crop import plan_from_detections
        from adaptive_crop.detect.io import load_detections

        from lib.crop import plan as crop_adapter

        detected = load_detections(json.loads(det_path.read_text(encoding="utf-8")))
        video = crop_adapter.video_info_from_meta(
            json.loads(meta_path.read_text(encoding="utf-8"))
        )
        # validate=False: 튜닝 노브를 돌릴 때마다 부르는 경로라 한 조합이 500이 되면
        # 안 된다. 클라이언트는 원본 좌표를 실제 프레임 크기에 맞춰 스케일한다.
        result = plan_from_detections(
            detected,
            crop_adapter.crop_spec_for(video.width, video.height),
            video,
            config=crop_adapter.resolve_clip_config(overrides),
            collect_debug=collect_debug,
            validate=False,
        )
        return result.to_dict(include_internal=True)  # 오버레이용 samples/debug 포함

    return await run_in_threadpool(_compute)
