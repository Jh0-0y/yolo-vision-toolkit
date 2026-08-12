"""영상 오버레이 렌더 — 클립 전체에 박스·크롭 창을 구워 mp4 로 낸다.

추론 없이 자르기만 하는 crop-cut 도 여기 둔다. 같은 잡 디렉터리
(`test_dir/annotate/{job_id}`)를 쓰고 events·result 엔드포인트를 그대로 재사용하기
때문이다 — 시작하는 방법만 다르다.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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
from app.services.test_jobs import sweep_old_annotations, test_job_manager
from lib.formats import VIDEO_EXTS

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/annotate", response_model=TestJobStart, status_code=201)
async def start_annotate(
    model_ids: str = Form(...),
    conf: float | None = Form(None),  # None → 파이프라인별 기본값 (crop 0.10 / object 0.4)
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    object_tracking: bool = Form(True),  # ByteTrack boxes + IDs + trails
    crop_tracking: bool = Form(True),  # adaptive-crop vertical 9:16 crop window
    crop_output: str = Form("label"),  # "none" = JSON only | "label" = overlay | "video" = cut clip
    draw_crop_box: bool = Form(True),  # label: 9:16 사각형(+데드존·센터선)
    show_dead_zone: bool = Form(True),  # label: 데드존 밴드
    show_center_line: bool = Form(True),  # label: 타깃 중심선·타입 라벨
    show_target_highlight: bool = Form(False),  # label: 선택 공/소유선수 마커
    overrides: str = Form("{}"),  # 크롭 튜닝 오버라이드 (JSON)
    detectors: str = Form("[]"),  # 검출기 엔트리 목록 (JSON)
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    ext = Path(file.filename or "v").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type: {ext}")
    if not object_tracking and not crop_tracking:
        raise HTTPException(422, "Enable object tracking, crop tracking, or both")
    if crop_output not in ("none", "label", "video"):
        raise HTTPException(422, "crop_output must be 'none', 'label' or 'video'")
    try:
        overrides_dict = json.loads(overrides) if overrides else {}
        if not isinstance(overrides_dict, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(422, "overrides must be a JSON object") from None
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, model_pt(session, mid, project_id)) for mid in ids]

    await run_in_threadpool(sweep_old_annotations)  # keep test_dir/annotate tidy
    job_id = uuid.uuid4().hex
    work = settings.test_dir / "annotate" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"source{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    cfg = {
        "source": str(src),
        "out": str(work / "out.mp4"),
        "specs": specs,
        "conf": conf,
        "iou_wbf": iou_wbf,
        "imgsz": imgsz,
        "device": device,
        "object_tracking": object_tracking,
        "crop_tracking": crop_tracking,
        "crop_output": crop_output,
        "draw_crop_box": draw_crop_box,
        "show_dead_zone": show_dead_zone,
        "show_center_line": show_center_line,
        "show_target_highlight": show_target_highlight,
        "overrides": overrides_dict,
        "detectors": detectors_cfg(session, detectors, project_id),
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/annotate/{job_id}/events")
async def annotate_events(job_id: str):
    return await job_event_stream(job_id)


@router.get("/annotate/{job_id}/result")
def annotate_result(job_id: str):
    if not job_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid job id")
    out = settings.test_dir / "annotate" / job_id / "out.mp4"
    if not out.exists():
        raise HTTPException(404, "Annotated video not ready")
    return FileResponse(out, media_type="video/mp4")  # FileResponse handles Range


@router.get("/annotate/{job_id}/crop")
def annotate_crop(job_id: str):
    """adaptive-crop's computed crop-X coordinates (keyframes + 100ms samples) as JSON.
    Present only when the job ran with crop tracking enabled."""
    if not job_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid job id")
    crop = settings.test_dir / "annotate" / job_id / "crop.json"
    if not crop.exists():
        raise HTTPException(404, "Crop coordinates not available")
    return FileResponse(crop, media_type="application/json", filename="crop.json")


# ---------- crop-cut: 추론 없이 세로 크롭 클립만 만든다 ----------


@router.post("/crop-cut", response_model=TestJobStart, status_code=201)
async def start_crop_cut(
    mode: str = Form(...),  # "json" = follow uploaded crop.json | "center" = fixed centre
    file: UploadFile = File(...),
    crop_json: UploadFile | None = File(None),
):
    """Cut a vertical 9:16 crop clip with NO model inference.

    - mode="json":   follow the coordinates in an uploaded crop.json.
    - mode="center": crop fixed to the frame centre (no JSON).
    Reuses the annotate job dir + events/result endpoints.
    """
    ext = Path(file.filename or "v").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type: {ext}")
    if mode not in ("json", "center"):
        raise HTTPException(422, "mode must be 'json' or 'center'")

    crop_bytes: bytes | None = None
    if mode == "json":
        if crop_json is None:
            raise HTTPException(422, "crop_json file is required for 'json' mode")
        crop_bytes = await crop_json.read()
        try:
            parsed = json.loads(crop_bytes)
            has_spec = isinstance(parsed.get("keyframes"), list) and parsed["keyframes"]
            has_legacy = isinstance(parsed.get("samples"), list) and parsed["samples"]
            if not (has_spec or has_legacy):
                raise ValueError
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(
                422,
                "crop_json must contain a non-empty 'keyframes' (or legacy 'samples') array",
            ) from None

    await run_in_threadpool(sweep_old_annotations)
    job_id = uuid.uuid4().hex
    work = settings.test_dir / "annotate" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"source{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    crop_json_path = None
    if crop_bytes is not None:
        crop_json_path = work / "crop_input.json"
        crop_json_path.write_bytes(crop_bytes)

    cfg = {
        "source": str(src),
        "out": str(work / "out.mp4"),
        "crop_source": mode,
        "crop_json_path": str(crop_json_path) if crop_json_path else None,
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)
