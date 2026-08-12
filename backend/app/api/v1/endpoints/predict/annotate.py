"""영상 오버레이 렌더 — 클립 전체에 박스·크롭 창을 구워 mp4 로 낸다.

추론 없이 자르기만 하는 crop-cut 도 여기 둔다. 같은 잡 디렉터리를 쓰고 워커도
같기 때문이다 — 시작하는 방법만 다르다.

여기는 **시작만** 한다. 산출물은 `projects/{project_id}/crops/{crop_id}/` 에 워커가
직접 쓰고, 그 뒤의 조회·다운로드·삭제는 전부 `endpoints/crops.py` 가 맡는다.
`crop_id` 가 곧 잡 id 라 진행률 파일도 같은 이름으로 찾을 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import detectors_cfg, model_pt
from app.db import get_session
from app.models import Project
from app.schemas.predict import TestJobStart
from app.services import crop_runs
from app.services.test_jobs import test_job_manager
from lib.formats import VIDEO_EXTS

router = APIRouter(prefix="/predict", tags=["predict"])


def _require_project(session: Session, project_id: str) -> None:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")


async def _stage_source(work: Path, file: UploadFile, ext: str) -> Path:
    """업로드 영상을 런 디렉터리에 받아 둔다. 잡이 끝나면 sweep 이 지운다."""
    src = work / f"{crop_runs.SOURCE_STEM}{ext}"
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return src


@router.post("/annotate", response_model=TestJobStart, status_code=201)
async def start_annotate(
    model_ids: str = Form(...),
    conf: float | None = Form(None),  # None → 파이프라인별 기본값 (crop 0.10 / object 0.4)
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str = Form(...),  # 산출물이 프로젝트 아래 남으므로 필수다
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
    _require_project(session, project_id)
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
    detectors_list = detectors_cfg(session, detectors, project_id)

    await run_in_threadpool(crop_runs.sweep_expired)
    # 메타를 먼저 남긴다 — 잡이 실패해도 "이 설정으로 돌렸다"가 목록에 남는다.
    job_id = await run_in_threadpool(
        crop_runs.create,
        project_id,
        "json",
        file.filename or "video",
        {
            "model_ids": ids,
            "conf": conf,
            "iou_wbf": iou_wbf,
            "imgsz": imgsz,
            "crop_output": crop_output,
            "object_tracking": object_tracking,
            "crop_tracking": crop_tracking,
            "overrides": overrides_dict,
            "detectors": json.loads(detectors) if detectors else [],
        },
    )
    work = crop_runs.run_dir(project_id, job_id)
    src = await _stage_source(work, file, ext)

    cfg = {
        "source": str(src),
        "out": str(work / crop_runs.VIDEO_NAME),
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
        "detectors": detectors_list,
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)


# ---------- crop-cut: 추론 없이 세로 크롭 클립만 만든다 ----------


@router.post("/crop-cut", response_model=TestJobStart, status_code=201)
async def start_crop_cut(
    project_id: str = Form(...),  # 산출물이 프로젝트 아래 남으므로 필수다
    mode: str = Form(...),  # "json" = follow uploaded crop.json | "center" = fixed centre
    file: UploadFile = File(...),
    crop_json: UploadFile | None = File(None),
    session: Session = Depends(get_session),
):
    """Cut a vertical 9:16 crop clip with NO model inference.

    - mode="json":   follow the coordinates in an uploaded crop.json.
    - mode="center": crop fixed to the frame centre (no JSON).
    Reuses the annotate worker + the crop-run layout.
    """
    _require_project(session, project_id)
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

    await run_in_threadpool(crop_runs.sweep_expired)
    job_id = await run_in_threadpool(
        crop_runs.create, project_id, "cut", file.filename or "video", {"mode": mode}
    )
    work = crop_runs.run_dir(project_id, job_id)
    src = await _stage_source(work, file, ext)

    crop_json_path = None
    if crop_bytes is not None:
        crop_json_path = work / "crop_input.json"
        crop_json_path.write_bytes(crop_bytes)

    cfg = {
        "source": str(src),
        "out": str(work / crop_runs.VIDEO_NAME),
        "crop_source": mode,
        "crop_json_path": str(crop_json_path) if crop_json_path else None,
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)
