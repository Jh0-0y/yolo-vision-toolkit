"""Test playground: run trained models on an image (or dataset frame) and get
detections back — no DB writes. Also manages warm model residency for fast
repeated tests / A-B comparison.

The endpoint is thin: resolve model paths + image path (business), then hand the
raw compute to InferManager (which owns the warm worker). torch never loads here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session
from app.domain.video import VIDEO_EXTS
from app.ml.labeling import IMAGE_EXTS
from app.models import ModelEntry
from app.schemas.predict import PredictResponse, ResidentModel, TestJobStart
from app.services.infer_manager import infer_manager
from app.services.label_manager import read_progress
from app.services.test_jobs import (
    sweep_old_annotations,
    sweep_old_compare,
    sweep_old_live,
    test_job_manager,
)

router = APIRouter(prefix="/predict", tags=["predict"])

_TERMINAL = {"done", "error", "cancelled"}


async def _job_event_stream(job_id: str):
    """SSE tail of jobs_dir/{job_id}/progress.jsonl until a terminal phase."""

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(read_progress, job_id, offset)
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in _TERMINAL:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())


def _model_pt(session: Session, model_id: str, project_id: str | None) -> str:
    """Resolve a model id to its .pt path, enforcing project scope."""
    entry = session.get(ModelEntry, model_id)
    if entry is None:
        raise HTTPException(422, f"Model not found: {model_id}")
    if entry.project_id is not None and project_id and entry.project_id != project_id:
        raise HTTPException(422, f"Model does not belong to this project: {model_id}")
    pt = settings.model_dir(entry.project_id, model_id) / "model.pt"
    if not pt.exists():
        raise HTTPException(422, f"Model file missing: {model_id}")
    return str(pt)


@router.post("", response_model=PredictResponse)
async def predict(
    model_ids: str = Form(...),  # comma-separated DB model ids
    conf: float = Form(0.4),
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    image_project_id: str | None = Form(None),
    image_name: str | None = Form(None),
    file: UploadFile | None = File(None),
    session: Session = Depends(get_session),
):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]

    tmp_path: Path | None = None
    if file is not None:
        ext = Path(file.filename or "img").suffix.lower() or ".jpg"
        if ext not in IMAGE_EXTS:
            raise HTTPException(422, f"Unsupported image type: {ext}")
        uploads = settings.test_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        tmp_path = uploads / f"{uuid.uuid4().hex}{ext}"
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        image_path = str(tmp_path)
    elif image_project_id and image_name:
        p = settings.projects_dir / image_project_id / "raw" / Path(image_name).name
        if not p.exists():
            raise HTTPException(404, "Image not found")
        image_path = str(p)
    else:
        raise HTTPException(422, "Provide an image file or a dataset image reference")

    cfg = {"conf": conf, "iou_wbf": iou_wbf, "imgsz": imgsz, "device": device}
    try:
        result = await run_in_threadpool(infer_manager.predict, specs, image_path, cfg)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return result


@router.get("/residents", response_model=list[ResidentModel])
async def list_residents():
    return await run_in_threadpool(infer_manager.residents)


@router.post("/residents/{model_id}", response_model=ResidentModel, status_code=201)
async def load_resident(
    model_id: str,
    project_id: str | None = None,
    device: str | None = None,
    session: Session = Depends(get_session),
):
    pt = _model_pt(session, model_id, project_id)
    return await run_in_threadpool(infer_manager.load, model_id, pt, device)


@router.delete("/residents/{model_id}", status_code=204)
async def unload_resident(model_id: str):
    await run_in_threadpool(infer_manager.unload, model_id)


# ---------- video annotation (batch: draw boxes onto the whole clip) ----------


@router.post("/annotate", response_model=TestJobStart, status_code=201)
async def start_annotate(
    model_ids: str = Form(...),
    conf: float | None = Form(None),  # None → 파이프라인별 기본값 (crop 0.10 / object 0.4)
    iou_wbf: float = Form(0.55),
    imgsz: int = Form(640),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    object_tracking: bool = Form(True),  # ByteTrack boxes + IDs + trails
    crop_tracking: bool = Form(True),  # trackcrop vertical 9:16 crop window
    crop_output: str = Form("label"),  # "none" = JSON only | "label" = overlay | "video" = cut clip
    draw_crop_box: bool = Form(True),  # label: 9:16 사각형(+데드존·센터선)
    show_dead_zone: bool = Form(True),  # label: 데드존 밴드
    show_center_line: bool = Form(True),  # label: 타깃 중심선·타입 라벨
    show_target_highlight: bool = Form(False),  # label: 선택 공/소유선수 마커
    overrides: str = Form("{}"),  # trackcrop 튜닝 오버라이드 (JSON)
    extra_detectors: str = Form("[]"),  # 공 전담 추가 검출기 목록 (JSON)
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
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]

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
        "extras": _extras_cfg(session, extra_detectors, project_id),
    }
    await run_in_threadpool(test_job_manager.submit_annotate, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/annotate/{job_id}/events")
async def annotate_events(job_id: str):
    return await _job_event_stream(job_id)


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
    """trackcrop's computed crop-X coordinates (keyframes + 100ms samples) as JSON.
    Present only when the job ran with crop tracking enabled."""
    if not job_id.isalnum():  # uuid4().hex — blocks path traversal
        raise HTTPException(422, "Invalid job id")
    crop = settings.test_dir / "annotate" / job_id / "crop.json"
    if not crop.exists():
        raise HTTPException(404, "Crop coordinates not available")
    return FileResponse(crop, media_type="application/json", filename="crop.json")


# ---------- crop-cut: make a vertical crop clip without inference ----------


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
            if not isinstance(parsed.get("samples"), list) or not parsed["samples"]:
                raise ValueError
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(
                422, "crop_json must be a crop.json with a non-empty 'samples' array"
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


# ---------- live preview: detect once, recompute crop coords on every tuning change ----------


def _live_dir(detect_id: str) -> Path:
    """Resolve a live-preview cache dir, blocking path traversal via the id shape."""
    if not detect_id.isalnum():  # uuid4().hex
        raise HTTPException(422, "Invalid detect id")
    return settings.test_dir / "live" / detect_id


def _extras_cfg(
    session: Session, extra_detectors: str, project_id: str | None
) -> list[dict]:
    """추가 검출기(공 전담) 목록 파싱 — [{model_id, mode, conf?, imgsz?, tile_size?,
    stride?, merge_iou?}]. 비어있으면 단일 모델 모드([])."""
    try:
        entries = json.loads(extra_detectors) if extra_detectors else []
        if not isinstance(entries, list):
            raise ValueError
    except ValueError:
        raise HTTPException(422, "extra_detectors must be a JSON array") from None

    extras: list[dict] = []
    for e in entries:
        if not isinstance(e, dict) or not e.get("model_id"):
            raise HTTPException(422, "each extra detector needs a model_id")
        mode = e.get("mode", "tiled")
        if mode not in ("full", "tiled"):
            raise HTTPException(422, "extra detector mode must be 'full' or 'tiled'")
        tile_size = int(e.get("tile_size", 640))
        stride = int(e.get("stride", 480))
        if mode == "tiled" and (stride <= 0 or stride > tile_size):
            raise HTTPException(422, "stride must be in (0, tile_size]")
        extras.append(
            {
                "pt": _model_pt(session, e["model_id"], project_id),
                "mode": mode,
                "conf": e.get("conf"),
                "imgsz": int(e.get("imgsz", 1920)),
                "tile_size": tile_size,
                "stride": stride,
                "merge_iou": float(e.get("merge_iou", 0.5)),
            }
        )
    return extras


@router.post("/live", response_model=TestJobStart, status_code=201)
async def start_live(
    model_ids: str = Form(...),
    conf: float | None = Form(None),  # None → crop 검출 기본값 0.10
    imgsz: int = Form(1920),
    device: str | None = Form(None),
    project_id: str | None = Form(None),
    sampling_interval_ms: int | None = Form(None),  # None → 기본값 (100ms)
    extra_detectors: str = Form("[]"),  # 공 전담 추가 검출기 목록 (JSON)
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
    specs = [(mid, _model_pt(session, mid, project_id)) for mid in ids]

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
        "extras": _extras_cfg(session, extra_detectors, project_id),
    }
    await run_in_threadpool(test_job_manager.submit_live, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/live/{job_id}/events")
async def live_events(job_id: str):
    return await _job_event_stream(job_id)


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


@router.post("/live/{detect_id}/plan")
async def live_plan(detect_id: str, body: dict = Body(default={})):
    """Recompute the crop trajectory from cached detections + tuning overrides.

    This is the cheap half of the pipeline (no model/torch) — it runs synchronously
    so the client can call it on every knob change and redraw the crop box at once.
    """
    work = _live_dir(detect_id)
    det_path = work / "detected.json"
    if not det_path.exists():
        raise HTTPException(404, "Detection cache not found")
    overrides = body.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(422, "overrides must be an object")
    collect_debug = bool(body.get("collect_debug", False))

    def _compute() -> dict:
        # trackcrop imports cv2/numpy but not torch here (Detector loads YOLO lazily).
        from app.ml.trackcrop.detection_io import load_detections
        from app.ml.trackcrop.pipeline import plan_from_detections

        data = json.loads(det_path.read_text(encoding="utf-8"))
        detected = load_detections(data)
        # validate=False: non-1080p input trips the 1920/608 geometry checks (same as
        # the batch annotate path); the client scales the overlay to the real frame.
        result = plan_from_detections(
            detected, overrides=overrides, collect_debug=collect_debug, validate=False,
        )
        return result.to_dict()

    return await run_in_threadpool(_compute)


# ---------- model comparison (score models vs an uploaded YOLO test set) ----------


def _extract_compare_dataset(zip_path: Path, dest: Path) -> Path:
    """Unzip a YOLO test set and return the directory holding its data.yaml
    (root or one level down). Raises 422 on a non-dataset zip."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        raise HTTPException(422, "Corrupted zip file")

    for candidate in ("data.yaml", "data.yml"):
        if (dest / candidate).exists():
            return dest
    found = sorted(list(dest.glob("*/data.yaml")) + list(dest.glob("*/data.yml")))
    if found:
        return found[0].parent
    raise HTTPException(422, "data.yaml not found in the zip (a YOLO-format dataset is required)")


@router.post("/compare", response_model=TestJobStart, status_code=201)
async def start_compare(
    project_id: str = Form(...),
    model_ids: str = Form(...),
    conf: float = Form(0.4),
    iou: float = Form(0.5),  # IoU match threshold (pred↔GT)
    imgsz: int = Form(640),
    device: str | None = Form(None),
    file: UploadFile = File(...),  # YOLO test set zip: images/ + labels/ + data.yaml
    session: Session = Depends(get_session),
):
    ext = Path(file.filename or "d").suffix.lower()
    if ext != ".zip":
        raise HTTPException(422, "Upload a .zip YOLO dataset (images/ + labels/ + data.yaml)")
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model")
    specs: list[tuple[str, str]] = []
    model_names: dict[str, str] = {}
    for mid in ids:
        specs.append((mid, _model_pt(session, mid, project_id)))
        entry = session.get(ModelEntry, mid)
        model_names[mid] = entry.name if entry else mid

    await run_in_threadpool(sweep_old_compare)  # keep test_dir/compare tidy
    job_id = uuid.uuid4().hex
    work = settings.test_dir / "compare" / job_id
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / "upload.zip"
    with open(zip_path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    dataset_dir = await run_in_threadpool(_extract_compare_dataset, zip_path, work / "dataset")

    cfg = {
        "project_id": project_id,
        "specs": specs,
        "model_names": model_names,
        "dataset_dir": str(dataset_dir),
        "conf": conf,
        "iou": iou,
        "iou_wbf": 0.55,
        "imgsz": imgsz,
        "device": device,
    }
    await run_in_threadpool(test_job_manager.submit_compare, job_id, cfg)
    return TestJobStart(job_id=job_id)


@router.get("/compare/{job_id}/events")
async def compare_events(job_id: str):
    return await _job_event_stream(job_id)


@router.get("/compare/{job_id}/result")
def compare_result(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(422, "Invalid job id")
    path = settings.jobs_dir / job_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "Comparison result not ready")
    return json.loads(path.read_text())


@router.get("/compare/{job_id}/images/{idx}")
def compare_image(job_id: str, idx: str):
    """Serve an uploaded test-set image by index for the overlay view. Images are
    referenced via the worker's images_manifest.json (index → absolute path);
    the resolved path is confined to this job's upload dir to block traversal."""
    if not job_id.isalnum() or not idx.isdigit():
        raise HTTPException(422, "Invalid id")
    manifest_path = settings.jobs_dir / job_id / "images_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Comparison images not available")
    target = json.loads(manifest_path.read_text()).get(idx)
    if not target:
        raise HTTPException(404, "Image not found")
    base = (settings.test_dir / "compare" / job_id).resolve()
    path = Path(target).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)
