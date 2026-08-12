"""모델 비교 — 업로드한 YOLO 테스트셋을 정답 삼아 모델 여러 개를 채점한다.

P/R/F1 과 박스 오버레이를 함께 돌려주므로 "어느 모델이 무엇을 놓쳤는지"를 눈으로
확인할 수 있다. 채점 계산은 워커(`lib/detect/evaluate`)가 하고 여기서는 업로드를
받아 잡을 띄운다.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import job_event_stream, model_pt
from app.core.config import settings
from app.db import get_session
from app.models import ModelEntry
from app.schemas.predict import TestJobStart
from app.services.test_jobs import sweep_old_compare, test_job_manager
from lib.train import dataset as train_dataset

router = APIRouter(prefix="/predict", tags=["predict"])


def _extract_compare_dataset(zip_path: Path, dest: Path) -> Path:
    """테스트셋 zip 을 풀고 data.yaml 이 있는 디렉터리를 돌려준다.

    데이터셋이 아닌 zip 이면 422. 탐색 규칙(루트, 없으면 한 단계 아래)은 학습
    데이터셋 임포트와 같은 것을 쓴다.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile:
        raise HTTPException(422, "Corrupted zip file")

    yaml_path = train_dataset.find_data_yaml(dest)
    if yaml_path is None:
        raise HTTPException(
            422, "data.yaml not found in the zip (a YOLO-format dataset is required)"
        )
    return yaml_path.parent


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
        specs.append((mid, model_pt(session, mid, project_id)))
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
    return await job_event_stream(job_id)


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
