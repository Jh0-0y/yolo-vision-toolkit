"""데이터셋으로 가져오기 — 동영상 · 기존 YOLO 데이터셋.

두 방식 모두 **미검수**로 들어온다. 검수는 그 다음 일이다.

동영상은 프레임을 얻는 수단일 뿐이라 **추출이 끝나면 지운다.** 그래서 업로드도
데이터셋 안 감춰진 자리(`.import/`)로 받는다 — 목록에 뜨는 자리가 아니다.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.db import get_session
from app.models import Project
from app.services import datasets
from app.services.video_manager import video_manager
from infra import jobs
from lib.formats import VIDEO_EXTS
from lib.fsutil import safe_stem, unique_stem
from lib.labels.import_yolo import YoloImportError, import_zip
from lib.media.extract import ExtractParams
from lib.media.tiling import TilingParams

router = APIRouter(
    prefix="/projects/{project_id}/datasets/{dataset_id}/import", tags=["datasets"]
)

TERMINAL_PHASES = {"done", "error", "cancelled"}


def _require_dataset(session: Session, project_id: str, dataset_id: str) -> Path:
    if session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    if not datasets.valid_id(dataset_id):
        raise HTTPException(422, "Invalid dataset id")
    if datasets.read_meta(project_id, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return datasets.dataset_dir(project_id, dataset_id)


def _staging_dir(dataset: Path) -> Path:
    """업로드를 잠깐 받아 두는 자리. 이미지 목록에 섞이지 않게 감춰 둔다."""
    d = dataset / ".import"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _frame_stem(project_id: str, dataset_id: str, raw_dir: Path, name: str) -> str:
    """프레임 이름 앞부분. 규칙은 `lib.fsutil` 에 있고, 여기서 자리만 본다.

    같은 이름으로 이미 뽑은 적이 있으면 `경기 영상 (2)` 로 갈라 준다. **프레임마다가
    아니라 추출 단위로** 붙이는 이유는, 장마다 붙이면 `경기_00001 (2).jpg` 처럼 번호와
    접미사가 뒤엉켜 어느 추출에서 나왔는지 알 수 없기 때문이다.

    안 갈라 주면 더 나쁘다 — fps 만 바꿔 다시 뽑으면 앞 번호는 새 프레임으로 덮이고
    뒤 번호는 옛 프레임이 남아 **두 설정이 섞인 데이터셋**이 되고, 그건 눈에 띄지 않는다.

    쓴 이름은 **출처 기록**에서 본다. 파일만 보면 프레임을 지운 순간 "쓴 적 없는 이름"이
    되어 다음 추출이 옛 이름을 다시 잡는다. 기록이 없던 시절 데이터셋을 위해 파일도
    함께 본다.
    """
    used = datasets.used_stems(project_id, dataset_id)
    return unique_stem(
        safe_stem(name, fallback="video"),
        lambda s: s in used or (raw_dir / f"{s}_00001.jpg").exists(),
    )


def _extract_params(
    target_fps: float,
    max_frames: int,
    start_sec: float,
    end_sec: float | None,
    dedup: bool,
    dedup_threshold: float,
    tile: bool,
    tile_size: int,
    stride: int,
) -> ExtractParams:
    if target_fps <= 0:
        raise HTTPException(422, "target_fps must be greater than 0")
    if max_frames <= 0:
        raise HTTPException(422, "max_frames must be greater than 0")
    if tile:
        errors = TilingParams(tile_size=tile_size, stride=stride).validate()
        if errors:
            raise HTTPException(422, f"Invalid tiling params: {errors}")
    return ExtractParams(
        target_fps=target_fps,
        max_frames=max_frames,
        start_sec=max(0.0, start_sec),
        end_sec=end_sec,
        dedup=dedup,
        dedup_threshold=dedup_threshold,
        tile=tile,
        tile_size=tile_size,
        stride=stride,
    )


@router.post("/video", status_code=201)
async def import_video(
    project_id: str,
    dataset_id: str,
    file: UploadFile,
    target_fps: float = Form(2.0),
    max_frames: int = Form(2000),
    start_sec: float = Form(0.0),
    end_sec: float | None = Form(None),
    dedup: bool = Form(True),
    dedup_threshold: float = Form(0.92),
    tile: bool = Form(False),  # 뽑은 프레임을 타일로 쪼개 저장
    tile_size: int = Form(640),
    stride: int = Form(480),  # 겹침 = tile_size - stride
    session: Session = Depends(get_session),
):
    """영상을 올려 프레임을 이 데이터셋으로 뽑는다. **영상은 끝나면 지운다.**

    `tile` 이면 프레임을 그대로 두지 않고 타일로 쪼개 저장한다 — 데이터셋에 들어가는
    이미지가 곧 타일이다.
    """
    dataset = _require_dataset(session, project_id, dataset_id)
    params = _extract_params(
        target_fps, max_frames, start_sec, end_sec, dedup, dedup_threshold,
        tile, tile_size, stride,
    )

    name = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(
            422, f"Unsupported video format: {ext or 'none'} ({', '.join(sorted(VIDEO_EXTS))})"
        )

    job_id = f"imp_{uuid.uuid4().hex[:10]}"
    source = _staging_dir(dataset) / f"{job_id}{ext}"
    try:
        with open(source, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
    except Exception:
        source.unlink(missing_ok=True)
        raise

    datasets.ensure_dirs(project_id, dataset_id)
    raw_dir = datasets.raw_dir(project_id, dataset_id)
    stem = _frame_stem(project_id, dataset_id, raw_dir, name)

    # 기록은 **시작할 때** 남긴다. 이름이 이 시점에 잡히므로, 추출이 죽더라도
    # 그 이름은 쓴 것으로 쳐야 다음 추출이 덮어쓰지 않는다.
    datasets.add_source(
        project_id,
        dataset_id,
        {
            "id": job_id,
            "kind": "video",
            "filename": name,
            "stem": stem,
            "at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "frames": None,
            # 타일링을 켜면 **프레임 수와 들어온 이미지 수가 다르다** — 프레임 하나가
            # 타일 여러 장이 된다. 둘 다 남겨야 목록의 수치가 설명된다.
            "tiles": None,
            "params": asdict(params),
        },
    )

    def _record(result: dict | None) -> None:
        if result is None:
            datasets.update_source(project_id, dataset_id, job_id, status="error")
            return
        datasets.update_source(
            project_id,
            dataset_id,
            job_id,
            status=result.get("status", "done"),
            frames=result.get("saved", 0),
            tiles=result.get("tiles", 0),
        )

    video_manager.submit(
        job_id, source, raw_dir, stem, params, delete_source=True, on_done=_record
    )
    return {"job_id": job_id, "filename": name, "stem": stem, "status": "running"}


@router.post("/dataset", status_code=201)
async def import_yolo_dataset(
    project_id: str,
    dataset_id: str,
    file: UploadFile,
    # 이미 사람이 검증한 데이터인가. 참이면 **검수완료로 들어오고**, zip 의
    # train/val/test 폴더가 그대로 분할 배정이 된다.
    reviewed: bool = Form(True),
    session: Session = Depends(get_session),
):
    """YOLO zip(이미지+labels+data.yaml)을 이 데이터셋으로 들여온다.

    클래스는 이 데이터셋의 `classes.json` 에 병합되고 라벨의 클래스 id 는 그에 맞춰
    다시 매겨진다 — 데이터셋마다 클래스가 다르므로 저쪽 번호를 그대로 쓸 수 없다.

    `reviewed=False` 면 폴더 구조를 **읽지 않는다** — 검증되지 않은 데이터의 분할을
    믿고 쓸 이유가 없어서, 전부 미검수·미할당으로 들어온다.
    """
    dataset = _require_dataset(session, project_id, dataset_id)

    staging = _staging_dir(dataset)
    tmp = staging / f"upload_{uuid.uuid4().hex[:8]}.zip"
    try:
        with open(tmp, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        result = await run_in_threadpool(import_zip, tmp, dataset)
    except YoloImportError as e:
        raise HTTPException(422, str(e)) from None
    finally:
        tmp.unlink(missing_ok=True)

    # stem 목록은 응답에 싣지 않는다 — 수천 개가 될 수 있고 화면이 쓰지 않는다
    imported_splits: dict[str, str] = result.pop("splits", {})
    fresh = set(result.pop("stems", []))

    assigned = 0
    if reviewed and fresh:
        # **방금 들어온 것만** 올린다. 이미 있던 미검수 이미지는 건드리지 않는다.
        datasets.write_reviewed(
            project_id, dataset_id, datasets.read_reviewed(project_id, dataset_id) | fresh
        )
        if imported_splits:
            splits = datasets.read_splits(project_id, dataset_id)
            splits.update(imported_splits)
            datasets.write_splits(project_id, dataset_id, splits)
            assigned = len(imported_splits)

    # zip 도 출처다 — "무슨 데이터를 썼나"는 영상만의 질문이 아니다.
    # `stem` 은 없다: zip 은 이름 하나가 아니라 여러 이미지를 들여온다.
    datasets.add_source(
        project_id,
        dataset_id,
        {
            "id": f"zip_{uuid.uuid4().hex[:10]}",
            "kind": "dataset",
            "filename": (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
            "stem": None,
            "at": datetime.now(timezone.utc).isoformat(),
            "status": "done",
            "images": result.get("images", 0),
            "labeled": result.get("labeled", 0),
            "reviewed": reviewed,
            "assigned": assigned,
        },
    )
    return {**result, "reviewed": reviewed, "assigned": assigned}


@router.post("/{job_id}/cancel")
def cancel_import(
    project_id: str, dataset_id: str, job_id: str, session: Session = Depends(get_session)
):
    _require_dataset(session, project_id, dataset_id)
    return {"cancelled": video_manager.cancel(job_id)}


@router.get("/{job_id}/events")
async def import_events(
    project_id: str, dataset_id: str, job_id: str, session: Session = Depends(get_session)
):
    """진행률 SSE. progress.jsonl 을 처음부터 재생하므로 언제 붙어도 같은 그림이다."""
    _require_dataset(session, project_id, dataset_id)
    if not (settings.jobs_dir / job_id / "progress.jsonl").exists():
        raise HTTPException(404, "Import task not found")

    async def stream():
        offset = 0
        while True:
            events, offset = await asyncio.to_thread(
                jobs.at(settings.jobs_dir, job_id).read, offset
            )
            terminal = False
            for ev in events:
                yield {"event": "progress", "data": json.dumps(ev)}
                if ev.get("phase") in TERMINAL_PHASES:
                    terminal = True
            if terminal:
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream())
