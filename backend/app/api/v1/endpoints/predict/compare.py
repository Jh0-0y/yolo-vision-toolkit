"""벤치마크 — 데이터셋의 test 분할을 정답 삼아 모델 여러 개를 채점한다.

P/R/F1 과 박스 오버레이를 함께 돌려주므로 "어느 모델이 무엇을 놓쳤는지"를 눈으로
확인할 수 있다. 채점 계산은 워커(`lib/detect/evaluate`)가 하고 여기서는 입력을
검증하고 잡을 띄운다. 런은 이력으로 남아 목록·삭제로 다룰 수 있다 — 자리와
수명 규약은 `services/benchmarks.py` 를 본다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlmodel import Session

from app.api.v1.endpoints.predict.common import job_event_stream, model_pt
from app.db import get_session
from app.models import ModelEntry
from app.schemas.predict import BenchmarkOut, BenchmarkStart, TestJobStart
from app.services import benchmarks, datasets
from app.services.test_jobs import test_job_manager
from lib.detect.tiled import TiledParams
from lib.labels import dataset_export

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


def _parse_dataset_token(token: str) -> tuple[str, str]:
    """`dataset:{project_id}:{dataset_id}` → `(project_id, dataset_id)`."""
    parts = token.split(":")
    if parts[0] != "dataset" or len(parts) != 3:
        raise HTTPException(422, f"Invalid dataset token: {token}")
    pid, dsid = parts[1], parts[2]
    if any(c in pid + dsid for c in "/\\.."):
        raise HTTPException(422, "Invalid dataset token")
    if datasets.read_meta(pid, dsid) is None:
        raise HTTPException(404, "Dataset not found")
    return pid, dsid


def _require_project_id(project_id: str) -> str:
    """`project_id` 는 그대로 경로 조각이 된다 — 토큰과 **같은 기준**으로 막는다.

    POST 는 토큰 안에서 걸러지지만 읽기·삭제 라우트는 쿼리스트링에서 바로 받으므로
    여기서 한 번 더 본다. 막지 않으면 `?project_id=../../lab` 같은 값이
    `projects_dir` 밖을 가리킨다.
    """
    if not project_id or any(c in project_id for c in "/\\.."):
        raise HTTPException(422, "Invalid project id")
    return project_id


@router.post("/benchmarks", response_model=TestJobStart, status_code=201)
async def start_benchmark(body: BenchmarkStart, session: Session = Depends(get_session)):
    """데이터셋의 **test 분할**을 정답 삼아 엔트리들을 채점한다.

    입력이 데이터셋이라 "이 점수가 어떤 데이터에서 나왔나"에 언제든 답할 수 있고,
    엔트리마다 배포 방식대로 돌리므로 풀 프레임 모델과 타일 모델을 같은 GT 위에서
    비교할 수 있다.
    """
    project_id, dataset_id = _parse_dataset_token(body.dataset)
    if not body.entries:
        raise HTTPException(422, "Select at least one model")
    for e in body.entries:
        if e.mode not in ("full", "tiled"):
            raise HTTPException(422, "detector mode must be 'full' or 'tiled'")
        if e.mode == "tiled":
            errors = TiledParams(
                tile_size=e.tile_size, stride=e.stride,
                merge_iou=e.merge_iou, border_margin_px=e.border_margin_px,
            ).validate()
            if errors:
                raise HTTPException(422, f"Invalid tiling params: {'; '.join(errors)}")

    # 모델 조회는 디렉터리를 만들기 **전에** 끝낸다 — model_pt() 가 여기서 422 를
    # 던지면(모델 없음·파일 없음·다른 프로젝트 소유) 남길 것이 아직 아무것도 없다.
    entries = []
    for i, e in enumerate(body.entries):
        entry = session.get(ModelEntry, e.model_id)
        entries.append({
            "entry_id": f"e{i}",
            "model_id": e.model_id,
            "name": entry.name if entry else e.model_id,
            "pt": model_pt(session, e.model_id, project_id),
            "mode": e.mode, "imgsz": e.imgsz, "tile_size": e.tile_size,
            "stride": e.stride, "merge_iou": e.merge_iou,
            "border_margin_px": e.border_margin_px,
        })

    ds_name = (datasets.read_meta(project_id, dataset_id) or {}).get("name", dataset_id)
    # 기록은 **시작할 때** 쓴다 — 실패한 시도도 목록에 남아야 무엇을 해봤는지가 남는다
    bench_id = benchmarks.create(
        project_id,
        {
            "dataset": body.dataset,
            "dataset_name": ds_name,
            "entries": len(body.entries),
            "conf": body.conf,
            "iou": body.iou,
            "detail": [e.model_dump() for e in body.entries],
        },
    )
    out_dir = benchmarks.run_dir(project_id, bench_id)
    dataset_dir = out_dir / benchmarks.DATASET_DIR

    # 자리를 만든 뒤로는 **어디서 실패하든** 런을 지운다. 잡 디렉터리 없이 run.json
    # 만 남으면 `JobDir.status()` 가 종료 이벤트를 못 찾아 영원히 "running" 을
    # 돌려주고, 목록에서도 상세에서도 끝나지 않는 런으로 남는다.
    try:
        try:
            await run_in_threadpool(
                dataset_export.materialize,
                dataset_dir=datasets.dataset_dir(project_id, dataset_id),
                out_dir=dataset_dir,
                kind="test",
                reviewed=datasets.read_reviewed(project_id, dataset_id)
                & datasets.image_stems(project_id, dataset_id),
                splits=datasets.read_splits(project_id, dataset_id),
            )
        except dataset_export.ExportError:
            raise HTTPException(
                422, "Nothing in the test split — split the dataset first"
            ) from None

        cfg = {
            "project_id": project_id,
            "entries": entries,
            "dataset_dir": str(dataset_dir),
            "out_dir": str(out_dir),
            "conf": body.conf,
            "iou": body.iou,
            "iou_wbf": 0.55,
            "device": body.device,
        }
        await run_in_threadpool(test_job_manager.submit_compare, bench_id, cfg)
    except HTTPException:
        benchmarks.delete(project_id, bench_id)
        raise
    except BaseException as exc:  # 디스크 가득 참 · 권한 · 풀 제출 실패, 그리고 요청 취소
        # `Exception` 이 아니라 `BaseException` 이다 — 클라이언트가 POST 를 끊으면
        # `CancelledError` 가 올라오는데, 그것도 자리를 남긴 채 빠져나가면 안 된다.
        benchmarks.delete(project_id, bench_id)
        if isinstance(exc, Exception):
            # 예외 문구에는 서버의 절대 경로가 섞여 나온다 — 로그에만 남기고 밖으로는 내지 않는다.
            logger.exception("Failed to start benchmark %s", bench_id)
            raise HTTPException(500, "Failed to start the benchmark") from exc
        raise
    return TestJobStart(job_id=bench_id)


@router.get("/benchmarks", response_model=list[BenchmarkOut])
def list_benchmarks(project_id: str):
    _require_project_id(project_id)
    return [BenchmarkOut(**row) for row in benchmarks.list_runs(project_id)]


@router.delete("/benchmarks/{bench_id}", status_code=204)
def delete_benchmark(bench_id: str, project_id: str):
    if not benchmarks.valid_id(bench_id):
        raise HTTPException(422, "Invalid benchmark id")
    _require_project_id(project_id)
    # 아직 돌고 있으면 멈추라고 알리고 **잡 디렉터리는 남긴다.** 취소 신호가 그
    # 안의 `CANCEL` 파일 하나뿐이라, 런과 함께 지우면 워커가 다음 확인 지점에
    # 닿기 전에 신호가 사라져 500장을 끝까지 돌린다. 워커는 이미지 한 장마다,
    # 그리고 **결과를 쓰기 전에** 확인하므로 신호를 보면 `benchmarks/{id}/` 를
    # 되살리지 않고 멈춘다.
    #
    # 대가: 이렇게 남은 잡 디렉터리는 아무도 회수하지 않는다 — `reconcile_on_boot`
    # 은 `*/run.json` 을 글롭하는데 그것이 이미 없다. progress.jsonl 몇 KB 를
    # 남기는 쪽이, 하드링크 데이터셋 트리를 통째로 쥔 채 목록에 뜨지도 않아
    # 지울 수도 없는 디렉터리를 남기는 쪽보다 낫다.
    active = test_job_manager.is_active(bench_id)
    if active:
        test_job_manager.cancel(bench_id)
    benchmarks.delete(project_id, bench_id, keep_job_dir=active)


@router.get("/benchmarks/{bench_id}/events")
async def benchmark_events(bench_id: str):
    if not benchmarks.valid_id(bench_id):
        raise HTTPException(422, "Invalid benchmark id")
    return await job_event_stream(bench_id)


@router.get("/benchmarks/{bench_id}/result")
def benchmark_result(bench_id: str, project_id: str):
    if not benchmarks.valid_id(bench_id):
        raise HTTPException(422, "Invalid benchmark id")
    _require_project_id(project_id)
    path = benchmarks.artifact(project_id, bench_id, benchmarks.RESULT_NAME)
    if not path.exists():
        raise HTTPException(404, "Benchmark result not ready")
    return json.loads(path.read_text())


@router.get("/benchmarks/{bench_id}/images/{idx}")
def benchmark_image(bench_id: str, idx: str, project_id: str):
    """오버레이용 이미지를 색인으로 낸다. 해석한 경로를 이 런의 디렉터리 안으로
    가둬 경로 탈출을 막는다."""
    if not benchmarks.valid_id(bench_id) or not idx.isdigit():
        raise HTTPException(422, "Invalid id")
    _require_project_id(project_id)
    manifest = benchmarks.artifact(project_id, bench_id, benchmarks.MANIFEST_NAME)
    if not manifest.exists():
        raise HTTPException(404, "Benchmark images not available")
    target = json.loads(manifest.read_text()).get(idx)
    if not target:
        raise HTTPException(404, "Image not found")
    base = benchmarks.run_dir(project_id, bench_id).resolve()
    path = Path(target).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)
