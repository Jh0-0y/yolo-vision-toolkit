"""타일링 잡 — 진행률 기록과 재시작 정정."""

import time

import pytest
from PIL import Image

from app.core.config import settings
from app.services import datasets
from app.services.tile_manager import reconcile_on_boot, tile_manager
from infra import jobs
from lib.labels.dataset_tile import TileDatasetParams


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """`settings.data_dir` 을 임시 디렉터리로 돌린다 — `jobs_dir`·`projects_dir` 이
    전부 여기서 파생되므로 **실제 `data/` 를 건드리지 않는다.**
    (`app/tests/test_datasets.py` 와 같은 픽스처다 — 공용 conftest 는 없다.)"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def _wait_done(job_id: str, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = jobs.at(settings.jobs_dir, job_id).status()
        if status != "running":
            return status
        time.sleep(0.1)
    pytest.fail("tiling job did not finish in time")


def test_submit_writes_tiles_and_a_done_event(data_dir, tmp_path):
    # 라벨 없음 — 여기서 보는 건 잡 배선과 완료 이벤트지 박스 판정이 아니다.
    # (전체 프레임 박스는 모든 타일에서 min_visibility 미만이라 finding 3 이후
    # "판정 불가"로 빠져 raw/ 가 통째로 비어버린다.)
    src = tmp_path / "src"
    (src / "raw").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    Image.new("RGB", (1920, 1080)).save(src / "raw" / "a.jpg")
    out = tmp_path / "out"

    recorded: list[dict | None] = []
    tile_manager.submit(
        "ds_tiletest01", src, out, {"a"},
        TileDatasetParams(keep_all_negatives=True),
        on_done=recorded.append,
    )
    assert _wait_done("ds_tiletest01") == "done"
    assert len(list((out / "raw").iterdir())) == 8
    assert recorded and recorded[0]["status"] == "done"


def test_reconcile_marks_interrupted_runs(data_dir):
    """재시작하면 스레드 풀이 죽는다 — running 인 채 영원히 남지 않게 한다."""
    project_id = "p_tiletest"
    (settings.projects_dir / project_id).mkdir(parents=True, exist_ok=True)
    dataset_id = datasets.create(project_id, "tiled test")
    datasets.add_source(project_id, dataset_id, {
        "id": dataset_id, "kind": "tiling", "filename": "src",
        "stem": None, "at": "2026-08-20T00:00:00+00:00", "status": "running",
    })

    reconcile_on_boot()

    sources = datasets.read_sources(project_id, dataset_id)
    assert sources[0]["status"] == "error"
    status, msg = jobs.at(settings.jobs_dir, dataset_id).status()
    assert status == "error"
    assert "restart" in (msg or "")
