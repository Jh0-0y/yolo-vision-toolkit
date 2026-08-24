"""벤치마크 런의 파일 규약 — 디렉터리가 곧 목록이고 상태는 진행률에서 파생한다."""

import pytest

from app.core.config import settings
from app.services import benchmarks
from infra import jobs


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """`settings.data_dir` 을 임시 디렉터리로 돌린다 — 실제 `data/` 를 건드리지 않는다.
    (`app/tests/test_datasets.py` 와 같은 픽스처다 — 공용 conftest 는 없다.)"""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


def test_create_lays_out_the_run_and_records_settings(data_dir):
    bid = benchmarks.create("p1", {"dataset_name": "공만", "entries": 2, "conf": 0.4})

    assert benchmarks.run_dir("p1", bid).is_dir()
    meta = benchmarks.read_meta("p1", bid)
    assert meta["dataset_name"] == "공만"
    assert meta["entries"] == 2
    assert meta["id"] == bid
    assert meta["created_at"]


def test_status_comes_from_the_progress_file(data_dir):
    """상태를 저장하지 않는다 — 진행률 파일의 마지막 종료 이벤트가 유일한 근거다."""
    bid = benchmarks.create("p1", {})
    assert benchmarks.status_of(bid)[0] == "running"

    jobs.at(settings.jobs_dir, bid).ensure().emit({"phase": "done"})
    assert benchmarks.status_of(bid)[0] == "done"


def test_list_is_newest_first_and_scoped_to_the_project(data_dir):
    # `create` 는 `**run_settings` 를 나중에 펼치므로 created_at 을 넘기면 그것이 이긴다
    benchmarks.create("p1", {"dataset_name": "old", "created_at": "2020-01-01T00:00:00+00:00"})
    benchmarks.create("p1", {"dataset_name": "new", "created_at": "2026-01-01T00:00:00+00:00"})
    benchmarks.create("p2", {"dataset_name": "other"})

    rows = benchmarks.list_runs("p1")
    assert [r["dataset_name"] for r in rows] == ["new", "old"]
    assert benchmarks.list_runs("nope") == []


def test_delete_takes_the_directory_and_the_job_dir(data_dir):
    bid = benchmarks.create("p1", {})
    jobs.at(settings.jobs_dir, bid).ensure().emit({"phase": "done"})

    benchmarks.delete("p1", bid)

    assert not benchmarks.run_dir("p1", bid).exists()
    assert not (settings.jobs_dir / bid).exists()
    assert benchmarks.list_runs("p1") == []


def test_reconcile_marks_interrupted_runs(data_dir):
    """워커 풀은 API 프로세스가 소유하므로 재시작하면 돌던 런이 죽는다 —
    영원히 running 으로 보이는 항목이 없게 한다."""
    bid = benchmarks.create("p1", {})

    benchmarks.reconcile_on_boot()

    status, msg = benchmarks.status_of(bid)
    assert status == "error"
    assert "restart" in (msg or "")


def test_reconcile_leaves_finished_runs_alone(data_dir):
    bid = benchmarks.create("p1", {})
    jobs.at(settings.jobs_dir, bid).ensure().emit({"phase": "done"})

    benchmarks.reconcile_on_boot()

    assert benchmarks.status_of(bid)[0] == "done"


def test_delete_keeps_the_job_dir_while_the_worker_is_still_running(data_dir, monkeypatch):
    """도는 잡을 지울 때는 잡 디렉터리를 남긴다 — 취소 신호가 그 안의 파일 하나라,
    함께 지우면 워커가 다음 확인 지점에 닿기 전에 신호가 사라진다."""
    from app.api.v1.endpoints.predict import compare
    from app.services.test_jobs import test_job_manager

    bid = benchmarks.create("p1", {})
    jobs.at(settings.jobs_dir, bid).prepare()
    # 진짜 워커를 띄우지 않고 "아직 돈다"만 흉내 낸다
    monkeypatch.setattr(test_job_manager, "is_active", lambda job_id: True)

    compare.delete_benchmark(bid, "p1")

    assert not benchmarks.run_dir("p1", bid).exists()
    job = jobs.at(settings.jobs_dir, bid)
    assert job.path.is_dir()
    assert job.cancelled()  # 워커가 다음 확인 지점에서 볼 수 있게 남아 있다
    assert benchmarks.list_runs("p1") == []


def test_delete_takes_the_job_dir_too_when_nothing_is_running(data_dir, monkeypatch):
    from app.api.v1.endpoints.predict import compare
    from app.services.test_jobs import test_job_manager

    bid = benchmarks.create("p1", {})
    jobs.at(settings.jobs_dir, bid).ensure().emit({"phase": "done"})
    monkeypatch.setattr(test_job_manager, "is_active", lambda job_id: False)

    compare.delete_benchmark(bid, "p1")

    assert not benchmarks.run_dir("p1", bid).exists()
    assert not (settings.jobs_dir / bid).exists()
