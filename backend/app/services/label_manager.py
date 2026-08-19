"""JobManager: single-worker process pool for labeling jobs.

One GPU → one job at a time; queued jobs wait in the executor. The parent
process updates the DB when a job finishes and resets the reviewed flag for
re-labeled images (auto labels always need a fresh user review).
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.db import session_scope
from app.models import Job
from app.workers.label_worker import run_label_job
from infra import jobs
from lib.labels.store import read_reviewed, write_reviewed


class JobManager:
    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._futures: dict[str, Future] = {}

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        return self._executor

    def submit_label_job(self, job_id: str, dataset_dir: Path, cfg: dict) -> None:
        """`dataset_dir` 은 라벨이 들어가는 **데이터셋** 디렉터리다 — 검수 표시가
        그 안의 reviewed.json 에 있으므로 끝나고 되돌릴 때 필요하다."""
        jobs.at(settings.jobs_dir, job_id).prepare()

        future = self._get_executor().submit(
            run_label_job, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future
        self._mark_status(job_id, "running")
        future.add_done_callback(lambda f: self._on_done(job_id, dataset_dir, f))

    def cancel(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        if future is None:
            return False
        if future.cancel():  # still queued — never started
            self._mark_status(job_id, "cancelled")
            return True
        # running: signal the worker via sentinel file
        jobs.at(settings.jobs_dir, job_id).request_cancel()
        return True

    def _mark_status(self, job_id: str, status: str, **fields) -> None:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.status = status
            for k, v in fields.items():
                setattr(job, k, v)
            session.add(job)
            session.commit()

    def _on_done(self, job_id: str, dataset_dir: Path, future: Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            self._mark_status(
                job_id, "error", error=str(e), finished_at=datetime.now(timezone.utc)
            )
            return
        finally:
            self._futures.pop(job_id, None)

        status = result.get("status", "done")
        self._mark_status(
            job_id,
            status,
            result_json=json.dumps(result),
            finished_at=datetime.now(timezone.utc),
        )
        if status == "done":
            reset_reviewed(dataset_dir, result.get("stems", []))


def reset_reviewed(dataset_dir: Path, stems: list[str]) -> None:
    """자동 라벨은 사람이 다시 봐야 한다 — 검수 표시를 뗀다.

    `dataset_dir` 은 **데이터셋** 디렉터리다. 프로젝트 디렉터리를 넘기면 검수 표시가
    거기 없으므로 조용히 아무 일도 안 하고, 기계 라벨이 검수완료 딱지를 단 채 남는다.
    """
    if not stems:
        return
    reviewed = read_reviewed(dataset_dir)
    remaining = reviewed - set(stems)
    if remaining != reviewed:
        write_reviewed(dataset_dir, remaining)


job_manager = JobManager()
