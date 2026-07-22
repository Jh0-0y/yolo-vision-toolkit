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

from app.config import settings
from app.core.labels import read_reviewed, write_reviewed
from app.db import session_scope
from app.jobs.worker import run_label_job
from app.models import Job


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

    def submit_label_job(self, job_id: str, project_id: str, cfg: dict) -> None:
        job_dir = settings.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "progress.jsonl").touch()

        future = self._get_executor().submit(
            run_label_job, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future
        self._mark_status(job_id, "running")
        future.add_done_callback(lambda f: self._on_done(job_id, project_id, f))

    def cancel(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        if future is None:
            return False
        if future.cancel():  # still queued — never started
            self._mark_status(job_id, "cancelled")
            return True
        # running: signal the worker via sentinel file
        (settings.jobs_dir / job_id / "CANCEL").touch()
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

    def _on_done(self, job_id: str, project_id: str, future: Future) -> None:
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
            reset_reviewed(project_id, result.get("stems", []))


def reset_reviewed(project_id: str, stems: list[str]) -> None:
    """Auto-labeled images need a fresh review — drop their reviewed flag."""
    if not stems:
        return
    pdir = settings.projects_dir / project_id
    reviewed = read_reviewed(pdir)
    remaining = reviewed - set(stems)
    if remaining != reviewed:
        write_reviewed(pdir, remaining)


job_manager = JobManager()


def read_progress(job_id: str, offset: int = 0) -> tuple[list[dict], int]:
    """Read progress events from byte offset; returns (events, new_offset)."""
    path = settings.jobs_dir / job_id / "progress.jsonl"
    if not path.exists():
        return [], offset
    events = []
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    consumed = chunk.rfind(b"\n") + 1  # leave any partial trailing line for next poll
    for line in chunk[:consumed].splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, offset + consumed
