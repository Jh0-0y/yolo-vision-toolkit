"""Test-playground long jobs: video annotation and dataset analysis.

These are long GPU jobs (whole-video encoding, full labeled-set inference) that
do NOT fit the warm inference worker — `infer_manager`'s 180s idle reaper would
kill a multi-minute job mid-run. So this manager owns its OWN
`ProcessPoolExecutor(max_workers=1)` with no reaper and no DB bookkeeping
(Test is a playground). Progress is streamed via `jobs_dir/{job_id}/progress.jsonl`
(reuses `read_progress` + the existing SSE endpoints pattern).
"""

from __future__ import annotations

import multiprocessing
import shutil
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor

from app.core.config import settings

# annotated videos live here transiently; swept after this age (playground = no
# permanent storage).
ANNOTATE_TTL_SEC = 3600


def sweep_old_annotations() -> None:
    """Delete annotate work dirs older than the TTL so nothing accumulates."""
    root = settings.test_dir / "annotate"
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > ANNOTATE_TTL_SEC:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


class TestJobManager:
    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    def _get_executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            return self._executor

    def _prepare(self, job_id: str):
        job_dir = settings.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "progress.jsonl").touch()
        (job_dir / "CANCEL").unlink(missing_ok=True)
        return job_dir

    def submit_annotate(self, job_id: str, cfg: dict) -> None:
        from app.workers import annotate_worker

        self._prepare(job_id)
        future = self._get_executor().submit(
            annotate_worker.run_annotate, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def submit_analyze(self, job_id: str, cfg: dict) -> None:
        from app.workers import analyze_worker

        self._prepare(job_id)
        future = self._get_executor().submit(
            analyze_worker.run_analyze, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def is_active(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        return future is not None and not future.done()

    def cancel(self, job_id: str) -> None:
        (settings.jobs_dir / job_id / "CANCEL").touch()

    def shutdown(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None


test_job_manager = TestJobManager()
