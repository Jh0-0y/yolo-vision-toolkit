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
from pathlib import Path

from app.core.config import settings

# annotated videos live here transiently; swept after this age (playground = no
# permanent storage).
ANNOTATE_TTL_SEC = 3600


def _sweep_dir(root: Path, ttl_sec: int) -> None:
    """Delete work dirs older than the TTL so nothing accumulates."""
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > ttl_sec:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def sweep_old_annotations() -> None:
    """Delete annotate work dirs older than the TTL so nothing accumulates."""
    _sweep_dir(settings.test_dir / "annotate", ANNOTATE_TTL_SEC)


def sweep_old_compare() -> None:
    """Delete compare upload/dataset dirs older than the TTL (playground = no
    permanent storage; uploaded test sets can be large)."""
    _sweep_dir(settings.test_dir / "compare", ANNOTATE_TTL_SEC)


def sweep_old_live() -> None:
    """Delete live-preview cache dirs (detections + preview.mp4) older than the TTL."""
    _sweep_dir(settings.test_dir / "live", ANNOTATE_TTL_SEC)


class TestJobManager:
    """Owns two single-worker pools so a long video-tracking job and a
    model-comparison/analysis job don't block each other (they used to share one
    worker). Each pool is spawn-based, has no idle reaper, and does no DB work."""

    def __init__(self) -> None:
        self._executors: dict[str, ProcessPoolExecutor] = {}
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    def _get_executor(self, kind: str) -> ProcessPoolExecutor:
        with self._lock:
            if kind not in self._executors:
                self._executors[kind] = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            return self._executors[kind]

    def _prepare(self, job_id: str):
        job_dir = settings.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "progress.jsonl").touch()
        (job_dir / "CANCEL").unlink(missing_ok=True)
        return job_dir

    def submit_annotate(self, job_id: str, cfg: dict) -> None:
        from app.workers import annotate_worker

        self._prepare(job_id)
        future = self._get_executor("video").submit(
            annotate_worker.run_annotate, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def submit_compare(self, job_id: str, cfg: dict) -> None:
        from app.workers import compare_worker

        self._prepare(job_id)
        future = self._get_executor("eval").submit(
            compare_worker.run_compare, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def submit_live(self, job_id: str, cfg: dict) -> None:
        """Live-preview detection pass — shares the single "video" worker with
        annotate so a heavy GPU detection and a heavy render never run at once."""
        from app.workers import live_worker

        self._prepare(job_id)
        future = self._get_executor("video").submit(
            live_worker.run_live, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def submit_live_render(self, job_id: str, cfg: dict) -> None:
        """라이브 세션 캐시로 오버레이 영상 렌더 — 추론 없음(cv2/ffmpeg만)."""
        from app.workers import live_worker

        self._prepare(job_id)
        future = self._get_executor("video").submit(
            live_worker.run_live_render, job_id, cfg, str(settings.jobs_dir)
        )
        self._futures[job_id] = future

    def is_active(self, job_id: str) -> bool:
        future = self._futures.get(job_id)
        return future is not None and not future.done()

    def cancel(self, job_id: str) -> None:
        (settings.jobs_dir / job_id / "CANCEL").touch()

    def shutdown(self) -> None:
        with self._lock:
            for ex in self._executors.values():
                ex.shutdown(wait=False, cancel_futures=True)
            self._executors = {}


test_job_manager = TestJobManager()
