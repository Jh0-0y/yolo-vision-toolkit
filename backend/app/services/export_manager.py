"""ExportManager: thread-pool runner for dataset export with streamed progress.

Export is pure file IO (copy + zip), so it runs on a small ThreadPoolExecutor —
mirroring VideoManager. Progress is written to ``jobs_dir/<export_id>/progress.jsonl``
so ``read_progress(export_id)`` and the SSE endpoint work exactly like videos.
The API process validates first and owns the DB; this only runs the build.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from infra import jobs
from lib.labels.export import ExportCancelled, build_export


def task_dir(export_id: str) -> Path:
    """Progress dir; reuses jobs_dir so the shared SSE reader finds it."""
    return settings.jobs_dir / export_id


def _run(
    export_id: str,
    pdir: Path,
    project_name: str,
    kind: str,
    names: list[str] | None,
    val_split: float,
    seed: int,
) -> None:
    job = jobs.at(settings.jobs_dir, export_id)
    try:
        meta = build_export(
            pdir=pdir,
            project_name=project_name,
            kind=kind,
            names=names,
            val_split=val_split,
            seed=seed,
            export_id=export_id,
            now=datetime.now(timezone.utc),
            emit=job.emit,
            cancel_path=job.cancel_path,
        )
        job.emit({"phase": "done", **meta})
    except ExportCancelled:
        job.emit({"phase": "cancelled", "msg": "Cancelled"})
    except Exception as e:  # noqa: BLE001 — surface any failure to the stream
        job.emit({"phase": "error", "msg": str(e)})


class ExportManager:
    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future] = {}

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="export-build"
            )
        return self._executor

    def submit(
        self,
        export_id: str,
        pdir: Path,
        project_name: str,
        kind: str,
        names: list[str] | None,
        val_split: float,
        seed: int,
    ) -> None:
        jobs.at(settings.jobs_dir, export_id).prepare()

        future = self._get_executor().submit(
            _run, export_id, pdir, project_name, kind, names, val_split, seed
        )
        self._futures[export_id] = future
        future.add_done_callback(lambda _f: self._futures.pop(export_id, None))

    def cancel(self, export_id: str) -> bool:
        future = self._futures.get(export_id)
        if future is not None and future.cancel():  # still queued — never started
            return True
        jobs.at(settings.jobs_dir, export_id).request_cancel()  # running: signal the build loop
        return True

    def is_active(self, export_id: str) -> bool:
        future = self._futures.get(export_id)
        return future is not None and not future.done()


export_manager = ExportManager()
