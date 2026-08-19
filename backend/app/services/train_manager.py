"""Spawns and tracks training subprocesses (one at a time)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from app.core.config import settings
from app.db import session_scope
from app.models import TrainRun
from infra import jobs
from lib.train.staging import stage_dataset, unstage


class TrainManager:
    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        # runs that are staging their dataset but haven't spawned a worker yet —
        # they must count as active so a second run can't start meanwhile
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def has_active(self) -> bool:
        with self._lock:
            if self._pending:
                return True
            return any(p.poll() is None for p in self._procs.values())

    def start(self, run_id: str) -> int | None:
        # reclaim any idle inference worker so training gets a clean device
        # (esp. Apple unified memory) — lazy import avoids a services cycle
        try:
            from app.services.infer_manager import infer_manager

            infer_manager.evict_for_training()
        except Exception:
            pass
        # fast-scratch staging can take minutes for a large dataset; do it off
        # the request thread so the HTTP start call returns immediately. Without
        # a cache dir configured, spawn synchronously as before (no behavior
        # change for the default HDD/SSD-single-disk setup).
        if settings.ssd_cache_dir is not None:
            with self._lock:
                self._pending.add(run_id)
            threading.Thread(
                target=self._prepare_and_run, args=(run_id,), daemon=True
            ).start()
            return None
        return self._spawn(run_id)

    def _prepare_and_run(self, run_id: str) -> None:
        """Stage the dataset onto fast scratch, then spawn the worker. Runs in a
        background thread; must always clear `_pending` so a staging failure
        can't wedge `has_active()` at True forever."""
        with session_scope() as session:
            run = session.get(TrainRun, run_id)
            dataset_path = run.dataset_path if run is not None else None
        override: str | None = None
        if dataset_path:
            self._append_progress(run_id, {"phase": "staging"})
            try:
                staged, note = stage_dataset(
                    Path(dataset_path), settings.ssd_cache_dir, run_id
                )
            except Exception as e:  # noqa: BLE001 — never let staging kill the run
                staged, note = None, f"staging error ({e}); training from original path"
            self._append_progress(run_id, {"phase": "staging", "msg": note})
            if staged is not None:
                override = str(staged)
        try:
            self._spawn(run_id, dataset_override=override)
        except Exception as e:  # noqa: BLE001 — spawn failed; unwedge and record
            with self._lock:
                self._pending.discard(run_id)
            self._append_progress(run_id, {"phase": "error", "msg": str(e)})
            self._update(
                run_id,
                status="error",
                error=str(e),
                finished_at=datetime.now(timezone.utc),
            )
            unstage(settings.ssd_cache_dir, run_id)

    def _spawn(self, run_id: str, dataset_override: str | None = None) -> int:
        backend_dir = Path(__file__).resolve().parents[2]
        with session_scope() as session:
            run = session.get(TrainRun, run_id)
            project_id = run.project_id if run is not None else None
        run_dir = settings.run_dir(project_id, run_id)
        env = {**os.environ, "YVT_DATA_DIR": str(settings.data_dir)}
        if dataset_override:
            env["YVT_DATASET_PATH_OVERRIDE"] = dataset_override
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.workers.train_runner", str(run_dir)],
            cwd=backend_dir,
            env=env,
            stdout=open(run_dir / "train.log", "w"),
            stderr=subprocess.STDOUT,
        )
        with self._lock:
            self._procs[run_id] = proc
            self._pending.discard(run_id)
        self._update(run_id, status="running", pid=proc.pid)
        threading.Thread(target=self._watch, args=(run_id, proc), daemon=True).start()
        return proc.pid

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(run_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True
        # process from a previous API lifetime: fall back to the stored pid
        with session_scope() as session:
            run = session.get(TrainRun, run_id)
            if run is None or run.pid is None or run.status != "running":
                return False
            try:
                os.kill(run.pid, signal.SIGTERM)
                return True
            except ProcessLookupError:
                self._update(run_id, status="stopped", finished_at=datetime.now(timezone.utc))
                return False

    def _watch(self, run_id: str, proc: subprocess.Popen) -> None:
        code = proc.wait()
        events, _ = jobs.at(settings.jobs_dir, run_id).read()
        phases = {e.get("phase") for e in events}
        last_metrics = next(
            (e.get("metrics") for e in reversed(events) if e.get("metrics")), None
        )
        if "done" in phases:
            status = "done"
        elif code in (-signal.SIGTERM, -signal.SIGINT, -signal.SIGKILL):
            status = "stopped"
            self._append_progress(run_id, {"phase": "cancelled"})
        else:
            status = "error"
            if "error" not in phases:
                self._append_progress(run_id, {"phase": "error", "msg": f"exit code {code}"})
        self._update(
            run_id,
            status=status,
            finished_at=datetime.now(timezone.utc),
            metrics_json=json.dumps(last_metrics) if last_metrics else None,
            error=None if status != "error" else f"exit code {code}",
        )
        with self._lock:
            self._procs.pop(run_id, None)
        # tear down the staged copy regardless of outcome (SIGTERM skips the
        # worker's own finally blocks, so cleanup has to live out here)
        if settings.ssd_cache_dir is not None:
            unstage(settings.ssd_cache_dir, run_id)
        # 학습이 먹은 트리(`run_dir/dataset`)는 **하드링크**라 디스크를 차지하지 않고,
        # 런을 지울 때 함께 사라진다 — 여기서 따로 치울 것이 없다.

    @staticmethod
    def _append_progress(run_id: str, event: dict) -> None:
        """워커가 죽어서 스스로 남기지 못한 종료 이벤트를 API 프로세스가 대신 쓴다."""
        jobs.at(settings.jobs_dir, run_id).ensure().emit(event)

    @staticmethod
    def _update(run_id: str, **fields) -> None:
        with session_scope() as session:
            run = session.get(TrainRun, run_id)
            if run is None:
                return
            for k, v in fields.items():
                setattr(run, k, v)
            session.add(run)
            session.commit()

    def reconcile_on_boot(self) -> None:
        """Mark runs that were 'running' when the API died, and sweep any staged
        dataset copies left behind by runs that are no longer alive."""
        alive_ids: set[str] = set()
        with session_scope() as session:
            for run in session.exec(select(TrainRun).where(TrainRun.status == "running")):
                alive = False
                if run.pid:
                    try:
                        os.kill(run.pid, 0)
                        alive = True
                    except ProcessLookupError:
                        pass
                if alive:
                    alive_ids.add(run.id)
                else:
                    run.status = "error"
                    run.error = "process missing after API restart"
                    session.add(run)
            session.commit()
        # a training subprocess from a previous API lifetime may still be running
        # and reading its staged copy — keep those; delete every other staged dir.
        if settings.ssd_cache_dir is not None and settings.ssd_cache_dir.exists():
            for d in settings.ssd_cache_dir.iterdir():
                if d.is_dir() and d.name not in alive_ids:
                    shutil.rmtree(d, ignore_errors=True)


train_manager = TrainManager()
