"""Spawns and tracks training subprocesses (one at a time)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from app.config import settings
from app.db import session_scope
from app.jobs.runner import read_progress
from app.models import TrainRun


class TrainManager:
    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def has_active(self) -> bool:
        with self._lock:
            return any(p.poll() is None for p in self._procs.values())

    def start(self, run_id: str) -> int:
        backend_dir = Path(__file__).resolve().parents[2]
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.core.training_runner", run_id],
            cwd=backend_dir,
            env={**os.environ, "YVT_DATA_DIR": str(settings.data_dir)},
            stdout=open(settings.runs_dir / run_id / "train.log", "w"),
            stderr=subprocess.STDOUT,
        )
        with self._lock:
            self._procs[run_id] = proc
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
        events, _ = read_progress(run_id)
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

    @staticmethod
    def _append_progress(run_id: str, event: dict) -> None:
        import time

        path = settings.jobs_dir / run_id / "progress.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"ts": time.time(), **event}) + "\n")

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
        """Mark runs that were 'running' when the API died."""
        with session_scope() as session:
            for run in session.exec(select(TrainRun).where(TrainRun.status == "running")):
                alive = False
                if run.pid:
                    try:
                        os.kill(run.pid, 0)
                        alive = True
                    except ProcessLookupError:
                        pass
                if not alive:
                    run.status = "error"
                    run.error = "process missing after API restart"
                    session.add(run)
            session.commit()


train_manager = TrainManager()
