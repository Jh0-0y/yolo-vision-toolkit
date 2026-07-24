"""InferManager: owns the warm inference worker + its lifecycle.

Lives in the API process (singleton). This is the *business* layer for the Test
playground: it resolves model paths/devices, makes policy decisions (adaptive
device, idle eviction, evict-on-training-start), and mirrors residency — while
delegating every torch operation to a single persistent pool worker
(workers/infer_worker.py) so models stay warm across calls.

Test inference is a playground: nothing is written to the DB here.

Sibling of label_manager/train_manager; borrows label_manager's persistent
ProcessPoolExecutor pattern, but adds explicit model residency + an idle reaper.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from app.workers import infer_worker

# Free the worker process (and its framework/CUDA context) after this much idle.
# Warm residency is a best-effort cache, not a guarantee — a walked-away Test
# page must not hold memory hostage from a training run.
IDLE_TTL_SEC = 180


class InferManager:
    def __init__(self) -> None:
        self._executor: ProcessPoolExecutor | None = None
        self._lock = threading.RLock()
        self._residents: dict[str, dict] = {}  # mirror of worker residency
        self._last_activity = time.time()
        self._reaper: threading.Thread | None = None

    # --- executor lifecycle ---
    def _get_executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
                self._start_reaper()
            return self._executor

    def _submit(self, fn, *args):
        executor = self._get_executor()
        self._last_activity = time.time()
        return executor.submit(fn, *args).result()

    def _start_reaper(self) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        while True:
            time.sleep(30)
            with self._lock:
                if self._executor is None:
                    return
                if time.time() - self._last_activity > IDLE_TTL_SEC:
                    self._shutdown_locked()
                    return

    def _shutdown_locked(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._residents.clear()

    def evict_for_training(self) -> None:
        """Called when a training job starts: reclaim the idle inference worker's
        memory so the training run gets a clean device."""
        with self._lock:
            self._shutdown_locked()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()

    # --- device policy (adaptive) ---
    def resolve_device(self, requested: str | None) -> str:
        from app.core.config import resolve_device as _resolve

        if requested and requested != "auto":
            return _resolve(requested)
        resolved = _resolve(None)  # cuda > mps > cpu
        # On Apple unified memory, a running train job shares the single GPU —
        # default test inference to CPU so it doesn't contend.
        if resolved == "mps" and self._training_active():
            return "cpu"
        return resolved

    def _training_active(self) -> bool:
        try:
            from app.services.train_manager import train_manager

            return train_manager.has_active()
        except Exception:
            return False

    # --- residency ---
    def load(self, model_id: str, pt_path: str, requested_device: str | None = None) -> dict:
        device = self.resolve_device(requested_device)
        meta = self._submit(infer_worker.load_model, model_id, pt_path, device)
        with self._lock:
            self._residents[model_id] = meta
        return meta

    def unload(self, model_id: str) -> bool:
        with self._lock:
            has_worker = self._executor is not None
        ok = self._submit(infer_worker.unload_model, model_id) if has_worker else False
        with self._lock:
            self._residents.pop(model_id, None)
        return ok

    def residents(self) -> list[dict]:
        with self._lock:
            if self._executor is None:
                return []
        return self._submit(infer_worker.list_residents)

    def resident_count(self) -> int:
        """Mirror-only count — never touches the worker, so resource polling
        doesn't keep the worker alive or spin it up."""
        with self._lock:
            return len(self._residents) if self._executor is not None else 0

    # --- prediction ---
    def predict(self, specs: list[tuple[str, str]], image, cfg: dict) -> dict:
        """specs: list of (model_id, pt_path). cfg carries requested device +
        conf/iou_wbf/imgsz. Returns the worker's JSON-ready result plus the
        resolved device actually used."""
        device = self.resolve_device(cfg.get("device"))
        result = self._submit(infer_worker.run_predict, specs, image, {**cfg, "device": device})
        with self._lock:
            for model_id, _ in specs:
                self._residents.setdefault(model_id, {"model_id": model_id, "device": device})
        result["device"] = device
        return result


infer_manager = InferManager()
