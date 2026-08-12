"""Inference worker: warm model residency + prediction, in a child process.

Runs as a long-lived pool worker (ProcessPoolExecutor, max_workers=1). The
module-global `_RESIDENT` dict keeps loaded models alive **across** submitted
calls, so repeated tests and video scrubbing don't re-read weights from disk.
Multiple models can be resident at once (for A/B and ensemble).

torch/ultralytics load lazily on first `load_model` — the API process never
imports them. All functions are module-level and picklable (spawn-safe).

The worker is a pure calculator: it loads what it's told, predicts, and returns
plain dicts. It has no DB access and makes no policy decisions — the manager
(services/infer_manager.py) resolves paths/devices and owns lifecycle.
"""

from __future__ import annotations

import time
from pathlib import Path

# model_id -> {model, path, device, names, task, weight_mb, loaded_at}
# Survives between submitted calls because the pool worker process persists.
_RESIDENT: dict[str, dict] = {}


def _meta(model_id: str, entry: dict) -> dict:
    return {
        "model_id": model_id,
        "device": entry["device"],
        "task": entry["task"],
        "classes": entry["names"],
        "weight_mb": entry["weight_mb"],
        "loaded_at": entry["loaded_at"],
    }


def load_model(model_id: str, pt_path: str, device: str) -> dict:
    """Load (or reuse) a model into memory on `device`. Returns its metadata."""
    existing = _RESIDENT.get(model_id)
    if existing is not None and existing["path"] == pt_path and existing["device"] == device:
        return _meta(model_id, existing)

    from ultralytics import YOLO

    model = YOLO(pt_path)
    try:
        model.to(device)
    except Exception:
        # some device strings (e.g. "mps" unavailable) — fall back at predict time
        pass
    names = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
    task = getattr(model, "task", "detect") or "detect"
    weight_mb = (
        round(Path(pt_path).stat().st_size / 1024 / 1024, 1) if Path(pt_path).exists() else 0.0
    )
    entry = {
        "model": model,
        "path": pt_path,
        "device": device,
        "names": names,
        "task": task,
        "weight_mb": weight_mb,
        "loaded_at": time.time(),
    }
    _RESIDENT[model_id] = entry
    return _meta(model_id, entry)


def unload_model(model_id: str) -> bool:
    """Drop a model from residency, freeing its weights (framework context stays
    until the worker process itself exits)."""
    entry = _RESIDENT.pop(model_id, None)
    if entry is None:
        return False
    entry["model"] = None
    return True


def list_residents() -> list[dict]:
    return [_meta(mid, e) for mid, e in _RESIDENT.items()]


def run_predict(specs: list[tuple[str, str]], image, cfg_dict: dict) -> dict:
    """Predict over one image with the given models.

    `specs` is a list of (model_id, pt_path). Any model not already resident on
    the requested device is loaded first (lazy warm). `image` is a path str.
    `cfg_dict` carries a resolved `device` plus conf/iou_wbf/imgsz.
    """
    from lib.detect.predict import PredictConfig, predict_image

    device = cfg_dict.get("device") or "cpu"
    models: list[tuple[str, object]] = []
    for model_id, pt_path in specs:
        entry = _RESIDENT.get(model_id)
        if entry is None or entry["path"] != pt_path or entry["device"] != device:
            load_model(model_id, pt_path, device)
            entry = _RESIDENT[model_id]
        models.append((model_id, entry["model"]))

    cfg = PredictConfig(
        conf=cfg_dict.get("conf", 0.4),
        iou_wbf=cfg_dict.get("iou_wbf", 0.55),
        imgsz=cfg_dict.get("imgsz", 640),
        device=device,
    )
    return predict_image(models, image, cfg)
