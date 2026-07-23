"""Training worker process: `python -m app.core.training_runner <run_dir>`.

Runs ultralytics train in its own process so it can be SIGTERM-stopped and
never loads CUDA/torch in the API. `run_dir` is the run's artifact directory
(project-scoped or the flat pool). Per-epoch metrics stream to
jobs/<run_id>/progress.jsonl (same tail contract as labeling jobs).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _emit(path: Path, event: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def main(run_dir_arg: str) -> int:
    from app.config import resolve_device, settings

    settings.ensure_dirs()
    run_dir = Path(run_dir_arg)
    run_id = run_dir.name
    cfg = json.loads((run_dir / "config.json").read_text())

    job_dir = settings.jobs_dir / run_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"

    from ultralytics import YOLO

    model = YOLO(cfg["base_model_path"])

    def on_fit_epoch_end(trainer) -> None:
        _emit(
            progress_path,
            {
                "phase": "epoch",
                "epoch": int(trainer.epoch) + 1,
                "epochs": int(trainer.epochs),
                "metrics": {k: round(float(v), 5) for k, v in trainer.metrics.items()},
            },
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    try:
        results = model.train(
            data=str(Path(cfg["dataset_path"]) / "data.yaml"),
            project=str(run_dir.parent),
            name=run_id,
            exist_ok=True,
            device=resolve_device(cfg.get("device")),
            plots=True,
            **cfg.get("params", {}),
        )
    except Exception as e:
        _emit(progress_path, {"phase": "error", "msg": str(e)})
        return 1

    final = {}
    if results is not None and getattr(results, "results_dict", None):
        final = {k: round(float(v), 5) for k, v in results.results_dict.items()}
    _emit(progress_path, {"phase": "done", "metrics": final})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
