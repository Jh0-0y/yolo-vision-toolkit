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


def _per_class_rows(box, names: dict) -> list[dict]:
    """Per-class P/R/mAP from an ultralytics metrics `box` (Metric) object."""
    nt = getattr(box, "nt_per_class", None)
    rows = []
    for i, ci in enumerate(box.ap_class_index):
        ci = int(ci)
        p, r, ap50, ap = box.class_result(i)
        rows.append(
            {
                "cls": ci,
                "name": names.get(ci, str(ci)),
                "instances": int(nt[ci]) if nt is not None else None,
                "precision": round(float(p), 5),
                "recall": round(float(r), 5),
                "mAP50": round(float(ap50), 5),
                "mAP50-95": round(float(ap), 5),
            }
        )
    return rows


def main(run_dir_arg: str) -> int:
    from app.config import resolve_device, settings

    settings.ensure_dirs()
    run_dir = Path(run_dir_arg)
    run_id = run_dir.name
    cfg = json.loads((run_dir / "config.json").read_text())

    job_dir = settings.jobs_dir / run_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"

    # loading torch/ultralytics + model weights takes a few seconds — tell the UI
    # we're in the prep stage so it doesn't look frozen before epoch 1.
    _emit(progress_path, {"phase": "preparing"})

    from ultralytics import YOLO

    model = YOLO(cfg["base_model_path"])

    def on_train_start(trainer) -> None:
        # dataset scanned & dataloaders built; training is about to begin
        _emit(progress_path, {"phase": "start", "epochs": int(trainer.epochs)})

    def on_train_epoch_start(trainer) -> None:
        # epoch N is now running (metrics only arrive when it finishes)
        _emit(
            progress_path,
            {"phase": "epoch_start", "epoch": int(trainer.epoch) + 1, "epochs": int(trainer.epochs)},
        )

    def on_fit_epoch_end(trainer) -> None:
        # trainer.metrics is val metrics + fitness; also fold in the epoch's
        # train losses and learning rates so the UI can chart them live.
        merged = dict(trainer.metrics)
        try:
            merged = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **merged}
        except Exception:
            pass
        try:
            merged = {**merged, **trainer.lr}
        except Exception:
            pass
        _emit(
            progress_path,
            {
                "phase": "epoch",
                "epoch": int(trainer.epoch) + 1,
                "epochs": int(trainer.epochs),
                "metrics": {
                    k: round(float(v), 6)
                    for k, v in merged.items()
                    if isinstance(v, (int, float))
                },
            },
        )
        # per-class metrics for this epoch (validation already ran) — appended so
        # the UI can chart "class metric over epochs". Never fail the run over it.
        try:
            v = getattr(trainer, "validator", None)
            box = getattr(getattr(v, "metrics", None), "box", None)
            names = getattr(v, "names", None) or getattr(trainer, "names", {}) or {}
            if box is not None and len(getattr(box, "ap_class_index", [])):
                with open(run_dir / "per_class_history.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {"epoch": int(trainer.epoch) + 1, "metrics": _per_class_rows(box, names)}
                        )
                        + "\n"
                    )
        except Exception:
            pass

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
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

    # per-class metrics (P/R/AP) from the final validation — not in results.csv
    try:
        rows = _per_class_rows(results.box, getattr(results, "names", {}) or {})
        (run_dir / "per_class.json").write_text(json.dumps(rows, ensure_ascii=False))
    except Exception:
        pass

    final = {}
    if results is not None and getattr(results, "results_dict", None):
        final = {k: round(float(v), 5) for k, v in results.results_dict.items()}
    _emit(progress_path, {"phase": "done", "metrics": final})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
