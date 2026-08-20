"""Training worker process: `python -m app.workers.train_runner <run_dir>`.

Runs ultralytics train in its own process so it can be SIGTERM-stopped and
never loads CUDA/torch in the API. `run_dir` is the run's artifact directory
(project-scoped or the flat pool). Per-epoch metrics stream to
jobs/<run_id>/progress.jsonl (same tail contract as labeling jobs).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from infra import jobs


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
    from app.core.config import resolve_device, settings

    settings.ensure_dirs()
    run_dir = Path(run_dir_arg)
    run_id = run_dir.name
    cfg = json.loads((run_dir / "config.json").read_text())
    # the service may have staged the dataset onto fast scratch (SSD) and points
    # us at the copy; fall back to the canonical path recorded in config.json.
    dataset_path = os.environ.get("YVT_DATASET_PATH_OVERRIDE") or cfg["dataset_path"]

    progress_path = jobs.at(settings.jobs_dir, run_id).ensure().progress_path

    # loading torch/ultralytics + model weights takes a few seconds — tell the UI
    # we're in the prep stage so it doesn't look frozen before epoch 1.
    jobs.emit(progress_path, {"phase": "preparing"})

    # 타일링을 켰으면 여기서 학습 트리를 만든다. 수천 장 인코딩이라 HTTP 요청
    # 안에서 하면 "학습 시작"이 몇 분 매달린다 — 런의 첫 국면으로 옮겼다.
    # tiling_cfg 는 finally 에서도 봐야 하니 try 밖에서 미리 읽어 둔다.
    tiling_cfg = cfg.get("tiling")

    try:
        if tiling_cfg:
            from app.services import datasets
            from lib.labels.dataset_tile import TileDatasetParams, materialize_for_training

            src = Path(cfg["dataset_dir"])
            pid, dsid = src.parent.parent.name, src.name
            composition = materialize_for_training(
                dataset_dir=src,
                out_dir=Path(dataset_path),
                reviewed=datasets.read_reviewed(pid, dsid) & datasets.image_stems(pid, dsid),
                splits=datasets.read_splits(pid, dsid),
                params=TileDatasetParams(
                    tile_size=tiling_cfg["tile_size"],
                    stride=tiling_cfg["stride"],
                    min_visibility=tiling_cfg["min_visibility"],
                    negative_ratio=tiling_cfg["negative_ratio"],
                    keep_all_negatives=tiling_cfg["keep_all_negatives"],
                    seed=tiling_cfg["seed"],
                ),
                emit=lambda ev: jobs.emit(progress_path, ev),
            )
            # 타일이 사라진 뒤에도 무엇으로 학습했는지가 남도록 분할별 구성을
            # progress.jsonl 에 기록해 둔다 (config.json 은 노브만 남긴다).
            jobs.emit(progress_path, {"phase": "materialized", "composition": composition})

        from ultralytics import YOLO

        model = YOLO(cfg["base_model_path"])

        def on_train_start(trainer) -> None:
            # dataset scanned & dataloaders built; training is about to begin
            jobs.emit(progress_path, {"phase": "start", "epochs": int(trainer.epochs)})

        def on_train_epoch_start(trainer) -> None:
            # epoch N is now running (metrics only arrive when it finishes)
            jobs.emit(
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
            jobs.emit(
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

        results = model.train(
            data=str(Path(dataset_path) / "data.yaml"),
            project=str(run_dir.parent),
            name=run_id,
            exist_ok=True,
            device=resolve_device(cfg.get("device")),
            plots=True,
            **cfg.get("params", {}),
        )
    except Exception as e:
        jobs.emit(progress_path, {"phase": "error", "msg": str(e)})
        return 1
    finally:
        # 타일은 진짜 바이트다. seed 까지 고정된 결정론적 산물이고 설정이
        # config.json 에 남으므로 언제든 같은 것을 다시 만들 수 있다 —
        # 런마다 한 벌씩 쌓아 둘 이유가 없다. 하드링크로 펼친 것은 공짜라 남긴다.
        if tiling_cfg:
            shutil.rmtree(dataset_path, ignore_errors=True)

    # per-class metrics (P/R/AP) from the final validation — not in results.csv
    try:
        rows = _per_class_rows(results.box, getattr(results, "names", {}) or {})
        (run_dir / "per_class.json").write_text(json.dumps(rows, ensure_ascii=False))
    except Exception:
        pass

    final = {}
    if results is not None and getattr(results, "results_dict", None):
        final = {k: round(float(v), 5) for k, v in results.results_dict.items()}
    jobs.emit(progress_path, {"phase": "done", "metrics": final})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
