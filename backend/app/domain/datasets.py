"""Uploaded-dataset lifecycle helpers.

Pure file IO — no DB, no torch, no process management. The *service* layer
(train_manager) decides **when** to clean up; this module only knows **how** to
locate an uploaded dataset's folder, read its self-delete flag, and remove it.

An "upload" is a dataset extracted under ``settings.datasets_dir/<id>`` with a
``dataset.json`` beside it. Project exports live elsewhere and are not touched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def upload_base(dataset_path: Path, datasets_dir: Path) -> Path | None:
    """The ``datasets_dir/<id>`` folder for an uploaded dataset.

    ``dataset_path`` may point at a subdirectory (the data.yaml dir) inside the
    upload; this returns the top-level ``<id>`` folder. Returns ``None`` when the
    path is not under ``datasets_dir`` (e.g. a project export)."""
    try:
        rel = dataset_path.resolve().relative_to(datasets_dir.resolve())
    except (ValueError, OSError):
        return None
    if not rel.parts:
        return None
    return datasets_dir.resolve() / rel.parts[0]


def is_auto_delete(base: Path) -> bool:
    """Whether an uploaded dataset was flagged to self-delete after training."""
    try:
        meta = json.loads((base / "dataset.json").read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(meta.get("auto_delete", False))


def delete_upload(base: Path) -> None:
    """Remove an uploaded dataset folder. Best-effort; safe if already gone."""
    shutil.rmtree(base, ignore_errors=True)
