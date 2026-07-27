"""Stage a training dataset onto fast scratch (e.g. an SSD) and tear it down.

Pure file IO — no DB, no torch, no process management. The *service* layer
decides **when** to stage and owns the cleanup lifecycle; this module only knows
**how** to copy a dataset directory onto the cache and whether it fits.

Motivation: YOLO re-reads every image each epoch in shuffled order. On an HDD
that random read pattern starves the GPU. When a small fast disk is available we
copy the active dataset there for the duration of the run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def staged_path(cache_dir: Path, run_id: str) -> Path:
    """Where a run's staged dataset copy lives — one folder per run."""
    return cache_dir / run_id


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _copytree(src: Path, dest: Path) -> None:
    """rsync when available (fast, handles huge trees well); else copytree."""
    if shutil.which("rsync"):
        # trailing slashes: copy *contents* of src into dest
        subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dest}/"], check=True)
    else:
        shutil.copytree(src, dest, dirs_exist_ok=True)


def stage_dataset(src: Path, cache_dir: Path, run_id: str) -> tuple[Path | None, str]:
    """Copy the ``src`` dataset dir onto ``cache_dir/<run_id>``.

    Returns ``(path_to_use_or_None, human_note)``. A ``None`` path means "don't
    stage — train from ``src`` as-is"; the caller falls back to the original
    path. This module never raises for the expected degradations (already on the
    fast disk, not enough room, copy failed): each returns ``(None, reason)`` so
    training still proceeds, just from the slower original location.
    """
    src = src.resolve()
    cache_dir = cache_dir.resolve()

    # already on the fast disk (e.g. datasets_dir itself points at the SSD)
    try:
        if src == cache_dir or cache_dir in src.parents:
            return None, "dataset already on fast storage; staging skipped"
    except OSError:
        pass

    dest = cache_dir / run_id
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return None, f"cannot create cache dir ({e}); training from original path"

    size = _dir_size_bytes(src)
    free = shutil.disk_usage(cache_dir).free
    # 10% margin so training scratch (label cache, val plots) still fits
    if free < size * 1.1:
        return None, (
            f"not enough fast-storage space (need ~{size // 2**20} MB, "
            f"free {free // 2**20} MB); training from original path"
        )

    # fresh copy every run — clear any stale leftover first
    try:
        shutil.rmtree(dest, ignore_errors=True)
        _copytree(src, dest)
    except Exception as e:  # noqa: BLE001 — degrade to original path on any copy error
        shutil.rmtree(dest, ignore_errors=True)
        return None, f"staging copy failed ({e}); training from original path"
    return dest, f"staged {size // 2**20} MB to fast storage"


def unstage(cache_dir: Path, run_id: str) -> None:
    """Remove a run's staged copy. Best-effort; safe when it doesn't exist."""
    shutil.rmtree(cache_dir / run_id, ignore_errors=True)
