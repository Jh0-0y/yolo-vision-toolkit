"""Dataset export build (pure file IO): copy labeled/original images into a
train/val split + data.yaml, then zip. Emits progress via an ``emit`` callback
so a job runner can stream per-image progress. No DB, no HTTP, no torch."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.domain.yolo_io import atomic_write_text, write_data_yaml
from app.ml.labeling import IMAGE_EXTS

# export kind -> human "type" token used in the name/filename
_EXPORT_TYPE = {"yolo": "train", "images": "original"}

Emit = Callable[[dict], None]


def safe_token(s: str) -> str:
    """Filename-safe token (keeps unicode letters/digits, e.g. Korean)."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s)).strip("_") or "x"


def fdt(dt: datetime) -> str:
    """Local-time stamp for filenames: YYYYMMDD_HHMMSS (naive is assumed UTC)."""
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y%m%d_%H%M%S")
    except Exception:
        return dt.strftime("%Y%m%d_%H%M%S")


def target_images(pdir: Path, names: list[str] | None, kind: str) -> list[Path]:
    """Images eligible for export. For 'yolo' only labeled images qualify."""
    raw = pdir / "raw"
    images = (
        sorted(p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if raw.exists()
        else []
    )
    if names is not None:
        wanted = set(names)
        images = [p for p in images if p.name in wanted]
    if kind == "yolo":
        labels_dir = pdir / "labels"
        images = [p for p in images if (labels_dir / f"{p.stem}.txt").exists()]
    return images


class ExportCancelled(Exception):
    """Raised when a CANCEL sentinel appears mid-build; the caller emits it as a
    terminal 'cancelled' event."""


def build_export(
    *,
    pdir: Path,
    project_name: str,
    kind: str,
    names: list[str] | None,
    val_split: float,
    seed: int,
    export_id: str,
    now: datetime,
    emit: Emit,
    cancel_path: Path | None = None,
) -> dict:
    """Build the export on disk and return its meta. Assumes the caller has
    already verified there is at least one eligible image. ``emit`` receives
    ``{phase: 'start'|'copy'|'zip', ...}`` events for streaming; the caller is
    responsible for the terminal 'done'/'error'/'cancelled' event. Raises
    ``ExportCancelled`` if ``cancel_path`` appears during the copy loop."""

    def _check_cancel() -> None:
        if cancel_path is not None and cancel_path.exists():
            raise ExportCancelled()

    labels_dir = pdir / "labels"
    images = target_images(pdir, names, kind)
    out = pdir / "exports" / export_id
    meta: dict = {
        "id": export_id,
        # {date_time}-{project}-{type} — date-time makes it collision-safe without an id
        "name": f"{fdt(now)}-{safe_token(project_name)}-{_EXPORT_TYPE.get(kind, kind)}",
        "kind": kind,
        "created_at": now.isoformat(),
        "val_split": 0.0,
        "seed": seed,
        "train": 0,
        "val": 0,
        "count": 0,
        "classes": 0,
    }
    total = len(images)
    emit({"phase": "start", "total": total})

    if kind == "images":
        (out / "images").mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images, 1):
            _check_cancel()
            shutil.copy2(img, out / "images" / img.name)
            emit({"phase": "copy", "copied": i, "total": total})
        meta["count"] = total
    else:
        rng = random.Random(seed)
        shuffled = images[:]
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * val_split)
        if val_split > 0 and n_val == 0 and len(shuffled) >= 2:
            n_val = 1
        val_set = set(shuffled[:n_val])

        counts = {"train": 0, "val": 0}
        for i, img in enumerate(images, 1):
            _check_cancel()
            split = "val" if img in val_set else "train"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(labels_dir / f"{img.stem}.txt", out / "labels" / split / f"{img.stem}.txt")
            counts[split] += 1
            emit({"phase": "copy", "copied": i, "total": total})

        classes_path = pdir / "classes.json"
        class_names: dict[int, str] = {}
        if classes_path.exists():
            for c in json.loads(classes_path.read_text()).get("classes", []):
                class_names[int(c["id"])] = c["name"]

        write_data_yaml(
            out / "data.yaml",
            class_names,
            train="images/train",
            val="images/val" if counts["val"] else "images/train",
        )
        meta.update(
            val_split=val_split,
            train=counts["train"],
            val=counts["val"],
            count=len(images),
            classes=len(class_names),
        )

    emit({"phase": "zip"})
    zip_base = pdir / "exports" / export_id
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=out))
    meta["size_bytes"] = zip_path.stat().st_size
    atomic_write_text(out / "export.json", json.dumps(meta))
    return meta
