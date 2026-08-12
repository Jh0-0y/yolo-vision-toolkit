"""YOLO label file read/write and data.yaml generation. Writes are atomic."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml


def xyxyn_to_yolo(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def yolo_to_xyxyn(cxcywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = cxcywh
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_label_file(path: Path, boxes: list[tuple[int, tuple[float, float, float, float]]]) -> None:
    """boxes: list of (class_id, xyxy normalized)."""
    lines = []
    for cls, xyxy in boxes:
        cx, cy, w, h = xyxyn_to_yolo(tuple(_clamp01(v) for v in xyxy))  # type: ignore[arg-type]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _meta_path(label_path: Path) -> Path:
    """Sidecar holding per-box metadata (confidence + review status), aligned by
    line index to the txt.

    Kept out of the YOLO .txt (which stays 5-column for training compatibility).
    Each entry is a dict like {"score": 0.42, "status": "needs_review"}.
    """
    return label_path.with_suffix(".meta.json")


def _meta_is_empty(meta: dict) -> bool:
    return meta.get("score") is None and not meta.get("status")


def write_box_meta(label_path: Path, metas: list[dict]) -> None:
    """Persist per-box metadata next to the label file, or remove the sidecar
    when nothing meaningful is stored (no scores and no statuses)."""
    path = _meta_path(label_path)
    if all(_meta_is_empty(m) for m in metas):
        path.unlink(missing_ok=True)
        return
    atomic_write_text(path, json.dumps(metas))


def read_box_meta(label_path: Path) -> list[dict]:
    path = _meta_path(label_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def read_label_file(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
        boxes.append((cls, yolo_to_xyxyn((cx, cy, w, h))))
    return boxes


def write_data_yaml(path: Path, names: dict[int, str], train: str, val: str | None = None) -> None:
    # names must cover the full contiguous id space so label ids stay valid.
    # No "path" key on purpose: ultralytics then resolves train/val relative to
    # this yaml's own directory, which keeps the exported zip portable.
    nc = max(names) + 1 if names else 0
    full = {i: names.get(i, f"class_{i}") for i in range(nc)}
    data = {
        "train": train,
        "val": val or train,
        "nc": nc,
        "names": full,
    }
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
