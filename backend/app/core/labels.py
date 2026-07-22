"""Read/write labels for the confirmed and review buckets, in editor box shape.

Editor box = {"id", "cls", "xyxy_n": [x1,y1,x2,y2], ...}. Confirmed labels live
as YOLO txt (confirmed/labels/{stem}.txt); review labels live as JSON with an
optional "edits" overlay (review/{stem}.json).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.yolo_io import read_label_file, write_label_file

BUCKETS = ("confirmed", "review")


def _confirmed_label_path(pdir: Path, stem: str) -> Path:
    return pdir / "confirmed" / "labels" / f"{stem}.txt"


def _review_json_path(pdir: Path, stem: str) -> Path:
    return pdir / "review" / f"{stem}.json"


def read_boxes(pdir: Path, bucket: str, stem: str) -> list[dict]:
    """Boxes for one image as a list of editor-shaped dicts (may be empty)."""
    if bucket == "confirmed":
        path = _confirmed_label_path(pdir, stem)
        if not path.exists():
            return []
        return [
            {"id": f"{stem}_{i}", "cls": cls, "xyxy_n": list(xyxy)}
            for i, (cls, xyxy) in enumerate(read_label_file(path))
        ]
    if bucket == "review":
        path = _review_json_path(pdir, stem)
        if not path.exists():
            return []
        payload = json.loads(path.read_text())
        return payload.get("edits") or payload.get("boxes", [])
    raise ValueError(f"unknown bucket: {bucket}")


def write_boxes(pdir: Path, bucket: str, stem: str, boxes: list[dict]) -> None:
    if bucket == "confirmed":
        write_label_file(
            _confirmed_label_path(pdir, stem),
            [(int(b["cls"]), tuple(b["xyxy_n"])) for b in boxes],
        )
        return
    if bucket == "review":
        path = _review_json_path(pdir, stem)
        payload = json.loads(path.read_text()) if path.exists() else {}
        payload["edits"] = boxes
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    raise ValueError(f"unknown bucket: {bucket}")
