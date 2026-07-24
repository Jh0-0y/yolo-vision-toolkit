"""Read/write per-image labels and the reviewed flag.

Labels live as YOLO txt at labels/{stem}.txt — file exists means "labeled"
(an empty file is a deliberate negative). The user-facing reviewed flag
lives in reviewed.json as {"<stem>": true}.

Editor box = {"id", "cls", "xyxy_n": [x1,y1,x2,y2], ...}.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.yolo_io import (
    atomic_write_text,
    read_box_meta,
    read_label_file,
    write_box_meta,
    write_label_file,
)


def label_path(pdir: Path, stem: str) -> Path:
    return pdir / "labels" / f"{stem}.txt"


def _reviewed_path(pdir: Path) -> Path:
    return pdir / "reviewed.json"


def is_labeled(pdir: Path, stem: str) -> bool:
    return label_path(pdir, stem).exists()


def read_boxes(pdir: Path, stem: str) -> list[dict]:
    """Boxes for one image as a list of editor-shaped dicts (may be empty).

    Attaches each box's confidence and review status from the meta sidecar.
    """
    path = label_path(pdir, stem)
    if not path.exists():
        return []
    metas = read_box_meta(path)
    boxes: list[dict] = []
    for i, (cls, xyxy) in enumerate(read_label_file(path)):
        box: dict = {"id": f"{stem}_{i}", "cls": cls, "xyxy_n": list(xyxy)}
        meta = metas[i] if i < len(metas) else {}
        if meta.get("score") is not None:
            box["score"] = meta["score"]
        if meta.get("status"):
            box["status"] = meta["status"]
        boxes.append(box)
    return boxes


def write_boxes(pdir: Path, stem: str, boxes: list[dict]) -> None:
    path = label_path(pdir, stem)
    write_label_file(
        path,
        [(int(b["cls"]), tuple(b["xyxy_n"])) for b in boxes],
    )
    write_box_meta(
        path,
        [{"score": b.get("score"), "status": b.get("status")} for b in boxes],
    )


def label_classes(pdir: Path, stem: str) -> set[int]:
    """Class ids present in one image's label file (empty if unlabeled/negative)."""
    path = label_path(pdir, stem)
    if not path.exists():
        return set()
    classes: set[int] = set()
    for line in path.read_text().splitlines():
        head = line.split(maxsplit=1)[:1]
        if head:
            try:
                classes.add(int(head[0]))
            except ValueError:
                continue
    return classes


def read_reviewed(pdir: Path) -> set[str]:
    path = _reviewed_path(pdir)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {stem for stem, flag in data.items() if flag}


def write_reviewed(pdir: Path, stems: set[str]) -> None:
    atomic_write_text(
        _reviewed_path(pdir),
        json.dumps({stem: True for stem in sorted(stems)}, ensure_ascii=False, indent=2),
    )


def set_reviewed(pdir: Path, stem: str, flag: bool) -> set[str]:
    stems = read_reviewed(pdir)
    if flag:
        stems.add(stem)
    else:
        stems.discard(stem)
    write_reviewed(pdir, stems)
    return stems
