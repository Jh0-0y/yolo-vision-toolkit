"""Per-project class registry management (classes.json).

Classes are stored as a contiguous, append-only id space so that YOLO label
files (which reference class ids by position) stay valid. Deleting a class is
therefore a project-wide operation: boxes of that class are dropped and every
higher class id is shifted down by one across all label files.
"""

from __future__ import annotations

import json
from pathlib import Path

from lib.labels.io import atomic_write_text, read_box_meta, write_box_meta


def _classes_path(pdir: Path) -> Path:
    return pdir / "classes.json"


def _normalize(name: str) -> str:
    return name.strip().lower()


def read_classes(pdir: Path) -> list[dict]:
    path = _classes_path(pdir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("classes", [])
    except (json.JSONDecodeError, OSError):
        return []


def write_classes(pdir: Path, classes: list[dict]) -> None:
    atomic_write_text(
        _classes_path(pdir),
        json.dumps({"classes": classes}, ensure_ascii=False, indent=2),
    )


def add_class(pdir: Path, name: str) -> dict:
    """Append a new class with the next contiguous id. Returns the new class.

    Raises ValueError on a blank or duplicate (case-insensitive) name.
    """
    name = name.strip()
    if not name:
        raise ValueError("Class name cannot be empty")
    classes = read_classes(pdir)
    if any(_normalize(c["name"]) == _normalize(name) for c in classes):
        raise ValueError(f'Class "{name}" already exists')
    new_class = {"id": len(classes), "name": name, "sources": ["manual"]}
    classes.append(new_class)
    write_classes(pdir, classes)
    return new_class


def rename_class(pdir: Path, class_id: int, name: str) -> dict:
    name = name.strip()
    if not name:
        raise ValueError("Class name cannot be empty")
    classes = read_classes(pdir)
    target = next((c for c in classes if c["id"] == class_id), None)
    if target is None:
        raise KeyError(class_id)
    if any(c["id"] != class_id and _normalize(c["name"]) == _normalize(name) for c in classes):
        raise ValueError(f'Class "{name}" already exists')
    target["name"] = name
    write_classes(pdir, classes)
    return target


def _reindex_label_file(txt_path: Path, drop_id: int) -> None:
    """Drop boxes of ``drop_id`` and shift higher class ids down by one, keeping
    the meta sidecar aligned by line index."""
    lines = txt_path.read_text().splitlines()
    metas = read_box_meta(txt_path)
    new_lines: list[str] = []
    new_metas: list[dict] = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        if cls == drop_id:
            continue  # box (and its meta) removed
        if cls > drop_id:
            parts[0] = str(cls - 1)
        new_lines.append(" ".join(parts))
        new_metas.append(metas[i] if i < len(metas) else {})
    atomic_write_text(txt_path, "\n".join(new_lines) + ("\n" if new_lines else ""))
    write_box_meta(txt_path, new_metas)


def count_boxes_with_class(pdir: Path, class_id: int) -> int:
    """How many boxes across the project reference ``class_id`` (for a delete warning)."""
    labels_dir = pdir / "labels"
    if not labels_dir.exists():
        return 0
    total = 0
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            head = line.split(maxsplit=1)[:1]
            if head:
                try:
                    if int(head[0]) == class_id:
                        total += 1
                except ValueError:
                    continue
    return total


def count_boxes_by_class(pdir: Path) -> dict[int, int]:
    """클래스 id → 박스 수. 라벨 파일을 **한 번만** 훑는다.

    `count_boxes_with_class` 는 클래스 하나를 위해 전부 훑으므로, 목록 화면처럼
    모든 클래스의 수가 필요할 때 그것을 반복해 부르면 클래스 수만큼 훑게 된다.

    쓰이지 않은 클래스는 키가 없다 — 0 으로 채우는 것은 호출자 몫이다.
    """
    labels_dir = pdir / "labels"
    if not labels_dir.exists():
        return {}
    counts: dict[int, int] = {}
    for txt in labels_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            head = line.split(maxsplit=1)[:1]
            if not head:
                continue
            try:
                cid = int(head[0])
            except ValueError:
                continue
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def delete_class(pdir: Path, class_id: int) -> list[dict]:
    """Remove a class: drop its boxes, reindex higher ids across all label files,
    and rewrite classes.json contiguously. Returns the remaining classes."""
    classes = read_classes(pdir)
    if not any(c["id"] == class_id for c in classes):
        raise KeyError(class_id)

    labels_dir = pdir / "labels"
    if labels_dir.exists():
        for txt in labels_dir.glob("*.txt"):
            _reindex_label_file(txt, class_id)

    remaining = [
        {
            "id": c["id"] if c["id"] < class_id else c["id"] - 1,
            "name": c["name"],
            "sources": c.get("sources", []),
        }
        for c in classes
        if c["id"] != class_id
    ]
    write_classes(pdir, remaining)
    return remaining
