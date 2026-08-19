"""외부 YOLO 데이터셋 zip 을 우리 레이아웃으로 들여온다.

받는 zip 은 출처가 제각각이다 — `names` 가 list 이기도 dict 이기도 하고, 라벨이
`labels/` 아래 어디쯤 있고, 클래스 id 는 그쪽 번호다. 여기서 하는 일은 셋이다.

    클래스 이름을 레지스트리에 병합 → local id → 우리 id 매핑
    이미지를 raw/ 로 복사 (또는 타일로 쪼개서)
    라벨을 그 매핑으로 다시 써서 labels/ 로

**HTTP 를 모른다.** 형식이 틀리면 `YoloImportError` 를 던지고, 그것을 몇 번 응답으로
옮길지는 호출자가 정한다.

들어가는 자리는 `dest / {raw,labels,classes.json}` 이다 — 데이터셋 디렉터리가 곧
그 모양이라 데이터셋을 그대로 넘기면 된다.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from lib.formats import IMAGE_EXTS
from lib.labels.io import atomic_write_text, read_label_file, write_label_file
from lib.labels.registry import ClassRegistry
from lib.media.tiling import TilingParams, clip_boxes_to_tile, tile_grid, tile_stem


class YoloImportError(Exception):
    """zip 이 YOLO 데이터셋으로 쓸 수 없는 형태다. 메시지는 사용자에게 보인다."""


def read_yaml_names(extract: Path) -> dict[int, str]:
    """data.yaml 의 클래스 이름 (list · dict 두 형태를 모두 받는다)."""
    yaml_path = None
    for candidate in ("data.yaml", "data.yml"):
        hits = sorted(extract.rglob(candidate))
        if hits:
            yaml_path = hits[0]
            break
    if yaml_path is None:
        raise YoloImportError(
            "data.yaml not found in the zip (a YOLO-format dataset is required)"
        )
    data = yaml.safe_load(yaml_path.read_text()) or {}
    raw = data.get("names")
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    return {}


def remap_label_text(path: Path, mapping: dict[int, int]) -> str:
    """라벨 파일의 클래스 id 를 `mapping`(저쪽 id → 우리 id)으로 다시 쓴다."""
    lines: list[str] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            old = int(parts[0])
        except ValueError:
            continue
        parts[0] = str(mapping.get(old, old))
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def _import_tiled(
    img: Path,
    lbl: Path | None,
    mapping: dict[int, int],
    raw_dir: Path,
    labels_dir: Path,
    tiling: TilingParams,
) -> tuple[int, int]:
    """이미지 한 장을 타일로 쪼개 저장한다. `(저장한 타일 수, 라벨 쓴 타일 수)`.

    라벨은 자동 변환한다 — 타일에 들어온 박스의 가시 비율이 min_visibility 이상이면
    타일 경계로 클립해 유지하고, 미만이면 버린다. drop_empty 면 (라벨 있는 원본에서)
    박스가 하나도 안 남은 타일은 저장하지 않는다.
    """
    import cv2

    frame = cv2.imread(str(img))
    if frame is None:
        return 0, 0
    img_h, img_w = frame.shape[:2]

    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    if lbl is not None:
        boxes = [(mapping.get(c, c), xyxy) for c, xyxy in read_label_file(lbl)]

    saved = labeled = 0
    for col, row, tx, ty in tile_grid(img_w, img_h, tiling):
        tile_boxes = clip_boxes_to_tile(boxes, img_w, img_h, tx, ty, tiling)
        if lbl is not None and tiling.drop_empty and not tile_boxes:
            continue
        stem = tile_stem(img.stem, col, row)
        crop = frame[ty : ty + tiling.tile_size, tx : tx + tiling.tile_size]
        cv2.imwrite(str(raw_dir / f"{stem}.jpg"), crop)
        saved += 1
        if lbl is not None:
            write_label_file(labels_dir / f"{stem}.txt", tile_boxes)
            labeled += 1
    return saved, labeled


def import_zip(tmp_zip: Path, dest: Path, tiling: TilingParams | None = None) -> dict:
    """zip 을 `dest/{raw,labels}` 로 들여오고 클래스를 `dest/classes.json` 에 병합한다.

    이미 있는 이름의 이미지는 **덮어쓴다** — 같은 zip 을 두 번 넣으면 두 벌이 아니라
    한 벌이다.
    """
    with tempfile.TemporaryDirectory() as td:
        extract = Path(td)
        try:
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(extract)
        except zipfile.BadZipFile as e:
            raise YoloImportError("Corrupted zip file") from e

        names = read_yaml_names(extract)

        classes_path = dest / "classes.json"
        if classes_path.exists():
            registry = ClassRegistry.from_dict(json.loads(classes_path.read_text()))
        else:
            registry = ClassRegistry()
        mapping = registry.add_model("import", names) if names else {}

        # YOLO 는 라벨을 `labels/` 아래 둔다 — 이미지 stem 으로 짝을 짓는다
        label_files: dict[str, Path] = {}
        for p in extract.rglob("*.txt"):
            if any(part.lower() == "labels" for part in p.parts):
                label_files.setdefault(p.stem, p)

        raw_dir = dest / "raw"
        labels_dir = dest / "labels"
        raw_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        added_images = added_labels = 0
        for img in sorted(extract.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or img.name.startswith("."):
                continue
            lbl = label_files.get(img.stem)
            if tiling is not None:
                saved, labeled = _import_tiled(img, lbl, mapping, raw_dir, labels_dir, tiling)
                added_images += saved
                added_labels += labeled
                continue
            shutil.copyfile(img, raw_dir / img.name)
            added_images += 1
            if lbl is not None:
                atomic_write_text(labels_dir / f"{img.stem}.txt", remap_label_text(lbl, mapping))
                added_labels += 1

        atomic_write_text(classes_path, json.dumps(registry.to_dict(), indent=2))
        return {
            "images": added_images,
            "labeled": added_labels,
            "classes": len(registry.classes),
        }
