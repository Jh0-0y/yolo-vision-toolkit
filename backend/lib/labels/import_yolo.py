"""외부 YOLO 데이터셋 zip 을 우리 레이아웃으로 들여온다.

받는 zip 은 출처가 제각각이다 — `names` 가 list 이기도 dict 이기도 하고, 라벨이
`labels/` 아래 어디쯤 있고, 클래스 id 는 그쪽 번호다. 여기서 하는 일은 셋이다.

    클래스 이름을 레지스트리에 병합 → local id → 우리 id 매핑
    이미지를 raw/ 로 복사
    라벨을 그 매핑으로 다시 써서 labels/ 로
    폴더 구조(train/val/test)에서 split 을 읽어 함께 돌려주기

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
from lib.fsutil import safe_stem, unique_stem
from lib.labels.io import atomic_write_text
from lib.labels.registry import ClassRegistry


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


# zip 안에서 split 을 나타내는 폴더 이름. `valid` 는 Roboflow 가 쓰는 이름이다.
_SPLIT_DIRS = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test",
}


def split_of(rel: Path) -> str | None:
    """이미지의 zip 내 상대경로에서 split 을 읽는다. 못 읽으면 None.

    배치가 출처마다 다르다 — `images/train/x.jpg` 도 있고 `train/images/x.jpg` 도
    있다. 그래서 자리를 못 박지 않고 **경로 어딘가에 있는 split 이름**을 찾는다.
    """
    for part in rel.parts[:-1]:
        name = _SPLIT_DIRS.get(part.lower())
        if name is not None:
            return name
    return None


def import_zip(tmp_zip: Path, dest: Path) -> dict:
    """zip 을 `dest/{raw,labels}` 로 들여오고 클래스를 `dest/classes.json` 에 병합한다.

    이미 있는 이름의 이미지는 **덮어쓰지 않고 `(2)` 로 넣는다.** 이름이 같아도 다른
    데이터일 수 있어서, 무엇을 남길지는 사람이 보고 정한다. 그래서 같은 zip 을 두 번
    넣으면 두 벌이 된다 — 이름으로 `(2)` 를 검색해 지우면 된다.

    `splits` 에 zip 의 폴더 구조에서 읽은 `{stem: split}` 을 함께 돌려준다. 여기서는
    **읽기만 한다** — 그것을 실제 배정으로 쓸지는 호출자가 정한다(사용자가 "이미 검증된
    데이터"라고 했을 때만 쓴다).
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
        splits: dict[str, str] = {}
        stems: list[str] = []
        for img in sorted(extract.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or img.name.startswith("."):
                continue
            lbl = label_files.get(img.stem)
            # 저쪽 이름을 그대로 믿지 않는다 — 앞에 점이 있으면 목록에서 사라지고,
            # `#?%` 가 있으면 이미지 URL 이 깨진다. 영상 프레임과 **같은 규칙**을 쓴다.
            ext = img.suffix.lower()
            # 같은 이름이 이미 있으면 덮어쓰지 않고 `(2)` 로 넣는다 — 이름만 같고
            # 내용은 다른 것일 수 있어서, 지울지는 사람이 보고 정한다.
            stem = unique_stem(
                safe_stem(img.name, fallback="image"),
                lambda s: (raw_dir / f"{s}{ext}").exists(),
            )
            shutil.copyfile(img, raw_dir / f"{stem}{ext}")
            added_images += 1
            stems.append(stem)
            split = split_of(img.relative_to(extract))
            if split is not None:
                splits[stem] = split
            if lbl is not None:
                atomic_write_text(labels_dir / f"{stem}.txt", remap_label_text(lbl, mapping))
                added_labels += 1

        atomic_write_text(classes_path, json.dumps(registry.to_dict(), indent=2))
        return {
            "images": added_images,
            "labeled": added_labels,
            "classes": len(registry.classes),
            "stems": stems,
            "splits": splits,
        }
