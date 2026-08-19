"""데이터셋 → YOLO 트리 → zip. 순수 파일 IO — DB 도 HTTP 도 torch 도 모른다.

세 가지를 내보낸다. 담는 split 만 다르고 만드는 방식은 하나다.

    train   train + val  (학습에 바로 쓸 것)
    test    test         (평가용)
    all     검수완료 전부 (분할 무관, 한 벌로)

이미지는 **하드링크**로 펼친다. 원본을 복사하지 않으니 내보내기가 몇 번이든
디스크는 한 벌이고, 실제 바이트가 드는 것은 zip 을 만들 때뿐이다.

라벨은 링크하지 않고 **다시 쓴다** — 클래스를 걸러 id 를 0..n-1 로 다시 매기기
때문이다. 박스가 하나도 안 남아도 빈 파일을 쓴다: 빈 라벨은 "안 그렸다"가 아니라
**"여기엔 없다"는 학습 신호**다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from lib.formats import IMAGE_EXTS
from lib.fsutil import link_or_copy
from lib.labels.io import read_label_file, write_data_yaml, write_label_file

Emit = Callable[[dict], None]

# 내보내기 종류 → 담을 split 집합. None 은 "분할 무관, 검수완료 전부".
EXPORT_KINDS: dict[str, tuple[str, ...] | None] = {
    "train": ("train", "val"),
    "test": ("test",),
    "all": None,
}


class ExportError(Exception):
    """내보낼 것이 없거나 종류가 틀렸다."""


class ExportCancelled(Exception):
    """CANCEL 센티널이 보였다."""


def _images_by_stem(raw_dir: Path) -> dict[str, Path]:
    if not raw_dir.exists():
        return {}
    return {
        p.stem: p
        for p in sorted(raw_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    }


def select_stems(
    kind: str, reviewed: set[str], splits: dict[str, str]
) -> dict[str, str]:
    """내보낼 `{stem: 담길 split}`. `all` 은 전부 `train` 자리에 담는다.

    **검수완료가 아닌 것은 절대 나가지 않는다** — 사람이 확인하지 않은 라벨을
    학습·평가에 흘리지 않는다.
    """
    if kind not in EXPORT_KINDS:
        raise ExportError(f"Unknown export kind: {kind}")
    wanted = EXPORT_KINDS[kind]
    if wanted is None:
        return {stem: "train" for stem in sorted(reviewed)}
    return {
        stem: splits[stem]
        for stem in sorted(reviewed)
        if splits.get(stem) in wanted
    }


def _remap_label(
    src: Path, dst: Path, keep: dict[int, int] | None
) -> None:
    """라벨을 옮겨 쓴다. `keep` 이 있으면 그 클래스만 남기고 id 를 다시 매긴다."""
    boxes = read_label_file(src) if src.exists() else []
    if keep is not None:
        boxes = [(keep[c], xyxy) for c, xyxy in boxes if c in keep]
    write_label_file(dst, boxes)


def materialize(
    *,
    dataset_dir: Path,
    out_dir: Path,
    kind: str,
    reviewed: set[str],
    splits: dict[str, str],
    class_ids: list[int] | None = None,
    emit: Emit | None = None,
    cancel_path: Path | None = None,
) -> dict:
    """YOLO 트리를 `out_dir` 에 펼친다. **zip 은 만들지 않는다.**

    내보내기(zip)와 학습(디렉터리)이 필요로 하는 것이 여기까지 같다 — 학습은
    이 결과를 그대로 ultralytics 에 넘기고, 내보내기는 이어서 zip 으로 묶는다.

    `class_ids` 를 주면 그 클래스만 담고 id 를 0 부터 다시 매긴다. None 이면 전부.
    """
    def _tick(ev: dict) -> None:
        if emit is not None:
            emit(ev)

    def _check_cancel() -> None:
        if cancel_path is not None and cancel_path.exists():
            raise ExportCancelled()

    wanted = select_stems(kind, reviewed, splits)
    if not wanted:
        raise ExportError("Nothing to export — review and split some images first")

    images = _images_by_stem(dataset_dir / "raw")
    labels_dir = dataset_dir / "labels"

    all_classes = []
    classes_path = dataset_dir / "classes.json"
    if classes_path.exists():
        try:
            all_classes = json.loads(classes_path.read_text()).get("classes", [])
        except (json.JSONDecodeError, OSError):
            all_classes = []

    if class_ids is None:
        names = {int(c["id"]): c["name"] for c in all_classes}
        keep = None
    else:
        chosen = [c for c in all_classes if int(c["id"]) in set(class_ids)]
        keep = {int(c["id"]): i for i, c in enumerate(chosen)}
        names = {i: c["name"] for i, c in enumerate(chosen)}

    total = len(wanted)
    _tick({"phase": "start", "total": total})

    counts = {"train": 0, "val": 0, "test": 0}
    linked = copied = 0
    for i, (stem, split) in enumerate(sorted(wanted.items()), 1):
        _check_cancel()
        img = images.get(stem)
        if img is None:
            continue  # 이미지가 지워졌다 — 배정표에만 남은 줄
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        how = link_or_copy(img, out_dir / "images" / split / img.name)
        linked += how == "link"
        copied += how == "copy"
        _remap_label(labels_dir / f"{stem}.txt", out_dir / "labels" / split / f"{stem}.txt", keep)
        counts[split] += 1
        if i % 20 == 0 or i == total:
            _tick({"phase": "copy", "copied": i, "total": total})

    # val 이 비면 ultralytics 가 학습을 못 돈다 — train 을 가리켜 두면 최소한 돈다
    write_data_yaml(
        out_dir / "data.yaml",
        names,
        train="images/train",
        val="images/val" if counts["val"] else "images/train",
    )

    return {
        "kind": kind,
        "count": sum(counts.values()),
        **counts,
        "classes": len(names),
        "linked": linked,
        "copied": copied,
    }


def build(*, out_dir: Path, **kwargs) -> dict:
    """`materialize` 한 뒤 zip 으로 묶는다. 펼친 트리는 지운다.

    하드링크라 트리를 지워도 원본 이미지는 그대로다 — 남는 것은 zip 하나뿐이다.
    """
    result = materialize(out_dir=out_dir, **kwargs)
    emit = kwargs.get("emit")
    if emit is not None:
        emit({"phase": "zip"})
    zip_path = Path(shutil.make_archive(str(out_dir), "zip", root_dir=out_dir))
    shutil.rmtree(out_dir, ignore_errors=True)
    return {**result, "size_bytes": zip_path.stat().st_size, "zip": str(zip_path)}
